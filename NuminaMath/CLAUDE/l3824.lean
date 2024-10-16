import Mathlib

namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3824_382447

theorem imaginary_part_of_complex_fraction :
  let i : ℂ := Complex.I
  let z : ℂ := (1 - i) / (1 + i)
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3824_382447


namespace NUMINAMATH_CALUDE_earliest_meeting_time_l3824_382476

def kelly_lap_time : ℕ := 5
def rachel_lap_time : ℕ := 8
def mike_lap_time : ℕ := 10

theorem earliest_meeting_time :
  let lap_times := [kelly_lap_time, rachel_lap_time, mike_lap_time]
  Nat.lcm (Nat.lcm kelly_lap_time rachel_lap_time) mike_lap_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_earliest_meeting_time_l3824_382476


namespace NUMINAMATH_CALUDE_circle_equation_proof_l3824_382402

/-- The standard equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
def standard_circle_equation (h k r x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Given a circle with center (1, -2) and radius 6, its standard equation is (x-1)^2 + (y+2)^2 = 36 -/
theorem circle_equation_proof :
  ∀ x y : ℝ, standard_circle_equation 1 (-2) 6 x y ↔ (x - 1)^2 + (y + 2)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l3824_382402


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l3824_382490

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 5| = |x + 3| :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l3824_382490


namespace NUMINAMATH_CALUDE_num_configurations_eq_num_fight_choices_l3824_382436

/-- The number of squares on the game board. -/
def board_size : ℕ := 2011

/-- The starting position of the black checker. -/
def black_start : ℕ := 3

/-- The function that calculates the number of different final configurations. -/
def num_configurations (board_size : ℕ) (black_start : ℕ) : ℕ :=
  board_size - black_start + 1

/-- Theorem stating that the number of different final configurations
    is equal to the number of possible choices for the number of fights. -/
theorem num_configurations_eq_num_fight_choices :
  num_configurations board_size black_start = board_size - black_start + 1 := by
  sorry

#eval num_configurations board_size black_start

end NUMINAMATH_CALUDE_num_configurations_eq_num_fight_choices_l3824_382436


namespace NUMINAMATH_CALUDE_find_number_l3824_382411

theorem find_number : ∃ x : ℝ, ((x * 0.5 + 26.1) / 0.4) - 35 = 35 := by
  use 3.8
  sorry

end NUMINAMATH_CALUDE_find_number_l3824_382411


namespace NUMINAMATH_CALUDE_fractional_sum_zero_l3824_382487

theorem fractional_sum_zero (a b c k : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) (h4 : k ≠ 0) 
  (h5 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (k * (b - c)^2) + b / (k * (c - a)^2) + c / (k * (a - b)^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fractional_sum_zero_l3824_382487


namespace NUMINAMATH_CALUDE_parallel_line_length_l3824_382442

/-- A triangle with a base of 24 inches and a parallel line dividing it into two equal areas -/
structure DividedTriangle where
  /-- The length of the base of the triangle -/
  base : ℝ
  /-- The length of the parallel line dividing the triangle -/
  parallel_line : ℝ
  /-- The base of the triangle is 24 inches -/
  base_length : base = 24
  /-- The parallel line divides the triangle into two equal areas -/
  equal_areas : parallel_line^2 = (1/2) * base^2

/-- The length of the parallel line in the divided triangle is 12√2 -/
theorem parallel_line_length (t : DividedTriangle) : t.parallel_line = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_length_l3824_382442


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3824_382433

theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 + 5*x + 6 > 0} = {x : ℝ | -1 < x ∧ x < 6} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3824_382433


namespace NUMINAMATH_CALUDE_chinese_and_math_books_same_student_probability_l3824_382463

def num_books : ℕ := 4
def num_students : ℕ := 2

def has_chinese_book : Bool := true
def has_math_book : Bool := true

def books_per_student : ℕ := num_books / num_students

theorem chinese_and_math_books_same_student_probability :
  let total_distributions := (num_books.choose books_per_student)
  let favorable_distributions := 2  -- Number of ways Chinese and Math books can be together
  (favorable_distributions : ℚ) / total_distributions = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_chinese_and_math_books_same_student_probability_l3824_382463


namespace NUMINAMATH_CALUDE_safari_arrangement_l3824_382475

/-- Represents the number of animal pairs in the safari park -/
def num_pairs : ℕ := 6

/-- Calculates the number of ways to arrange animals with alternating genders -/
def arrange_animals : ℕ := sorry

/-- Theorem stating the number of ways to arrange the animals -/
theorem safari_arrangement :
  arrange_animals = 86400 := by sorry

end NUMINAMATH_CALUDE_safari_arrangement_l3824_382475


namespace NUMINAMATH_CALUDE_inequality_implies_range_l3824_382452

theorem inequality_implies_range (a : ℝ) : 
  (∀ x ∈ Set.Icc (0 : ℝ) (1/2), 4^x + x - a ≤ 3/2) → a ∈ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_range_l3824_382452


namespace NUMINAMATH_CALUDE_min_packs_needed_l3824_382415

def pack_sizes : List Nat := [8, 15, 30]

/-- The target number of cans to be purchased -/
def target_cans : Nat := 120

/-- A function to check if a combination of packs can achieve the target number of cans -/
def achieves_target (x y z : Nat) : Prop :=
  8 * x + 15 * y + 30 * z = target_cans

/-- The minimum number of packs needed -/
def min_packs : Nat := 4

theorem min_packs_needed : 
  (∃ x y z : Nat, achieves_target x y z) ∧ 
  (∀ x y z : Nat, achieves_target x y z → x + y + z ≥ min_packs) ∧
  (∃ x y z : Nat, achieves_target x y z ∧ x + y + z = min_packs) :=
sorry

end NUMINAMATH_CALUDE_min_packs_needed_l3824_382415


namespace NUMINAMATH_CALUDE_f_negative_two_l3824_382440

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x + x^3 + 1

theorem f_negative_two (a : ℝ) (h : f a 2 = 3) : f a (-2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_two_l3824_382440


namespace NUMINAMATH_CALUDE_smallest_digit_change_correct_change_l3824_382458

def original_sum : ℕ := 738 + 625 + 841
def incorrect_sum : ℕ := 2104
def correct_sum : ℕ := 2204

def change_digit (n : ℕ) (place : ℕ) (new_digit : ℕ) : ℕ :=
  n - (n / 10^place % 10) * 10^place + new_digit * 10^place

theorem smallest_digit_change :
  ∀ (d : ℕ),
    d < 6 →
    ¬∃ (n : ℕ) (place : ℕ),
      (n = 738 ∨ n = 625 ∨ n = 841) ∧
      change_digit n place d + 
        (if n = 738 then 625 + 841
         else if n = 625 then 738 + 841
         else 738 + 625) = correct_sum :=
by sorry

theorem correct_change :
  change_digit 625 2 5 + 738 + 841 = correct_sum :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_change_correct_change_l3824_382458


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_n_binomial_coefficient_1000_1000_l3824_382473

theorem binomial_coefficient_n_n (n : ℕ) : Nat.choose n n = 1 := by
  sorry

theorem binomial_coefficient_1000_1000 : Nat.choose 1000 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_n_binomial_coefficient_1000_1000_l3824_382473


namespace NUMINAMATH_CALUDE_not_balanced_numbers_l3824_382489

/-- Definition of balanced numbers with respect to l -/
def balanced (a b : ℝ) : Prop := a + b = 2

/-- Given equation -/
axiom equation : ∃ m : ℝ, (Real.sqrt 3 + m) * (Real.sqrt 3 - 1) = 2

/-- Theorem to prove -/
theorem not_balanced_numbers : ¬∃ m : ℝ, 
  (Real.sqrt 3 + m) * (Real.sqrt 3 - 1) = 2 ∧ 
  balanced (m + Real.sqrt 3) (2 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_not_balanced_numbers_l3824_382489


namespace NUMINAMATH_CALUDE_exists_perfect_pair_with_122_l3824_382429

/-- Two natural numbers form a perfect pair if their sum and product are both perfect squares. -/
def IsPerfectPair (a b : ℕ) : Prop :=
  ∃ (x y : ℕ), a + b = x^2 ∧ a * b = y^2

/-- There exists a natural number that forms a perfect pair with 122. -/
theorem exists_perfect_pair_with_122 : ∃ (n : ℕ), IsPerfectPair 122 n := by
  sorry

end NUMINAMATH_CALUDE_exists_perfect_pair_with_122_l3824_382429


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3824_382431

theorem solve_exponential_equation :
  ∃ x : ℝ, 16 = 4 * (4 : ℝ) ^ (x - 1) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3824_382431


namespace NUMINAMATH_CALUDE_log_cube_l3824_382409

theorem log_cube (x : ℝ) (h : Real.log x / Real.log 3 = 5) : 
  Real.log (x^3) / Real.log 3 = 15 := by
sorry

end NUMINAMATH_CALUDE_log_cube_l3824_382409


namespace NUMINAMATH_CALUDE_tom_walking_distance_l3824_382484

/-- Tom's walking rate in miles per minute -/
def walking_rate : ℚ := 2 / 36

/-- The time Tom walks in minutes -/
def walking_time : ℚ := 9

/-- The distance Tom walks in miles -/
def walking_distance : ℚ := walking_rate * walking_time

theorem tom_walking_distance :
  walking_distance = 1/2 := by sorry

end NUMINAMATH_CALUDE_tom_walking_distance_l3824_382484


namespace NUMINAMATH_CALUDE_square_sum_geq_linear_l3824_382432

theorem square_sum_geq_linear (a b : ℝ) : a^2 + b^2 ≥ 2*a - 2*b - 2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_linear_l3824_382432


namespace NUMINAMATH_CALUDE_triangle_abc_theorem_l3824_382445

open Real

theorem triangle_abc_theorem (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  sin A / a = sin B / b ∧ sin B / b = sin C / c →
  cos (2 * C) - cos (2 * A) = 2 * sin (π / 3 + C) * sin (π / 3 - C) →
  a = sqrt 3 →
  b ≥ a →
  A = π / 3 ∧ sqrt 3 ≤ 2 * b - c ∧ 2 * b - c < 2 * sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_theorem_l3824_382445


namespace NUMINAMATH_CALUDE_halloween_candy_weight_l3824_382461

/-- The combined weight of candy Frank and Gwen received for Halloween -/
def combined_candy_weight (frank_candy : ℕ) (gwen_candy : ℕ) : ℕ :=
  frank_candy + gwen_candy

/-- Theorem: The combined weight of candy Frank and Gwen received is 17 pounds -/
theorem halloween_candy_weight :
  combined_candy_weight 10 7 = 17 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_weight_l3824_382461


namespace NUMINAMATH_CALUDE_probability_at_least_one_shot_l3824_382426

/-- The probability of making at least one shot out of three, given a success rate of 3/5 for each shot. -/
theorem probability_at_least_one_shot (success_rate : ℝ) (num_shots : ℕ) : 
  success_rate = 3/5 → num_shots = 3 → 1 - (1 - success_rate)^num_shots = 0.936 := by
  sorry


end NUMINAMATH_CALUDE_probability_at_least_one_shot_l3824_382426


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3824_382492

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 7 + a 13 = 4 →
  a 2 + a 12 = 8/3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3824_382492


namespace NUMINAMATH_CALUDE_a_range_l3824_382477

theorem a_range (a : ℝ) : a < 9 * a^3 - 11 * a ∧ 9 * a^3 - 11 * a < |a| → -2 * Real.sqrt 3 / 3 < a ∧ a < -Real.sqrt 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l3824_382477


namespace NUMINAMATH_CALUDE_min_domain_length_l3824_382483

open Real

theorem min_domain_length (f : ℝ → ℝ) (m n : ℝ) :
  (∀ x ∈ Set.Icc m n, f x = sin x * sin (x + π/3) - 1/4) →
  m < n →
  Set.range f = Set.Icc (-1/2) (1/4) →
  n - m ≥ 2*π/3 :=
sorry

end NUMINAMATH_CALUDE_min_domain_length_l3824_382483


namespace NUMINAMATH_CALUDE_greatest_base7_digit_sum_l3824_382417

/-- Represents a base-7 digit (0 to 6) -/
def Base7Digit := Fin 7

/-- Represents a base-7 number as a list of digits -/
def Base7Number := List Base7Digit

/-- Converts a natural number to its base-7 representation -/
def toBase7 (n : ℕ) : Base7Number :=
  sorry

/-- Calculates the sum of digits in a base-7 number -/
def digitSum (num : Base7Number) : ℕ :=
  sorry

/-- Checks if a base-7 number is less than 1729 in decimal -/
def isLessThan1729 (num : Base7Number) : Prop :=
  sorry

theorem greatest_base7_digit_sum :
  ∃ (n : Base7Number), isLessThan1729 n ∧
    digitSum n = 22 ∧
    ∀ (m : Base7Number), isLessThan1729 m → digitSum m ≤ 22 :=
  sorry

end NUMINAMATH_CALUDE_greatest_base7_digit_sum_l3824_382417


namespace NUMINAMATH_CALUDE_mike_pears_count_l3824_382469

/-- The number of pears picked by Jason -/
def jason_pears : ℕ := 46

/-- The number of pears picked by Keith -/
def keith_pears : ℕ := 47

/-- The total number of pears picked -/
def total_pears : ℕ := 105

/-- The number of pears picked by Mike -/
def mike_pears : ℕ := total_pears - (jason_pears + keith_pears)

theorem mike_pears_count : mike_pears = 12 := by
  sorry

end NUMINAMATH_CALUDE_mike_pears_count_l3824_382469


namespace NUMINAMATH_CALUDE_vehicle_value_theorem_l3824_382444

def vehicle_value_last_year : ℝ := 20000

def depreciation_factor : ℝ := 0.8

def vehicle_value_this_year : ℝ := depreciation_factor * vehicle_value_last_year

theorem vehicle_value_theorem : vehicle_value_this_year = 16000 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_value_theorem_l3824_382444


namespace NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l3824_382479

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℕ → ℝ :=
  fun k => a₁ + (k - 1 : ℝ) * d

theorem third_term_of_arithmetic_sequence :
  ∀ (a₁ aₙ : ℝ) (n : ℕ),
  n = 10 →
  a₁ = 5 →
  aₙ = 32 →
  let d := (aₙ - a₁) / (n - 1 : ℝ)
  let seq := arithmetic_sequence a₁ d n
  seq 3 = 11 := by
sorry

end NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l3824_382479


namespace NUMINAMATH_CALUDE_min_value_sum_squared_ratios_l3824_382460

theorem min_value_sum_squared_ratios (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b)^2 + (b / c)^2 + (c / a)^2 ≥ 3 ∧
  ((a / b)^2 + (b / c)^2 + (c / a)^2 = 3 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squared_ratios_l3824_382460


namespace NUMINAMATH_CALUDE_theater_ticket_cost_is_3320_l3824_382491

/-- Calculates the total cost of theater tickets sold given the following conditions:
    - Total tickets sold: 370
    - Orchestra ticket price: $12
    - Balcony ticket price: $8
    - 190 more balcony tickets sold than orchestra tickets
-/
def theater_ticket_cost : ℕ := by
  -- Define the total number of tickets sold
  let total_tickets : ℕ := 370
  -- Define the price of orchestra tickets
  let orchestra_price : ℕ := 12
  -- Define the price of balcony tickets
  let balcony_price : ℕ := 8
  -- Define the difference between balcony and orchestra tickets sold
  let balcony_orchestra_diff : ℕ := 190
  
  -- Calculate the number of orchestra tickets sold
  let orchestra_tickets : ℕ := (total_tickets - balcony_orchestra_diff) / 2
  -- Calculate the number of balcony tickets sold
  let balcony_tickets : ℕ := total_tickets - orchestra_tickets
  
  -- Calculate and return the total cost
  exact orchestra_price * orchestra_tickets + balcony_price * balcony_tickets

/-- Theorem stating that the total cost of theater tickets is $3320 -/
theorem theater_ticket_cost_is_3320 : theater_ticket_cost = 3320 := by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_cost_is_3320_l3824_382491


namespace NUMINAMATH_CALUDE_no_x_squared_term_l3824_382465

theorem no_x_squared_term (p : ℚ) : 
  (∀ x, (x^2 + p*x) * (x^2 - 3*x + 1) = x^4 + (p-3)*x^3 + 0*x^2 + p*x) → p = 1/3 := by
sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l3824_382465


namespace NUMINAMATH_CALUDE_minimum_bottles_needed_l3824_382438

def large_bottle_capacity : ℕ := 450
def small_bottle_capacity : ℕ := 45
def extra_large_bottle_capacity : ℕ := 900

theorem minimum_bottles_needed :
  ∃ (large_count small_count : ℕ),
    large_count * large_bottle_capacity + small_count * small_bottle_capacity = extra_large_bottle_capacity ∧
    large_count + small_count = 2 ∧
    ∀ (l s : ℕ), l * large_bottle_capacity + s * small_bottle_capacity = extra_large_bottle_capacity →
      l + s ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_minimum_bottles_needed_l3824_382438


namespace NUMINAMATH_CALUDE_equation_solution_l3824_382468

theorem equation_solution (x : ℝ) : 
  Real.sqrt (9 + Real.sqrt (27 + 3*x)) + Real.sqrt (3 + Real.sqrt (9 + x)) = 3 + 3 * Real.sqrt 3 →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3824_382468


namespace NUMINAMATH_CALUDE_saras_quarters_l3824_382449

/-- The number of quarters Sara has after receiving some from her dad -/
def total_quarters (initial_quarters given_quarters : ℝ) : ℝ :=
  initial_quarters + given_quarters

/-- Theorem stating that Sara's total quarters is the sum of her initial quarters and those given by her dad -/
theorem saras_quarters (initial_quarters given_quarters : ℝ) :
  total_quarters initial_quarters given_quarters = initial_quarters + given_quarters :=
by sorry

end NUMINAMATH_CALUDE_saras_quarters_l3824_382449


namespace NUMINAMATH_CALUDE_volume_ratio_cylinder_sphere_cone_l3824_382421

/-- Given a square ABCD with side length 2a and an inscribed circle O tangent to AB at Q and CD at P,
    and an isosceles triangle PAB, prove that the ratio of volumes of the cylinder, sphere, and cone
    formed by rotating these shapes around the axis of symmetry PQ is 3 : 2 : 1. -/
theorem volume_ratio_cylinder_sphere_cone (a : ℝ) (h : a > 0) :
  ∃ (v_cylinder v_sphere v_cone : ℝ),
    v_cylinder = 2 * Real.pi * a^3 ∧
    v_sphere = (4/3) * Real.pi * a^3 ∧
    v_cone = (2/3) * Real.pi * a^3 ∧
    v_cylinder / v_sphere = 3/2 ∧
    v_cylinder / v_cone = 3 ∧
    v_sphere / v_cone = 2 :=
by sorry

end NUMINAMATH_CALUDE_volume_ratio_cylinder_sphere_cone_l3824_382421


namespace NUMINAMATH_CALUDE_center_C_range_l3824_382472

-- Define the points and line
def A : ℝ × ℝ := (0, 3)
def l (x : ℝ) : ℝ := 2 * x - 4

-- Define circle C
def C (a : ℝ) : ℝ × ℝ := (a, l a)
def radius_C : ℝ := 1

-- Define moving point M
def M : ℝ × ℝ → Prop := λ (x, y) => (x^2 + (y - 3)^2) = 4 * (x^2 + y^2)

-- Define the intersection condition
def intersects (C : ℝ × ℝ) (M : ℝ × ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), M (x, y) ∧ (x - C.1)^2 + (y - C.2)^2 = radius_C^2

-- Theorem statement
theorem center_C_range (a : ℝ) :
  (C a).2 = l (C a).1 →  -- Center of C lies on line l
  intersects (C a) M →   -- M intersects with C
  0 ≤ a ∧ a ≤ 12/5 :=
by sorry

end NUMINAMATH_CALUDE_center_C_range_l3824_382472


namespace NUMINAMATH_CALUDE_prop_A_prop_B_l3824_382493

-- Define the function f(x) = (x-2)^2
def f (x : ℝ) : ℝ := (x - 2)^2

-- Proposition A: f(x+2) is an even function
theorem prop_A : ∀ x : ℝ, f (x + 2) = f (-x + 2) := by sorry

-- Proposition B: f(x) is decreasing on (-∞, 2) and increasing on (2, +∞)
theorem prop_B :
  (∀ x y : ℝ, x < y → y < 2 → f y < f x) ∧
  (∀ x y : ℝ, 2 < x → x < y → f x < f y) := by sorry

end NUMINAMATH_CALUDE_prop_A_prop_B_l3824_382493


namespace NUMINAMATH_CALUDE_remaining_seeds_l3824_382457

def initial_seeds : ℝ := 8.75
def sowed_seeds : ℝ := 2.75

theorem remaining_seeds :
  initial_seeds - sowed_seeds = 6 := by sorry

end NUMINAMATH_CALUDE_remaining_seeds_l3824_382457


namespace NUMINAMATH_CALUDE_triangle_abc_theorem_l3824_382412

noncomputable section

variables {a b c : ℝ} {A B C : ℝ} {O P : ℝ × ℝ}

def triangle_abc (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

def angle_condition (a b c A B C : ℝ) : Prop :=
  a * Real.sin A + a * Real.sin C * Real.cos B + b * Real.sin C * Real.cos A = 
  b * Real.sin B + c * Real.sin A

def acute_triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2

def circumradius (a b c : ℝ) : ℝ :=
  (a * b * c) / (4 * Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)))

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem triangle_abc_theorem (a b c A B C : ℝ) (O P : ℝ × ℝ) :
  triangle_abc a b c →
  angle_condition a b c A B C →
  (a = 2 → acute_triangle A B C → 
    3 + Real.sqrt 3 < a + b + c ∧ a + b + c < 6 + 2 * Real.sqrt 3) →
  (b^2 = a*c → circumradius a b c = 2 → 
    -2 ≤ dot_product (P.1 - O.1, P.2 - O.2) (P.1 - O.1 - a, P.2 - O.2) ∧
    dot_product (P.1 - O.1, P.2 - O.2) (P.1 - O.1 - a, P.2 - O.2) ≤ 6) →
  B = Real.pi / 3 := by sorry

end

end NUMINAMATH_CALUDE_triangle_abc_theorem_l3824_382412


namespace NUMINAMATH_CALUDE_book_reading_time_l3824_382416

/-- The number of weeks required to read a book -/
def weeks_to_read (total_pages : ℕ) (pages_per_week : ℕ) : ℕ :=
  (total_pages + pages_per_week - 1) / pages_per_week

theorem book_reading_time : 
  let total_pages : ℕ := 2100
  let pages_per_day1 : ℕ := 100
  let pages_per_day2 : ℕ := 150
  let days_type1 : ℕ := 3
  let days_type2 : ℕ := 2
  let pages_per_week : ℕ := pages_per_day1 * days_type1 + pages_per_day2 * days_type2
  weeks_to_read total_pages pages_per_week = 4 := by
sorry

end NUMINAMATH_CALUDE_book_reading_time_l3824_382416


namespace NUMINAMATH_CALUDE_tv_sales_decrease_l3824_382434

theorem tv_sales_decrease (original_price original_quantity : ℝ) 
  (h_price_increase : ℝ) (h_revenue_increase : ℝ) :
  original_price > 0 →
  original_quantity > 0 →
  h_price_increase = 0.6 →
  h_revenue_increase = 0.28 →
  let new_price := original_price * (1 + h_price_increase)
  let new_revenue := (1 + h_revenue_increase) * (original_price * original_quantity)
  let sales_decrease := 1 - (new_revenue / (new_price * original_quantity))
  sales_decrease = 0.2 := by
sorry

end NUMINAMATH_CALUDE_tv_sales_decrease_l3824_382434


namespace NUMINAMATH_CALUDE_na_minimum_at_3_l3824_382495

-- Define the sequence S_n
def S (n : ℕ) : ℤ := n^2 - 10*n

-- Define a_n as the difference between consecutive S_n terms
def a (n : ℕ) : ℤ := S n - S (n-1)

-- Define na_n
def na (n : ℕ) : ℤ := n * (a n)

-- Theorem statement
theorem na_minimum_at_3 :
  ∀ k : ℕ, k ≥ 1 → na 3 ≤ na k :=
sorry

end NUMINAMATH_CALUDE_na_minimum_at_3_l3824_382495


namespace NUMINAMATH_CALUDE_carol_initial_amount_l3824_382418

/-- Carol's initial amount of money -/
def carol_initial : ℕ := sorry

/-- Carol's weekly savings -/
def carol_weekly_savings : ℕ := 9

/-- Mike's initial amount of money -/
def mike_initial : ℕ := 90

/-- Mike's weekly savings -/
def mike_weekly_savings : ℕ := 3

/-- Number of weeks -/
def weeks : ℕ := 5

theorem carol_initial_amount :
  carol_initial = 60 :=
by
  have h1 : carol_initial + weeks * carol_weekly_savings = mike_initial + weeks * mike_weekly_savings :=
    sorry
  sorry

end NUMINAMATH_CALUDE_carol_initial_amount_l3824_382418


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l3824_382413

-- Define the total number of votes
def total_votes : ℕ := 560000

-- Define the percentage of valid votes received by candidate A
def candidate_A_percentage : ℚ := 55 / 100

-- Define the number of valid votes received by candidate A
def candidate_A_votes : ℕ := 261800

-- Define the percentage of invalid votes
def invalid_vote_percentage : ℚ := 15 / 100

-- Theorem statement
theorem invalid_votes_percentage :
  (1 - (candidate_A_votes : ℚ) / (candidate_A_percentage * total_votes)) = invalid_vote_percentage := by
  sorry

end NUMINAMATH_CALUDE_invalid_votes_percentage_l3824_382413


namespace NUMINAMATH_CALUDE_probability_of_shared_character_l3824_382471

/-- Represents an idiom card -/
structure IdiomCard where
  idiom : String

/-- The set of all idiom cards -/
def idiomCards : Finset IdiomCard := sorry

/-- Two cards share a character -/
def shareCharacter (card1 card2 : IdiomCard) : Prop := sorry

/-- The number of ways to choose 2 cards from the set -/
def totalChoices : Nat := Nat.choose idiomCards.card 2

/-- The number of ways to choose 2 cards that share a character -/
def favorableChoices : Nat := sorry

theorem probability_of_shared_character :
  (favorableChoices : ℚ) / totalChoices = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_of_shared_character_l3824_382471


namespace NUMINAMATH_CALUDE_increasing_iff_a_gt_two_l3824_382414

-- Define the linear function
def f (a x : ℝ) : ℝ := (2*a - 4)*x + 3

-- State the theorem
theorem increasing_iff_a_gt_two :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a > 2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_iff_a_gt_two_l3824_382414


namespace NUMINAMATH_CALUDE_initial_books_l3824_382443

theorem initial_books (total : ℕ) (additional : ℕ) (initial : ℕ) : 
  total = 77 → additional = 23 → total = initial + additional → initial = 54 := by
sorry

end NUMINAMATH_CALUDE_initial_books_l3824_382443


namespace NUMINAMATH_CALUDE_blue_ball_count_l3824_382464

/-- The number of balls of each color in a box --/
structure BallCounts where
  blue : ℕ
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- The conditions of the ball counting problem --/
def ballProblem (counts : BallCounts) : Prop :=
  counts.red = 4 ∧
  counts.green = 3 * counts.blue ∧
  counts.yellow = 2 * counts.red ∧
  counts.blue + counts.red + counts.green + counts.yellow = 36

theorem blue_ball_count :
  ∃ (counts : BallCounts), ballProblem counts ∧ counts.blue = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_ball_count_l3824_382464


namespace NUMINAMATH_CALUDE_cory_fruit_arrangements_l3824_382446

/-- The number of ways to arrange indistinguishable objects of different types -/
def multinomial_coefficient (n : ℕ) (ks : List ℕ) : ℕ :=
  Nat.factorial n / (List.prod (List.map Nat.factorial ks))

/-- The number of distinct arrangements of Cory's fruit -/
theorem cory_fruit_arrangements :
  let total_fruit : ℕ := 7
  let fruit_counts : List ℕ := [3, 2, 2]
  multinomial_coefficient total_fruit fruit_counts = 210 := by
  sorry

end NUMINAMATH_CALUDE_cory_fruit_arrangements_l3824_382446


namespace NUMINAMATH_CALUDE_harmonic_mean_inequality_l3824_382459

theorem harmonic_mean_inequality (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 2) :
  1/m + 1/n ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_harmonic_mean_inequality_l3824_382459


namespace NUMINAMATH_CALUDE_billy_ice_cubes_l3824_382441

/-- Calculates the total number of ice cubes that can be made given the tray capacity and number of trays. -/
def total_ice_cubes (tray_capacity : ℕ) (num_trays : ℕ) : ℕ :=
  tray_capacity * num_trays

/-- Proves that with a tray capacity of 48 ice cubes and 24 trays, the total number of ice cubes is 1152. -/
theorem billy_ice_cubes : total_ice_cubes 48 24 = 1152 := by
  sorry

end NUMINAMATH_CALUDE_billy_ice_cubes_l3824_382441


namespace NUMINAMATH_CALUDE_intersection_M_N_l3824_382439

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3824_382439


namespace NUMINAMATH_CALUDE_polygon_sides_proof_l3824_382456

theorem polygon_sides_proof (x y : ℕ) : 
  (x - 2) * 180 + (y - 2) * 180 = 21 * (x + y + x * (x - 3) / 2 + y * (y - 3) / 2) - 39 →
  x * (x - 3) / 2 + y * (y - 3) / 2 - (x + y) = 99 →
  ((x = 17 ∧ y = 3) ∨ (x = 3 ∧ y = 17)) :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_proof_l3824_382456


namespace NUMINAMATH_CALUDE_square_area_12cm_l3824_382428

/-- The area of a square with side length 12 cm is 144 square centimeters. -/
theorem square_area_12cm (s : ℝ) (h : s = 12) : s^2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_square_area_12cm_l3824_382428


namespace NUMINAMATH_CALUDE_t_shirt_problem_l3824_382454

/-- Represents a t-shirt package with its size and price -/
structure Package where
  size : Nat
  price : Rat

/-- Calculates the total number of t-shirts and the discounted price -/
def calculate_total (small medium large : Package) 
                    (small_qty medium_qty large_qty : Nat) : Nat × Rat :=
  let total_shirts := small.size * small_qty + medium.size * medium_qty + large.size * large_qty
  let total_price := small.price * small_qty + medium.price * medium_qty + large.price * large_qty
  let total_packages := small_qty + medium_qty + large_qty
  let discounted_price := if total_packages > 25 
                          then total_price * (1 - 5 / 100) 
                          else total_price
  (total_shirts, discounted_price)

theorem t_shirt_problem :
  let small : Package := ⟨6, 12⟩
  let medium : Package := ⟨12, 20⟩
  let large : Package := ⟨20, 30⟩
  let (total_shirts, discounted_price) := calculate_total small medium large 15 10 4
  total_shirts = 290 ∧ discounted_price = 475 := by
  sorry

end NUMINAMATH_CALUDE_t_shirt_problem_l3824_382454


namespace NUMINAMATH_CALUDE_max_product_l3824_382450

def digits : List ℕ := [1, 3, 5, 8, 9]

def is_valid_combination (a b c d e : ℕ) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

def product (a b c d e : ℕ) : ℕ := (100 * a + 10 * b + c) * (10 * d + e)

theorem max_product :
  ∀ a b c d e : ℕ,
    is_valid_combination a b c d e →
    product a b c d e ≤ product 9 3 1 8 5 :=
by sorry

end NUMINAMATH_CALUDE_max_product_l3824_382450


namespace NUMINAMATH_CALUDE_rabbit_count_l3824_382486

/-- Given a total number of heads and a relationship between rabbit and chicken feet,
    prove the number of rabbits. -/
theorem rabbit_count (total_heads : ℕ) (rabbit_feet chicken_feet : ℕ → ℕ) : 
  total_heads = 40 →
  (∀ x, rabbit_feet x = 10 * chicken_feet (total_heads - x) - 8) →
  (∃ x, x = 33 ∧ 
        rabbit_feet x = 4 * x ∧ 
        chicken_feet (total_heads - x) = 2 * (total_heads - x)) :=
by sorry

end NUMINAMATH_CALUDE_rabbit_count_l3824_382486


namespace NUMINAMATH_CALUDE_parallel_lines_theorem_l3824_382400

/-- Two lines are parallel if their slopes are equal -/
def parallel_lines (m1 m2 : ℝ) : Prop := m1 = m2

/-- Slope of the line ax + y - 1 - a = 0 -/
def slope1 (a : ℝ) : ℝ := -a

/-- Slope of the line x - 1/2y = 0 -/
def slope2 : ℝ := 2

/-- Theorem: If ax + y - 1 - a = 0 is parallel to x - 1/2y = 0, then a = -2 -/
theorem parallel_lines_theorem (a : ℝ) : 
  parallel_lines (slope1 a) slope2 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_theorem_l3824_382400


namespace NUMINAMATH_CALUDE_sum_equals_four_l3824_382494

/-- Custom binary operation on real numbers -/
def custom_op (x y : ℝ) : ℝ := x * (1 - y)

/-- The solution set of the inequality -/
def solution_set : Set ℝ := Set.Ioo 2 3

/-- Theorem stating the sum of a and b equals 4 -/
theorem sum_equals_four (a b : ℝ) 
  (h : ∀ x ∈ solution_set, custom_op (x - a) (x - b) > 0) 
  (h_unique : ∀ x ∉ solution_set, custom_op (x - a) (x - b) ≤ 0) : 
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_four_l3824_382494


namespace NUMINAMATH_CALUDE_empty_solution_set_range_min_value_distance_sum_l3824_382498

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 2| > a^2 + a + 1) ↔ (-1 < a ∧ a < 0) := by
  sorry

theorem min_value_distance_sum : 
  ∀ x : ℝ, |x - 1| + |x - 2| ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_min_value_distance_sum_l3824_382498


namespace NUMINAMATH_CALUDE_solution_set_f_leq_5_range_of_m_l3824_382401

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x + 3| + |2*x - 1|

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set_f_leq_5 :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -7/4 ≤ x ∧ x ≤ 3/4} := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, f x < |m - 2|) → (m > 6 ∨ m < -2) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_5_range_of_m_l3824_382401


namespace NUMINAMATH_CALUDE_product_trailing_zeroes_l3824_382481

/-- The number of trailing zeroes in a positive integer -/
def trailingZeroes (n : ℕ) : ℕ := sorry

/-- The product of 25^5, 150^4, and 2008^3 -/
def largeProduct : ℕ := 25^5 * 150^4 * 2008^3

theorem product_trailing_zeroes :
  trailingZeroes largeProduct = 13 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeroes_l3824_382481


namespace NUMINAMATH_CALUDE_sequence_a_bounds_l3824_382427

def sequence_a : ℕ → ℚ
  | 0 => 1/2
  | n+1 => sequence_a n + (1 : ℚ)/(n+1)^2 * (sequence_a n)^2

theorem sequence_a_bounds : ∀ n : ℕ, (n+1 : ℚ)/(n+2) < sequence_a n ∧ sequence_a n < n+1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_bounds_l3824_382427


namespace NUMINAMATH_CALUDE_pamphlets_total_l3824_382488

/-- Calculates the total number of pamphlets printed by Mike and Leo -/
def total_pamphlets (mike_initial_speed : ℕ) (mike_initial_hours : ℕ) (mike_final_hours : ℕ) : ℕ :=
  let mike_initial_pamphlets := mike_initial_speed * mike_initial_hours
  let mike_final_speed := mike_initial_speed / 3
  let mike_final_pamphlets := mike_final_speed * mike_final_hours
  let leo_hours := mike_initial_hours / 3
  let leo_speed := mike_initial_speed * 2
  let leo_pamphlets := leo_speed * leo_hours
  mike_initial_pamphlets + mike_final_pamphlets + leo_pamphlets

/-- Theorem stating that the total number of pamphlets printed is 9400 -/
theorem pamphlets_total : total_pamphlets 600 9 2 = 9400 := by
  sorry

end NUMINAMATH_CALUDE_pamphlets_total_l3824_382488


namespace NUMINAMATH_CALUDE_julian_airplane_models_l3824_382405

theorem julian_airplane_models : 
  ∀ (total_legos : ℕ) (legos_per_model : ℕ) (additional_legos_needed : ℕ),
    total_legos = 400 →
    legos_per_model = 240 →
    additional_legos_needed = 80 →
    (total_legos + additional_legos_needed) / legos_per_model = 2 := by
  sorry

end NUMINAMATH_CALUDE_julian_airplane_models_l3824_382405


namespace NUMINAMATH_CALUDE_final_position_total_consumption_l3824_382466

-- Define the list of mileage values
def mileage : List Int := [-6, -2, 8, -3, 6, -4, 6, 3]

-- Define the electricity consumption rate per kilometer
def consumption_rate : Float := 0.15

-- Theorem for the final position
theorem final_position (m : List Int := mileage) :
  m.sum = 8 := by sorry

-- Theorem for total electricity consumption
theorem total_consumption (m : List Int := mileage) (r : Float := consumption_rate) :
  (m.map Int.natAbs).sum.toFloat * r = 5.7 := by sorry

end NUMINAMATH_CALUDE_final_position_total_consumption_l3824_382466


namespace NUMINAMATH_CALUDE_gwen_book_collection_l3824_382420

/-- Represents the number of books in Gwen's collection --/
def total_books : ℕ :=
  let mystery_shelves : ℕ := 8
  let mystery_books_per_shelf : ℕ := 6
  let picture_shelves : ℕ := 5
  let picture_books_per_shelf : ℕ := 4
  let scifi_shelves : ℕ := 4
  let scifi_books_per_shelf : ℕ := 7
  let nonfiction_shelves : ℕ := 3
  let nonfiction_books_per_shelf : ℕ := 5
  let mystery_books_lent : ℕ := 2
  let scifi_books_lent : ℕ := 3
  let picture_books_borrowed : ℕ := 5

  let mystery_total := mystery_shelves * mystery_books_per_shelf - mystery_books_lent
  let picture_total := picture_shelves * picture_books_per_shelf
  let scifi_total := scifi_shelves * scifi_books_per_shelf - scifi_books_lent
  let nonfiction_total := nonfiction_shelves * nonfiction_books_per_shelf

  mystery_total + picture_total + scifi_total + nonfiction_total

theorem gwen_book_collection : total_books = 106 := by
  sorry

end NUMINAMATH_CALUDE_gwen_book_collection_l3824_382420


namespace NUMINAMATH_CALUDE_sam_distance_l3824_382455

/-- Given Marguerite's cycling distance and time, and Sam's cycling time,
    prove that Sam's distance is equal to (Marguerite's distance / Marguerite's time) * Sam's time,
    assuming they cycle at the same average speed. -/
theorem sam_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ)
    (h1 : marguerite_distance > 0)
    (h2 : marguerite_time > 0)
    (h3 : sam_time > 0) :
  let sam_distance := (marguerite_distance / marguerite_time) * sam_time
  sam_distance = (marguerite_distance / marguerite_time) * sam_time :=
by
  sorry

#check sam_distance

end NUMINAMATH_CALUDE_sam_distance_l3824_382455


namespace NUMINAMATH_CALUDE_tom_height_l3824_382423

theorem tom_height (t m : ℝ) : 
  t = 0.75 * m →                     -- Tom was 25% shorter than Mary two years ago
  m + 4 = 1.2 * (1.2 * t) →          -- Mary is now 20% taller than Tom after both have grown
  1.2 * t = 45 :=                    -- Tom's current height is 45 inches
by sorry

end NUMINAMATH_CALUDE_tom_height_l3824_382423


namespace NUMINAMATH_CALUDE_convex_curve_sum_containment_l3824_382451

/-- A convex curve in a 2D plane -/
structure ConvexCurve where
  points : Set (ℝ × ℝ)
  convex : sorry -- Add appropriate convexity condition

/-- The Minkowski sum of two convex curves -/
def minkowski_sum (K L : ConvexCurve) : ConvexCurve :=
  sorry

/-- One curve does not go beyond another -/
def not_beyond (K L : ConvexCurve) : Prop :=
  K.points ⊆ L.points

theorem convex_curve_sum_containment
  (K₁ K₂ L₁ L₂ : ConvexCurve)
  (h₁ : not_beyond K₁ L₁)
  (h₂ : not_beyond K₂ L₂) :
  not_beyond (minkowski_sum K₁ K₂) (minkowski_sum L₁ L₂) :=
sorry

end NUMINAMATH_CALUDE_convex_curve_sum_containment_l3824_382451


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l3824_382407

theorem at_least_one_geq_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l3824_382407


namespace NUMINAMATH_CALUDE_weight_of_eight_moles_l3824_382406

/-- The total weight of a given number of moles of a compound -/
def total_weight (molecular_weight : ℝ) (moles : ℝ) : ℝ :=
  molecular_weight * moles

/-- Proof that 8 moles of a compound with molecular weight 496 g/mol has a total weight of 3968 g -/
theorem weight_of_eight_moles :
  let molecular_weight : ℝ := 496
  let moles : ℝ := 8
  total_weight molecular_weight moles = 3968 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_eight_moles_l3824_382406


namespace NUMINAMATH_CALUDE_meow_to_paw_ratio_l3824_382408

/-- The number of cats in Cat Cafe Cool -/
def cool_cats : ℕ := 5

/-- The number of cats in Cat Cafe Paw -/
def paw_cats : ℕ := 2 * cool_cats

/-- The total number of cats in Cat Cafe Meow and Cat Cafe Paw combined -/
def total_cats : ℕ := 40

/-- The number of cats in Cat Cafe Meow -/
def meow_cats : ℕ := total_cats - paw_cats

/-- The theorem stating that Cat Cafe Meow has 3 times as many cats as Cat Cafe Paw -/
theorem meow_to_paw_ratio : meow_cats = 3 * paw_cats := by
  sorry

end NUMINAMATH_CALUDE_meow_to_paw_ratio_l3824_382408


namespace NUMINAMATH_CALUDE_locus_definition_correct_l3824_382404

-- Define the space we're working in (e.g., a metric space)
variable {X : Type*} [MetricSpace X]

-- Define the locus and the distance
variable (P : X) (r : ℝ) (locus : Set X)

-- Define the condition for a point to be at distance r from P
def atDistanceR (x : X) := dist x P = r

-- State the theorem
theorem locus_definition_correct :
  (∀ x : X, atDistanceR P r x → x ∈ locus) ∧
  (∀ x : X, x ∈ locus → atDistanceR P r x) :=
sorry

end NUMINAMATH_CALUDE_locus_definition_correct_l3824_382404


namespace NUMINAMATH_CALUDE_candy_difference_l3824_382419

theorem candy_difference (anna_per_house billy_per_house anna_houses billy_houses : ℕ) 
  (h1 : anna_per_house = 14)
  (h2 : billy_per_house = 11)
  (h3 : anna_houses = 60)
  (h4 : billy_houses = 75) :
  anna_per_house * anna_houses - billy_per_house * billy_houses = 15 := by
  sorry

end NUMINAMATH_CALUDE_candy_difference_l3824_382419


namespace NUMINAMATH_CALUDE_max_value_of_trig_function_l3824_382410

theorem max_value_of_trig_function :
  let f : ℝ → ℝ := fun x ↦ 2 * Real.sin x + 3 * Real.cos x
  ∃ M : ℝ, M = Real.sqrt 13 ∧ ∀ x : ℝ, f x ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_trig_function_l3824_382410


namespace NUMINAMATH_CALUDE_space_geometry_theorem_l3824_382470

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)

-- Define the theorem
theorem space_geometry_theorem 
  (m n : Line) (α β : Plane) 
  (hm_neq_n : m ≠ n) (hα_neq_β : α ≠ β) :
  (perpendicularLP m α ∧ perpendicularLP n β ∧ perpendicular m n → perpendicularPP α β) ∧
  (perpendicularLP m α ∧ parallelLP n β ∧ parallelPP α β → perpendicular m n) :=
sorry

end NUMINAMATH_CALUDE_space_geometry_theorem_l3824_382470


namespace NUMINAMATH_CALUDE_emily_sees_leo_l3824_382485

/-- The time Emily can see Leo given their speeds and distances -/
theorem emily_sees_leo (emily_speed leo_speed : ℝ) (initial_distance final_distance : ℝ) : 
  emily_speed = 15 →
  leo_speed = 10 →
  initial_distance = 0.75 →
  final_distance = 0.6 →
  (initial_distance + final_distance) / (emily_speed - leo_speed) * 60 = 16.2 := by
  sorry

end NUMINAMATH_CALUDE_emily_sees_leo_l3824_382485


namespace NUMINAMATH_CALUDE_smallest_b_value_l3824_382425

theorem smallest_b_value (a b : ℝ) : 
  (2 < a ∧ a < b) →
  (2 + a ≤ b) →
  (1 / a + 1 / b ≤ 2) →
  b ≥ (3 + Real.sqrt 7) / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l3824_382425


namespace NUMINAMATH_CALUDE_clown_balloons_l3824_382497

/-- The number of balloons a clown had initially, given the number of boys and girls who bought balloons, and the number of balloons remaining. -/
def initial_balloons (boys girls remaining : ℕ) : ℕ :=
  boys + girls + remaining

/-- Theorem stating that the clown initially had 36 balloons -/
theorem clown_balloons : initial_balloons 3 12 21 = 36 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_l3824_382497


namespace NUMINAMATH_CALUDE_star_neg_x_not_2x_squared_l3824_382499

-- Define the star operation
def star (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem stating that x ⋆ (-x) = 2x^2 is false
theorem star_neg_x_not_2x_squared : ¬ ∀ x : ℝ, star x (-x) = 2 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_star_neg_x_not_2x_squared_l3824_382499


namespace NUMINAMATH_CALUDE_parabola_square_intersection_l3824_382478

/-- A parabola y = px^2 has a common point with the square defined by vertices A(1,1), B(2,1), C(2,2), and D(1,2) if and only if 1/4 ≤ p ≤ 2 -/
theorem parabola_square_intersection (p : ℝ) : 
  (∃ x y : ℝ, y = p * x^2 ∧ 
    ((x = 1 ∧ y = 1) ∨ 
     (x = 2 ∧ y = 1) ∨ 
     (x = 2 ∧ y = 2) ∨ 
     (x = 1 ∧ y = 2) ∨
     (1 ≤ x ∧ x ≤ 2 ∧ y = 1) ∨
     (x = 2 ∧ 1 ≤ y ∧ y ≤ 2) ∨
     (1 ≤ x ∧ x ≤ 2 ∧ y = 2) ∨
     (x = 1 ∧ 1 ≤ y ∧ y ≤ 2))) ↔ 
  (1/4 : ℝ) ≤ p ∧ p ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_square_intersection_l3824_382478


namespace NUMINAMATH_CALUDE_system_solution_unique_equation_no_solution_l3824_382448

-- Problem 1
theorem system_solution_unique (x y : ℝ) : 
  x - 3*y = 4 ∧ 2*x - y = 3 ↔ x = 1 ∧ y = -1 :=
sorry

-- Problem 2
theorem equation_no_solution : 
  ¬∃ x : ℝ, (x ≠ 2) ∧ (1 / (x - 2) + 3 = (1 - x) / (2 - x)) :=
sorry

end NUMINAMATH_CALUDE_system_solution_unique_equation_no_solution_l3824_382448


namespace NUMINAMATH_CALUDE_root_values_l3824_382496

theorem root_values (a b c d e k : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (he : e ≠ 0)
  (hk1 : a * k^3 + b * k^2 + c * k + d = e)
  (hk2 : b * k^3 + c * k^2 + d * k + e = a) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
sorry

end NUMINAMATH_CALUDE_root_values_l3824_382496


namespace NUMINAMATH_CALUDE_detergent_amount_in_new_solution_l3824_382480

/-- Represents a solution with bleach, detergent, and water -/
structure Solution where
  bleach : ℝ
  detergent : ℝ
  water : ℝ

/-- The original ratio of the solution -/
def original_ratio : Solution :=
  { bleach := 2, detergent := 40, water := 100 }

/-- The new ratio after adjustments -/
def new_ratio (s : Solution) : Solution :=
  { bleach := 3 * s.bleach,
    detergent := s.detergent,
    water := 2 * s.water }

/-- The theorem stating the amount of detergent in the new solution -/
theorem detergent_amount_in_new_solution :
  let s := new_ratio original_ratio
  let water_amount := 300
  let detergent_amount := (s.detergent / s.water) * water_amount
  detergent_amount = 120 := by sorry

end NUMINAMATH_CALUDE_detergent_amount_in_new_solution_l3824_382480


namespace NUMINAMATH_CALUDE_division_problem_l3824_382422

theorem division_problem : ∃ (a b c d : Nat), 
  a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9 ∧
  19858 / 102 = 1000 * a + 100 * b + 10 * c + d ∧
  19858 % 102 = 0 :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l3824_382422


namespace NUMINAMATH_CALUDE_no_function_satisfies_inequality_l3824_382424

theorem no_function_satisfies_inequality :
  ¬∃ (f : ℝ → ℝ), (∀ x > 0, f x > 0) ∧
    (∀ x y, x > 0 → y > 0 → f x ^ 2 ≥ f (x + y) * (f x + y)) := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_inequality_l3824_382424


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3824_382453

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem cubic_function_properties (a b c x₀ : ℝ) 
  (h1 : f a b c (-1) = 0)
  (h2 : f a b c 1 = 0)
  (h3 : f a b c x₀ = 0)
  (h4 : 2 < x₀) (h5 : x₀ < 3) :
  (a + c = 0) ∧ (2 < c ∧ c < 3) ∧ (4*a + 2*b + c < -8) := by
  sorry


end NUMINAMATH_CALUDE_cubic_function_properties_l3824_382453


namespace NUMINAMATH_CALUDE_sequence_a_closed_form_l3824_382474

def sequence_a : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 6
  | (n + 3) => (n + 7) * sequence_a (n + 2) - 4 * (n + 3) * sequence_a (n + 1) + (4 * (n + 3) - 8) * sequence_a n

theorem sequence_a_closed_form (n : ℕ) : sequence_a n = n.factorial + 2^n := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_closed_form_l3824_382474


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l3824_382482

theorem value_of_a_minus_b (a b : ℝ) (h1 : 2 * a - b = 5) (h2 : a - 2 * b = 4) : a - b = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l3824_382482


namespace NUMINAMATH_CALUDE_exactly_one_correct_probability_l3824_382467

theorem exactly_one_correct_probability
  (probA : ℝ) (probB : ℝ) (probC : ℝ)
  (hprobA : probA = 3/4)
  (hprobB : probB = 2/3)
  (hprobC : probC = 2/3)
  (hprobA_bounds : 0 ≤ probA ∧ probA ≤ 1)
  (hprobB_bounds : 0 ≤ probB ∧ probB ≤ 1)
  (hprobC_bounds : 0 ≤ probC ∧ probC ≤ 1) :
  probA * (1 - probB) * (1 - probC) +
  (1 - probA) * probB * (1 - probC) +
  (1 - probA) * (1 - probB) * probC = 7/36 :=
sorry

end NUMINAMATH_CALUDE_exactly_one_correct_probability_l3824_382467


namespace NUMINAMATH_CALUDE_max_value_of_function_l3824_382403

theorem max_value_of_function (x : ℝ) (h : x < 0) :
  3 * x + 4 / x ≤ -4 * Real.sqrt 3 ∧ ∃ y < 0, 3 * y + 4 / y = -4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3824_382403


namespace NUMINAMATH_CALUDE_max_sum_given_sum_squares_and_product_l3824_382435

theorem max_sum_given_sum_squares_and_product (x y : ℝ) : 
  x^2 + y^2 = 100 → xy = 40 → x + y ≤ 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_sum_squares_and_product_l3824_382435


namespace NUMINAMATH_CALUDE_production_reduction_percentage_l3824_382437

/-- Represents the production and financial data of the entrepreneur -/
structure EntrepreneurData where
  initialProduction : ℕ
  pricePerItem : ℕ
  profit : ℕ
  variableCostPerItem : ℕ

/-- Calculates the constant costs based on the entrepreneur's data -/
def constantCosts (data : EntrepreneurData) : ℕ :=
  data.initialProduction * data.pricePerItem - data.profit

/-- Calculates the break-even quantity -/
def breakEvenQuantity (data : EntrepreneurData) (constCosts : ℕ) : ℕ :=
  constCosts / (data.pricePerItem - data.variableCostPerItem)

/-- Theorem: The production volume reduction percentage that makes income equal to total cost is 20% -/
theorem production_reduction_percentage
  (data : EntrepreneurData)
  (h1 : data.initialProduction = 4000)
  (h2 : data.pricePerItem = 6250)
  (h3 : data.profit = 2000000)
  (h4 : data.variableCostPerItem = 3750) :
  (data.initialProduction - breakEvenQuantity data (constantCosts data)) * 100 / data.initialProduction = 20 := by
  sorry


end NUMINAMATH_CALUDE_production_reduction_percentage_l3824_382437


namespace NUMINAMATH_CALUDE_function_max_value_solution_l3824_382430

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := 4 * x^2 - 4 * a * x + a^2 - 2 * a + 2

/-- The maximum value of f(x) on the interval [0, 2] -/
def max_value : ℝ := 3

/-- The theorem stating the solution -/
theorem function_max_value_solution (a : ℝ) :
  (∀ x ∈ Set.Icc 0 2, f a x ≤ max_value) ∧
  (∃ x ∈ Set.Icc 0 2, f a x = max_value) →
  a = 5 - Real.sqrt 10 ∨ a = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_function_max_value_solution_l3824_382430


namespace NUMINAMATH_CALUDE_abs_negative_2023_l3824_382462

theorem abs_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_negative_2023_l3824_382462
