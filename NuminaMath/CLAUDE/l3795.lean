import Mathlib

namespace NUMINAMATH_CALUDE_initial_milk_percentage_l3795_379507

/-- Given a mixture of milk and water, prove the initial percentage of milk. -/
theorem initial_milk_percentage
  (total_initial_volume : ℝ)
  (added_water : ℝ)
  (final_milk_percentage : ℝ)
  (h1 : total_initial_volume = 60)
  (h2 : added_water = 40.8)
  (h3 : final_milk_percentage = 50) :
  (total_initial_volume * 84 / 100) / total_initial_volume = 
  (total_initial_volume * final_milk_percentage / 100) / (total_initial_volume + added_water) :=
by sorry

end NUMINAMATH_CALUDE_initial_milk_percentage_l3795_379507


namespace NUMINAMATH_CALUDE_cameron_work_time_l3795_379573

theorem cameron_work_time (cameron_alone : ℝ) 
  (h1 : cameron_alone > 0)
  (h2 : 9 / cameron_alone + 1 / 2 = 1)
  (h3 : (1 / cameron_alone + 1 / 7) * 7 = 1) : 
  cameron_alone = 18 := by
sorry

end NUMINAMATH_CALUDE_cameron_work_time_l3795_379573


namespace NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l3795_379579

/-- A rectangular prism with different dimensions for length, width, and height. -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : length > 0
  width_pos : width > 0
  height_pos : height > 0
  different_dimensions : length ≠ width ∧ width ≠ height ∧ length ≠ height

/-- The number of pairs of parallel edges in a rectangular prism. -/
def parallel_edge_pairs (prism : RectangularPrism) : ℕ := 12

/-- Theorem stating that a rectangular prism has exactly 12 pairs of parallel edges. -/
theorem rectangular_prism_parallel_edges (prism : RectangularPrism) :
  parallel_edge_pairs prism = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l3795_379579


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3795_379599

theorem solve_linear_equation (y : ℚ) (h : -3 * y - 8 = 10 * y + 5) : y = -1 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3795_379599


namespace NUMINAMATH_CALUDE_tomato_plants_l3795_379545

theorem tomato_plants (first_plant : ℕ) : 
  (∃ (second_plant third_plant fourth_plant : ℕ),
    second_plant = first_plant + 4 ∧
    third_plant = 3 * (first_plant + second_plant) ∧
    fourth_plant = 3 * (first_plant + second_plant) ∧
    first_plant + second_plant + third_plant + fourth_plant = 140) →
  first_plant = 8 := by
  sorry

end NUMINAMATH_CALUDE_tomato_plants_l3795_379545


namespace NUMINAMATH_CALUDE_equal_money_time_l3795_379528

/-- 
Proves that Carol and Mike will have the same amount of money after 5 weeks,
given their initial amounts and weekly savings rates.
-/
theorem equal_money_time (carol_initial : ℕ) (mike_initial : ℕ) 
  (carol_weekly : ℕ) (mike_weekly : ℕ) :
  carol_initial = 60 →
  mike_initial = 90 →
  carol_weekly = 9 →
  mike_weekly = 3 →
  ∃ w : ℕ, w = 5 ∧ carol_initial + w * carol_weekly = mike_initial + w * mike_weekly :=
by
  sorry

#check equal_money_time

end NUMINAMATH_CALUDE_equal_money_time_l3795_379528


namespace NUMINAMATH_CALUDE_min_distance_line_parabola_l3795_379555

noncomputable def line (x : ℝ) : ℝ := (15/8) * x - 8

noncomputable def parabola (x : ℝ) : ℝ := x^2

theorem min_distance_line_parabola :
  ∃ (x₁ x₂ : ℝ),
    (∀ y₁ y₂ : ℝ,
      y₁ = line x₁ ∧ y₂ = parabola x₂ →
      (x₂ - x₁)^2 + (y₂ - y₁)^2 ≥ (1823/544)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_parabola_l3795_379555


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3795_379598

theorem absolute_value_inequality (m : ℝ) : 
  (∀ x : ℝ, |x - 2| + |x + 4| ≥ m^2 - 5*m) ↔ -1 ≤ m ∧ m ≤ 6 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3795_379598


namespace NUMINAMATH_CALUDE_square_difference_l3795_379500

theorem square_difference (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 5) : b^2 - a^2 = -15 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3795_379500


namespace NUMINAMATH_CALUDE_sqrt_sin_identity_l3795_379537

theorem sqrt_sin_identity : Real.sqrt (1 - Real.sin 2) + Real.sqrt (1 + Real.sin 2) = 2 * Real.sin 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sin_identity_l3795_379537


namespace NUMINAMATH_CALUDE_group_size_proof_l3795_379570

theorem group_size_proof (n : ℕ) (D : ℝ) (h : D > 0) : 
  (n : ℝ) / 8 * D + (n : ℝ) / 10 * D = D → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l3795_379570


namespace NUMINAMATH_CALUDE_triangle_acute_from_angle_ratio_l3795_379519

/-- Theorem: In a triangle ABC where the ratio of angles A:B:C is 2:3:4, all angles are less than 90 degrees. -/
theorem triangle_acute_from_angle_ratio (A B C : ℝ) (h_ratio : ∃ (x : ℝ), A = 2*x ∧ B = 3*x ∧ C = 4*x) 
  (h_sum : A + B + C = 180) : A < 90 ∧ B < 90 ∧ C < 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_acute_from_angle_ratio_l3795_379519


namespace NUMINAMATH_CALUDE_mystery_book_shelves_l3795_379531

/-- Proves that the number of mystery book shelves is 8 --/
theorem mystery_book_shelves :
  let books_per_shelf : ℕ := 7
  let picture_book_shelves : ℕ := 2
  let total_books : ℕ := 70
  let mystery_book_shelves : ℕ := (total_books - picture_book_shelves * books_per_shelf) / books_per_shelf
  mystery_book_shelves = 8 := by
sorry

end NUMINAMATH_CALUDE_mystery_book_shelves_l3795_379531


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l3795_379585

/-- Given a rectangular plot with the following properties:
  - The length is 10 meters more than the breadth
  - The cost of fencing is 26.50 per meter
  - The total cost of fencing is 5300
  Prove that the length of the plot is 55 meters. -/
theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) : 
  length = breadth + 10 →
  26.50 * (2 * (length + breadth)) = 5300 →
  length = 55 := by sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l3795_379585


namespace NUMINAMATH_CALUDE_jackies_tree_climb_l3795_379557

theorem jackies_tree_climb (h : ℝ) : 
  h > 0 →                             -- Height is positive
  (h + h/2 + h/2 + (h + 200)) / 4 = 800 →  -- Average height condition
  h = 1000 := by
sorry

end NUMINAMATH_CALUDE_jackies_tree_climb_l3795_379557


namespace NUMINAMATH_CALUDE_students_in_both_competitions_l3795_379576

/-- The number of students who participated in both Go and Chess competitions -/
def both_competitions (total : ℕ) (go : ℕ) (chess : ℕ) : ℕ :=
  go + chess - total

/-- Theorem stating the number of students in both competitions -/
theorem students_in_both_competitions :
  both_competitions 32 18 23 = 9 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_competitions_l3795_379576


namespace NUMINAMATH_CALUDE_expression_simplification_l3795_379522

theorem expression_simplification (x : ℝ) (h : x = 3) : 
  (x - 1 + (2 - 2*x) / (x + 1)) / ((x^2 - x) / (x + 1)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3795_379522


namespace NUMINAMATH_CALUDE_unfair_die_expected_value_l3795_379583

/-- An unfair 8-sided die with specific probability distribution -/
structure UnfairDie where
  /-- The probability of rolling an 8 -/
  p_eight : ℝ
  /-- The probability of rolling any number from 1 to 7 -/
  p_others : ℝ
  /-- The die has 8 sides -/
  sides : Nat
  sides_eq : sides = 8
  /-- The probability of rolling an 8 is 3/8 -/
  p_eight_eq : p_eight = 3/8
  /-- The sum of all probabilities is 1 -/
  prob_sum : p_eight + 7 * p_others = 1

/-- The expected value of rolling the unfair die -/
def expected_value (d : UnfairDie) : ℝ :=
  d.p_others * (1 + 2 + 3 + 4 + 5 + 6 + 7) + d.p_eight * 8

/-- Theorem: The expected value of rolling this unfair 8-sided die is 5.5 -/
theorem unfair_die_expected_value (d : UnfairDie) : expected_value d = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_unfair_die_expected_value_l3795_379583


namespace NUMINAMATH_CALUDE_new_year_markup_percentage_l3795_379534

/-- Proves that given specific markups and profit, the New Year season markup is 25% -/
theorem new_year_markup_percentage
  (initial_markup : ℝ)
  (february_discount : ℝ)
  (final_profit : ℝ)
  (h1 : initial_markup = 0.20)
  (h2 : february_discount = 0.10)
  (h3 : final_profit = 0.35)
  : ∃ (new_year_markup : ℝ),
    (1 + initial_markup) * (1 + new_year_markup) * (1 - february_discount) = 1 + final_profit ∧
    new_year_markup = 0.25 :=
sorry

end NUMINAMATH_CALUDE_new_year_markup_percentage_l3795_379534


namespace NUMINAMATH_CALUDE_horse_catches_dog_l3795_379515

/-- Represents the relative speed and step distance of animals -/
structure AnimalData where
  steps_per_time_unit : ℕ
  distance_per_steps : ℕ

/-- Calculates the distance an animal covers in one time unit -/
def speed (a : AnimalData) : ℕ := a.steps_per_time_unit * a.distance_per_steps

theorem horse_catches_dog (dog : AnimalData) (horse : AnimalData) 
  (h1 : dog.steps_per_time_unit = 5)
  (h2 : horse.steps_per_time_unit = 3)
  (h3 : 4 * horse.distance_per_steps = 7 * dog.distance_per_steps)
  (initial_distance : ℕ)
  (h4 : initial_distance = 30) :
  (speed horse - speed dog) * 600 = initial_distance * (speed horse) :=
sorry

end NUMINAMATH_CALUDE_horse_catches_dog_l3795_379515


namespace NUMINAMATH_CALUDE_product_remainder_mod_17_l3795_379525

theorem product_remainder_mod_17 : (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_17_l3795_379525


namespace NUMINAMATH_CALUDE_sin_cos_power_six_bounds_l3795_379593

theorem sin_cos_power_six_bounds :
  ∀ x : ℝ, (1 : ℝ) / 4 ≤ Real.sin x ^ 6 + Real.cos x ^ 6 ∧
            Real.sin x ^ 6 + Real.cos x ^ 6 ≤ 1 ∧
            (∃ y : ℝ, Real.sin y ^ 6 + Real.cos y ^ 6 = (1 : ℝ) / 4) ∧
            (∃ z : ℝ, Real.sin z ^ 6 + Real.cos z ^ 6 = 1) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_power_six_bounds_l3795_379593


namespace NUMINAMATH_CALUDE_octal_1072_equals_base5_4240_l3795_379560

def octal_to_decimal (n : List Nat) : Nat :=
  n.enum.foldr (λ (i, d) acc => acc + d * (8 ^ i)) 0

def decimal_to_base5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem octal_1072_equals_base5_4240 :
  decimal_to_base5 (octal_to_decimal [2, 7, 0, 1]) = [4, 2, 4, 0] := by
  sorry

end NUMINAMATH_CALUDE_octal_1072_equals_base5_4240_l3795_379560


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3795_379572

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 2 + a 8 = 12) :
  a 5 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l3795_379572


namespace NUMINAMATH_CALUDE_different_parrot_extra_toes_l3795_379546

/-- Represents the nail trimming scenario for Cassie's pets -/
structure PetNails where
  num_dogs : Nat
  num_parrots : Nat
  dog_nails_per_foot : Nat
  dog_feet : Nat
  parrot_claws_per_leg : Nat
  parrot_legs : Nat
  total_nails_to_cut : Nat

/-- Calculates the number of extra toes on the different parrot -/
def extra_toes (p : PetNails) : Nat :=
  let standard_dog_nails := p.num_dogs * p.dog_nails_per_foot * p.dog_feet
  let standard_parrot_claws := (p.num_parrots - 1) * p.parrot_claws_per_leg * p.parrot_legs
  let standard_nails := standard_dog_nails + standard_parrot_claws
  p.total_nails_to_cut - standard_nails - (p.parrot_claws_per_leg * p.parrot_legs)

/-- Theorem stating that the number of extra toes on the different parrot is 7 -/
theorem different_parrot_extra_toes :
  ∃ (p : PetNails), 
    p.num_dogs = 4 ∧ 
    p.num_parrots = 8 ∧ 
    p.dog_nails_per_foot = 4 ∧ 
    p.dog_feet = 4 ∧ 
    p.parrot_claws_per_leg = 3 ∧ 
    p.parrot_legs = 2 ∧ 
    p.total_nails_to_cut = 113 ∧ 
    extra_toes p = 7 := by
  sorry

end NUMINAMATH_CALUDE_different_parrot_extra_toes_l3795_379546


namespace NUMINAMATH_CALUDE_lecture_room_seating_l3795_379550

theorem lecture_room_seating (m n : ℕ) : 
  (∃ boys_per_row girls_per_column unoccupied : ℕ,
    boys_per_row = 6 ∧ 
    girls_per_column = 8 ∧ 
    unoccupied = 15 ∧
    m * n = boys_per_row * m + girls_per_column * n + unoccupied) →
  (m - 8) * (n - 6) = 63 :=
by sorry

end NUMINAMATH_CALUDE_lecture_room_seating_l3795_379550


namespace NUMINAMATH_CALUDE_red_balls_count_l3795_379533

theorem red_balls_count (total_balls : ℕ) (prob_red : ℚ) : 
  total_balls = 20 → prob_red = 1/4 → (prob_red * total_balls : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l3795_379533


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3795_379562

theorem fraction_subtraction : ((5 / 2) / (7 / 12)) - 4 / 9 = 242 / 63 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3795_379562


namespace NUMINAMATH_CALUDE_ring_toss_earnings_l3795_379521

/-- The ring toss game earnings problem -/
theorem ring_toss_earnings (total_earnings : ℕ) (num_days : ℕ) (daily_earnings : ℕ) : 
  total_earnings = 165 → num_days = 5 → total_earnings = num_days * daily_earnings → daily_earnings = 33 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_earnings_l3795_379521


namespace NUMINAMATH_CALUDE_imaginary_product_condition_l3795_379597

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the theorem
theorem imaginary_product_condition (a : ℝ) : 
  (((1 : ℂ) + i) * ((1 : ℂ) + a * i)).re = 0 → a = 1 := by
  sorry

-- Note: We use .re to get the real part of the complex number, 
-- which should be 0 for a purely imaginary number.

end NUMINAMATH_CALUDE_imaginary_product_condition_l3795_379597


namespace NUMINAMATH_CALUDE_simple_interest_principal_l3795_379589

def simple_interest_rate : ℚ := 8 / 100
def simple_interest_time : ℕ := 5
def compound_principal : ℕ := 8000
def compound_interest_rate : ℚ := 15 / 100
def compound_interest_time : ℕ := 2

def compound_interest (P : ℕ) (r : ℚ) (t : ℕ) : ℚ :=
  P * ((1 + r) ^ t - 1)

theorem simple_interest_principal :
  ∃ (P : ℕ), 
    (P : ℚ) * simple_interest_rate * simple_interest_time = 
    (1 / 2) * compound_interest compound_principal compound_interest_rate compound_interest_time ∧
    P = 3225 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l3795_379589


namespace NUMINAMATH_CALUDE_common_off_days_count_l3795_379563

/-- Charlie's work cycle in days -/
def charlie_cycle : ℕ := 6

/-- Dana's work cycle in days -/
def dana_cycle : ℕ := 7

/-- Total number of days -/
def total_days : ℕ := 1500

/-- Function to calculate the number of common off days -/
def common_off_days (charlie_cycle dana_cycle total_days : ℕ) : ℕ :=
  2 * (total_days / (charlie_cycle.lcm dana_cycle))

/-- Theorem stating that Charlie and Dana have 70 common off days -/
theorem common_off_days_count : 
  common_off_days charlie_cycle dana_cycle total_days = 70 := by
  sorry

end NUMINAMATH_CALUDE_common_off_days_count_l3795_379563


namespace NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l3795_379592

/-- Converts a base 8 number to base 10 -/
def base8ToBase10 (n : Nat) : Nat :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

/-- Checks if a number is a valid 3-digit base 8 number -/
def isValid3DigitBase8 (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 777

theorem greatest_3digit_base8_divisible_by_7 :
  ∃ (n : Nat), isValid3DigitBase8 n ∧
               base8ToBase10 n % 7 = 0 ∧
               ∀ (m : Nat), isValid3DigitBase8 m ∧ base8ToBase10 m % 7 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l3795_379592


namespace NUMINAMATH_CALUDE_limes_given_correct_l3795_379527

/-- The number of limes Dan initially picked -/
def initial_limes : ℕ := 9

/-- The number of limes Dan has now -/
def current_limes : ℕ := 5

/-- The number of limes Dan gave to Sara -/
def limes_given : ℕ := initial_limes - current_limes

theorem limes_given_correct : limes_given = 4 := by sorry

end NUMINAMATH_CALUDE_limes_given_correct_l3795_379527


namespace NUMINAMATH_CALUDE_inequality_proof_l3795_379517

theorem inequality_proof (a b : ℝ) (h : a > b) : -3 * a < -3 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3795_379517


namespace NUMINAMATH_CALUDE_factorization_equality_l3795_379511

theorem factorization_equality (a b : ℝ) : a * b^2 - a = a * (b + 1) * (b - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3795_379511


namespace NUMINAMATH_CALUDE_quadratic_root_conditions_l3795_379520

theorem quadratic_root_conditions (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^2 + (a^2 - 1)*x + a - 2 = 0 ∧ 
    y^2 + (a^2 - 1)*y + a - 2 = 0 ∧ 
    x > 1 ∧ y < 1) ↔ 
  -2 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_conditions_l3795_379520


namespace NUMINAMATH_CALUDE_unique_first_degree_polynomial_l3795_379513

/-- The polynomial p(x) = 2x + 1 -/
def p (x : ℝ) : ℝ := 2 * x + 1

/-- The polynomial q(x) = x -/
def q (x : ℝ) : ℝ := x

theorem unique_first_degree_polynomial :
  ∀ (x : ℝ), p (p (q x)) = q (p (p x)) ∧
  ∀ (r : ℝ → ℝ), (∃ (a b : ℝ), ∀ (x : ℝ), r x = a * x + b) →
  (∀ (x : ℝ), p (p (r x)) = r (p (p x))) →
  r = q :=
sorry

end NUMINAMATH_CALUDE_unique_first_degree_polynomial_l3795_379513


namespace NUMINAMATH_CALUDE_correct_passengers_off_l3795_379564

/-- Calculates the number of passengers who got off the bus at other stops -/
def passengers_who_got_off (initial : ℕ) (first_stop : ℕ) (other_stops : ℕ) (final : ℕ) : ℕ :=
  initial + first_stop - (final - other_stops)

theorem correct_passengers_off : passengers_who_got_off 50 16 5 49 = 22 := by
  sorry

end NUMINAMATH_CALUDE_correct_passengers_off_l3795_379564


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3795_379566

theorem quadratic_equation_coefficients :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 4 * x - 1
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c ∧ a = 3 ∧ b = -4 ∧ c = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3795_379566


namespace NUMINAMATH_CALUDE_complex_sum_and_product_l3795_379542

theorem complex_sum_and_product : ∃ (z₁ z₂ : ℂ),
  z₁ = 2 + 5*I ∧ z₂ = 3 - 7*I ∧ z₁ + z₂ = 5 - 2*I ∧ z₁ * z₂ = -29 + I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_sum_and_product_l3795_379542


namespace NUMINAMATH_CALUDE_probability_of_strong_l3795_379502

def word_train : Finset Char := {'T', 'R', 'A', 'I', 'N'}
def word_shield : Finset Char := {'S', 'H', 'I', 'E', 'L', 'D'}
def word_grow : Finset Char := {'G', 'R', 'O', 'W'}
def word_strong : Finset Char := {'S', 'T', 'R', 'O', 'N', 'G'}

def prob_train : ℚ := 1 / (word_train.card.choose 3)
def prob_shield : ℚ := 3 / (word_shield.card.choose 4)
def prob_grow : ℚ := 1 / (word_grow.card.choose 2)

theorem probability_of_strong :
  prob_train * prob_shield * prob_grow = 1 / 300 :=
sorry

end NUMINAMATH_CALUDE_probability_of_strong_l3795_379502


namespace NUMINAMATH_CALUDE_angle_bisector_length_l3795_379504

/-- Given a triangle PQR with side lengths PQ and PR, and the cosine of angle P,
    calculate the length of the angle bisector PS. -/
theorem angle_bisector_length (PQ PR : ℝ) (cos_P : ℝ) (h_PQ : PQ = 4) (h_PR : PR = 8) (h_cos_P : cos_P = 1/9) :
  ∃ (PS : ℝ), PS = Real.sqrt ((43280 - 128 * Real.sqrt 41) / 81) :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_length_l3795_379504


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3795_379505

-- Define a quadratic function with integer coefficients
def QuadraticFunction (a b c : ℤ) : ℤ → ℤ := fun x ↦ a * x^2 + b * x + c

-- Define the set of possible values for f(0), f(3), and f(4)
def PossibleValues : Set ℤ := {2, 20, 202, 2022}

-- Theorem statement
theorem quadratic_function_property (a b c : ℤ) :
  let f := QuadraticFunction a b c
  (f 0 ∈ PossibleValues) ∧
  (f 3 ∈ PossibleValues) ∧
  (f 4 ∈ PossibleValues) ∧
  (f 0 ≠ f 3) ∧
  (f 0 ≠ f 4) ∧
  (f 3 ≠ f 4) →
  (f 1 = -80) ∨ (f 1 = -990) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3795_379505


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_l3795_379595

/-- Conversion from rectangular to cylindrical coordinates -/
theorem rect_to_cylindrical (x y z : ℝ) :
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 ∧ y < 0 then 2 * Real.pi + Real.arctan (y / x) else Real.arctan (y / x)
  x = 3 ∧ y = -3 * Real.sqrt 3 ∧ z = 2 →
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi →
  (r, θ, z) = (6, 5 * Real.pi / 3, 2) := by
sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_l3795_379595


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3795_379543

/-- Given a geometric sequence where the fifth term is 48 and the sixth term is 72,
    the first term of the sequence is 768/81. -/
theorem geometric_sequence_first_term :
  ∀ (a r : ℚ),
    a * r^4 = 48 →
    a * r^5 = 72 →
    a = 768/81 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3795_379543


namespace NUMINAMATH_CALUDE_range_of_a_l3795_379532

/-- The equation x^2 + 2ax + 1 = 0 has two real roots greater than -1 -/
def p (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > -1 ∧ x₂ > -1 ∧ x₁^2 + 2*a*x₁ + 1 = 0 ∧ x₂^2 + 2*a*x₂ + 1 = 0

/-- The solution set to the inequality ax^2 - ax + 1 > 0 is ℝ -/
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, a*x^2 - a*x + 1 > 0

theorem range_of_a :
  (∀ a : ℝ, p a ∨ q a) →
  (∀ a : ℝ, ¬q a) →
  {a : ℝ | a ≤ -1} = {a : ℝ | p a} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3795_379532


namespace NUMINAMATH_CALUDE_cosine_angle_between_vectors_l3795_379556

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (5, 12)

theorem cosine_angle_between_vectors :
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  dot_product / (magnitude_a * magnitude_b) = 63 / 65 := by
sorry

end NUMINAMATH_CALUDE_cosine_angle_between_vectors_l3795_379556


namespace NUMINAMATH_CALUDE_unique_x_l3795_379581

theorem unique_x : ∃! x : ℕ, x > 0 ∧ ∃ k : ℕ, x = 9 * k ∧ x^2 < 200 ∧ x < 25 := by
  sorry

end NUMINAMATH_CALUDE_unique_x_l3795_379581


namespace NUMINAMATH_CALUDE_david_presents_l3795_379536

theorem david_presents (christmas_presents : ℕ) (birthday_presents : ℕ) : 
  christmas_presents = 2 * birthday_presents →
  christmas_presents = 60 →
  christmas_presents + birthday_presents = 90 := by
sorry

end NUMINAMATH_CALUDE_david_presents_l3795_379536


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l3795_379553

theorem unique_four_digit_number : ∃! n : ℕ, 
  (1000 ≤ n ∧ n ≤ 9999) ∧ 
  (∃ a : ℕ, n = a^2) ∧
  (∃ b : ℕ, n % 1000 = b^3) ∧
  (∃ c : ℕ, n % 100 = c^4) ∧
  n = 9216 :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l3795_379553


namespace NUMINAMATH_CALUDE_cookie_circle_properties_l3795_379544

/-- Given a circle described by the equation x^2 + y^2 + 10 = 6x + 12y,
    this theorem proves its radius, circumference, and area. -/
theorem cookie_circle_properties :
  let equation := fun (x y : ℝ) => x^2 + y^2 + 10 = 6*x + 12*y
  ∃ (center : ℝ × ℝ) (r : ℝ),
    (∀ x y, equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = r^2) ∧
    r = Real.sqrt 35 ∧
    2 * Real.pi * r = 2 * Real.pi * Real.sqrt 35 ∧
    Real.pi * r^2 = 35 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cookie_circle_properties_l3795_379544


namespace NUMINAMATH_CALUDE_sculpture_cost_in_cny_l3795_379567

/-- Exchange rate from US dollars to Namibian dollars -/
def usd_to_nad : ℚ := 8

/-- Exchange rate from US dollars to Chinese yuan -/
def usd_to_cny : ℚ := 5

/-- Cost of the sculpture in Namibian dollars -/
def sculpture_cost_nad : ℚ := 160

/-- Converts Namibian dollars to Chinese yuan -/
def nad_to_cny (nad : ℚ) : ℚ :=
  nad * (usd_to_cny / usd_to_nad)

theorem sculpture_cost_in_cny :
  nad_to_cny sculpture_cost_nad = 100 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_cost_in_cny_l3795_379567


namespace NUMINAMATH_CALUDE_birthday_celebration_attendance_l3795_379512

/-- The number of people who stayed at a birthday celebration --/
def people_stayed (total_guests : ℕ) (men : ℕ) (children_left : ℕ) : ℕ :=
  let women := total_guests / 2
  let children := total_guests - women - men
  let men_left := men / 3
  total_guests - men_left - children_left

/-- Theorem about the number of people who stayed at the birthday celebration --/
theorem birthday_celebration_attendance :
  people_stayed 60 15 5 = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_birthday_celebration_attendance_l3795_379512


namespace NUMINAMATH_CALUDE_sugar_consumption_reduction_l3795_379541

theorem sugar_consumption_reduction (initial_price new_price : ℝ) 
  (h1 : initial_price = 6)
  (h2 : new_price = 7.50) :
  let price_increase_ratio := new_price / initial_price
  let consumption_reduction_percentage := (1 - 1 / price_increase_ratio) * 100
  consumption_reduction_percentage = 25 := by sorry

end NUMINAMATH_CALUDE_sugar_consumption_reduction_l3795_379541


namespace NUMINAMATH_CALUDE_tangent_line_property_l3795_379590

/-- Given a differentiable function f : ℝ → ℝ with a tangent line y = (1/2)x + 2
    at the point (1, f(1)), prove that f(1) + f'(1) = 3 -/
theorem tangent_line_property (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_tangent : (fun x ↦ (1/2 : ℝ) * x + 2) = fun x ↦ f 1 + deriv f 1 * (x - 1)) :
  f 1 + deriv f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_property_l3795_379590


namespace NUMINAMATH_CALUDE_train_length_l3795_379529

/-- Given a train that crosses a bridge and passes a lamp post, calculate its length. -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (post_time : ℝ)
  (h1 : bridge_length = 2500)
  (h2 : bridge_time = 120)
  (h3 : post_time = 30) :
  bridge_length * (post_time / bridge_time) / (1 - post_time / bridge_time) = 2500 * (1/4) / (1 - 1/4) :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3795_379529


namespace NUMINAMATH_CALUDE_carls_ride_distance_l3795_379571

/-- The distance between Carl's house and Ralph's house -/
def distance : ℝ := 10

/-- The time Carl spent riding to Ralph's house in hours -/
def time : ℝ := 5

/-- Carl's speed in miles per hour -/
def speed : ℝ := 2

/-- Theorem: The distance between Carl's house and Ralph's house is 10 miles -/
theorem carls_ride_distance : distance = speed * time := by
  sorry

end NUMINAMATH_CALUDE_carls_ride_distance_l3795_379571


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3795_379559

theorem complex_fraction_equality : Complex.I ^ 2 = -1 → (1 + Complex.I) / (3 - Complex.I) - Complex.I / (3 + Complex.I) = (1 + Complex.I) / 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3795_379559


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3795_379558

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + 3 * b = 1) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ 2 * x + 3 * y = 1 → 1 / a + 1 / b ≤ 1 / x + 1 / y) ∧
  1 / a + 1 / b = 65 / 6 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3795_379558


namespace NUMINAMATH_CALUDE_triangle_properties_l3795_379575

/-- Given a triangle ABC with sides a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C) 
  (h2 : t.a + t.c = 6) 
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = Real.sqrt 3) : 
  t.B = π/3 ∧ t.a + t.b + t.c = 6 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3795_379575


namespace NUMINAMATH_CALUDE_rose_flyers_count_l3795_379508

def total_flyers : ℕ := 1236
def jack_flyers : ℕ := 120
def left_flyers : ℕ := 796

theorem rose_flyers_count : total_flyers - jack_flyers - left_flyers = 320 := by
  sorry

end NUMINAMATH_CALUDE_rose_flyers_count_l3795_379508


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l3795_379551

theorem more_girls_than_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) 
  (h_total : total_students = 49)
  (h_ratio : boy_ratio = 3 ∧ girl_ratio = 4) : 
  let y := total_students / (boy_ratio + girl_ratio)
  let num_boys := boy_ratio * y
  let num_girls := girl_ratio * y
  num_girls - num_boys = 7 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l3795_379551


namespace NUMINAMATH_CALUDE_smallest_n_boxes_two_boxes_satisfies_two_is_smallest_l3795_379594

theorem smallest_n_boxes (n : ℕ) : 
  (∃ k : ℕ, 15 * n - 2 = 7 * k) → n ≥ 2 :=
by
  sorry

theorem two_boxes_satisfies : 
  ∃ k : ℕ, 15 * 2 - 2 = 7 * k :=
by
  sorry

theorem two_is_smallest : 
  ∀ m : ℕ, m < 2 → ¬(∃ k : ℕ, 15 * m - 2 = 7 * k) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_n_boxes_two_boxes_satisfies_two_is_smallest_l3795_379594


namespace NUMINAMATH_CALUDE_even_quadratic_function_l3795_379524

/-- A quadratic function f(x) = ax^2 + (2a^2 - a)x + 1 is even if and only if a = 1/2 -/
theorem even_quadratic_function (a : ℝ) :
  (∀ x, (a * x^2 + (2 * a^2 - a) * x + 1) = (a * (-x)^2 + (2 * a^2 - a) * (-x) + 1)) ↔
  a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_even_quadratic_function_l3795_379524


namespace NUMINAMATH_CALUDE_books_read_l3795_379596

theorem books_read (total_books : ℕ) (books_left : ℕ) (h : total_books = 19 ∧ books_left = 15) : 
  total_books - books_left = 4 := by
  sorry

end NUMINAMATH_CALUDE_books_read_l3795_379596


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l3795_379503

/-- Given that c + √d and its radical conjugate have a sum of 0 and a product of 9, prove that c + d = -9 -/
theorem radical_conjugate_sum_product (c d : ℝ) : 
  ((c + Real.sqrt d) + (c - Real.sqrt d) = 0) ∧ 
  ((c + Real.sqrt d) * (c - Real.sqrt d) = 9) → 
  c + d = -9 := by sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l3795_379503


namespace NUMINAMATH_CALUDE_expansion_without_x_squared_l3795_379518

theorem expansion_without_x_squared (n : ℕ+) (h1 : 5 ≤ n) (h2 : n ≤ 8) :
  (∀ (r : ℕ), r ≤ n → n - 4 * r ≠ 0 ∧ n - 4 * r ≠ 1 ∧ n - 4 * r ≠ 2) ↔ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_expansion_without_x_squared_l3795_379518


namespace NUMINAMATH_CALUDE_modes_of_test_scores_l3795_379587

/-- Represents a frequency distribution of test scores -/
def FrequencyDistribution := List (Nat × Nat)

/-- Finds the modes (most frequent scores) in a frequency distribution -/
def findModes (scores : FrequencyDistribution) : List Nat :=
  sorry

/-- The actual frequency distribution of the test scores -/
def testScores : FrequencyDistribution := [
  (62, 1), (65, 2), (70, 1), (74, 2), (78, 1),
  (81, 2), (86, 1), (89, 1), (92, 1), (97, 3),
  (101, 4), (104, 4), (110, 3)
]

theorem modes_of_test_scores :
  findModes testScores = [101, 104] :=
sorry

end NUMINAMATH_CALUDE_modes_of_test_scores_l3795_379587


namespace NUMINAMATH_CALUDE_mobile_phone_purchase_price_l3795_379514

/-- The purchase price of the refrigerator in rupees -/
def refrigerator_price : ℝ := 15000

/-- The loss percentage on the refrigerator sale -/
def refrigerator_loss_percent : ℝ := 3

/-- The profit percentage on the mobile phone sale -/
def mobile_profit_percent : ℝ := 10

/-- The overall profit in rupees -/
def overall_profit : ℝ := 350

/-- The purchase price of the mobile phone in rupees -/
def mobile_price : ℝ := 8000

theorem mobile_phone_purchase_price :
  ∃ (x : ℝ),
    x = mobile_price ∧
    refrigerator_price * (1 - refrigerator_loss_percent / 100) +
    x * (1 + mobile_profit_percent / 100) =
    refrigerator_price + x + overall_profit :=
by sorry

end NUMINAMATH_CALUDE_mobile_phone_purchase_price_l3795_379514


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3795_379568

/-- A right triangle with sides 5, 12, and 13 (hypotenuse) -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  hypotenuse : c = 13
  right_angle : a^2 + b^2 = c^2
  side_a : a = 5
  side_b : b = 12

/-- First inscribed square with side length x -/
def first_square (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x < t.a ∧ x < t.b ∧ x / t.a = x / t.b

/-- Second inscribed square with side length y -/
def second_square (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y < t.c ∧ (t.a - y) / y = (t.b - y) / y

/-- The main theorem -/
theorem inscribed_squares_ratio (t : RightTriangle) 
  (x y : ℝ) (h1 : first_square t x) (h2 : second_square t y) : 
  x / y = 78 / 102 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3795_379568


namespace NUMINAMATH_CALUDE_equation_solution_l3795_379510

theorem equation_solution :
  ∀ t : ℂ, (2 / (t + 3) + 3 * t / (t + 3) - 5 / (t + 3) = t + 2) ↔ 
  (t = -1 + 2 * Complex.I * Real.sqrt 2 ∨ t = -1 - 2 * Complex.I * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3795_379510


namespace NUMINAMATH_CALUDE_recipe_fraction_is_two_thirds_l3795_379584

/-- Represents the amount of an ingredient required for a recipe --/
structure RecipeIngredient where
  amount : ℚ
  deriving Repr

/-- Represents the amount of an ingredient available --/
structure AvailableIngredient where
  amount : ℚ
  deriving Repr

/-- Calculates the fraction of the recipe that can be made for a single ingredient --/
def ingredientFraction (required : RecipeIngredient) (available : AvailableIngredient) : ℚ :=
  available.amount / required.amount

/-- Finds the maximum fraction of the recipe that can be made given all ingredients --/
def maxRecipeFraction (sugar : RecipeIngredient × AvailableIngredient) 
                      (milk : RecipeIngredient × AvailableIngredient)
                      (flour : RecipeIngredient × AvailableIngredient) : ℚ :=
  min (ingredientFraction sugar.1 sugar.2)
      (min (ingredientFraction milk.1 milk.2)
           (ingredientFraction flour.1 flour.2))

theorem recipe_fraction_is_two_thirds :
  let sugar_required := RecipeIngredient.mk (3/4)
  let sugar_available := AvailableIngredient.mk (2/4)
  let milk_required := RecipeIngredient.mk (2/3)
  let milk_available := AvailableIngredient.mk (1/2)
  let flour_required := RecipeIngredient.mk (3/8)
  let flour_available := AvailableIngredient.mk (1/4)
  maxRecipeFraction (sugar_required, sugar_available)
                    (milk_required, milk_available)
                    (flour_required, flour_available) = 2/3 := by
  sorry

#eval maxRecipeFraction (RecipeIngredient.mk (3/4), AvailableIngredient.mk (2/4))
                        (RecipeIngredient.mk (2/3), AvailableIngredient.mk (1/2))
                        (RecipeIngredient.mk (3/8), AvailableIngredient.mk (1/4))

end NUMINAMATH_CALUDE_recipe_fraction_is_two_thirds_l3795_379584


namespace NUMINAMATH_CALUDE_inequality_preservation_l3795_379561

theorem inequality_preservation (a b c : ℝ) (h : a > b) : a + c > b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3795_379561


namespace NUMINAMATH_CALUDE_aquarium_purchase_cost_l3795_379538

/-- Calculates the total cost of an aquarium purchase with given discounts and tax rates -/
theorem aquarium_purchase_cost 
  (original_price : ℝ)
  (aquarium_discount : ℝ)
  (coupon_discount : ℝ)
  (additional_items_cost : ℝ)
  (aquarium_tax_rate : ℝ)
  (other_items_tax_rate : ℝ)
  (h1 : original_price = 120)
  (h2 : aquarium_discount = 0.5)
  (h3 : coupon_discount = 0.1)
  (h4 : additional_items_cost = 75)
  (h5 : aquarium_tax_rate = 0.05)
  (h6 : other_items_tax_rate = 0.08) :
  let discounted_price := original_price * (1 - aquarium_discount)
  let final_aquarium_price := discounted_price * (1 - coupon_discount)
  let aquarium_tax := final_aquarium_price * aquarium_tax_rate
  let other_items_tax := additional_items_cost * other_items_tax_rate
  let total_cost := final_aquarium_price + aquarium_tax + additional_items_cost + other_items_tax
  total_cost = 137.70 := by
sorry


end NUMINAMATH_CALUDE_aquarium_purchase_cost_l3795_379538


namespace NUMINAMATH_CALUDE_min_value_at_two_l3795_379578

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^2 - 12 * x + 1

-- State the theorem
theorem min_value_at_two :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_at_two_l3795_379578


namespace NUMINAMATH_CALUDE_wilsonTotalIsCorrect_l3795_379552

/-- Calculates the total amount Wilson pays at a fast-food restaurant -/
def wilsonTotal : ℝ :=
  let hamburgerPrice := 5
  let hamburgerCount := 2
  let colaPrice := 2
  let colaCount := 3
  let friesPrice := 3
  let sundaePrice := 4
  let nuggetPrice := 1.5
  let nuggetCount := 4
  let saladPrice := 6.25
  let couponDiscount := 4
  let loyaltyDiscount := 0.1
  let freeNuggetCount := 1

  let initialTotal := hamburgerPrice * hamburgerCount + colaPrice * colaCount + 
                      friesPrice + sundaePrice + nuggetPrice * nuggetCount + saladPrice
  let promotionDiscount := nuggetPrice * freeNuggetCount
  let afterPromotionTotal := initialTotal - promotionDiscount
  let afterCouponTotal := afterPromotionTotal - couponDiscount
  let finalTotal := afterCouponTotal * (1 - loyaltyDiscount)

  finalTotal

theorem wilsonTotalIsCorrect : wilsonTotal = 26.77 := by sorry

end NUMINAMATH_CALUDE_wilsonTotalIsCorrect_l3795_379552


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3795_379547

/-- An arithmetic sequence with the given properties has the general term a_n = n. -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) (d : ℝ) : 
  d ≠ 0 ∧ 
  (∀ n, a (n + 1) = a n + d) ∧ 
  a 2 ^ 2 = a 1 * a 4 ∧ 
  a 5 + a 6 = 11 → 
  ∀ n, a n = n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3795_379547


namespace NUMINAMATH_CALUDE_unique_solution_conditions_l3795_379509

/-- The system has a unique solution if and only if a = arctan(4) + πk or a = -arctan(2) + πk, where k is an integer -/
theorem unique_solution_conditions (a : ℝ) : 
  (∃! x y : ℝ, x * Real.cos a + y * Real.sin a = 5 * Real.cos a + 2 * Real.sin a ∧ 
                -3 ≤ x + 2*y ∧ x + 2*y ≤ 7 ∧ 
                -9 ≤ 3*x - 4*y ∧ 3*x - 4*y ≤ 1) ↔ 
  (∃ k : ℤ, a = Real.arctan 4 + k * Real.pi ∨ a = -Real.arctan 2 + k * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_conditions_l3795_379509


namespace NUMINAMATH_CALUDE_second_candidate_votes_l3795_379526

theorem second_candidate_votes (total_votes : ℕ) (first_candidate_percentage : ℚ) :
  total_votes = 1200 →
  first_candidate_percentage = 60 / 100 →
  (total_votes : ℚ) * (1 - first_candidate_percentage) = 480 :=
by sorry

end NUMINAMATH_CALUDE_second_candidate_votes_l3795_379526


namespace NUMINAMATH_CALUDE_subtracted_value_proof_l3795_379530

theorem subtracted_value_proof (n : ℕ) (x : ℕ) : 
  n = 36 → 
  ((n + 10) * 2) / 2 - x = 88 / 2 ↔ 
  x = 2 := by
sorry

end NUMINAMATH_CALUDE_subtracted_value_proof_l3795_379530


namespace NUMINAMATH_CALUDE_smallest_multiple_of_5_and_711_l3795_379523

theorem smallest_multiple_of_5_and_711 :
  ∀ n : ℕ, n > 0 ∧ 5 ∣ n ∧ 711 ∣ n → n ≥ 3555 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_5_and_711_l3795_379523


namespace NUMINAMATH_CALUDE_chloe_david_distance_difference_l3795_379535

-- Define the speeds and time
def chloe_speed : ℝ := 18
def david_speed : ℝ := 15
def bike_time : ℝ := 5

-- Define the theorem
theorem chloe_david_distance_difference :
  chloe_speed * bike_time - david_speed * bike_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_chloe_david_distance_difference_l3795_379535


namespace NUMINAMATH_CALUDE_five_candies_three_kids_l3795_379540

/-- The number of ways to distribute n candies among k kids, with each kid getting at least one candy -/
def distribute_candies (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 6 ways to distribute 5 candies among 3 kids, with each kid getting at least one candy -/
theorem five_candies_three_kids : distribute_candies 5 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_five_candies_three_kids_l3795_379540


namespace NUMINAMATH_CALUDE_cube_equation_solution_l3795_379577

theorem cube_equation_solution (a x : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * x * 45 * 35) : x = 35 := by
  sorry

end NUMINAMATH_CALUDE_cube_equation_solution_l3795_379577


namespace NUMINAMATH_CALUDE_salvadore_earnings_l3795_379569

/-- Given that Salvadore earned S dollars, Santo earned half of that, and their combined earnings were $2934, prove that Salvadore earned $1956. -/
theorem salvadore_earnings (S : ℝ) 
  (h1 : S + S / 2 = 2934) : S = 1956 := by
  sorry

end NUMINAMATH_CALUDE_salvadore_earnings_l3795_379569


namespace NUMINAMATH_CALUDE_currency_conversion_area_conversion_l3795_379580

-- Define the currency units
def yuan : ℝ := 1
def jiao : ℝ := 0.1
def fen : ℝ := 0.01

-- Define the area units
def hectare : ℝ := 10000
def square_meter : ℝ := 1

-- Theorem for currency conversion
theorem currency_conversion :
  6.89 * yuan = 6 * yuan + 8 * jiao + 9 * fen := by sorry

-- Theorem for area conversion
theorem area_conversion :
  2 * hectare + 60 * square_meter = 20060 * square_meter := by sorry

end NUMINAMATH_CALUDE_currency_conversion_area_conversion_l3795_379580


namespace NUMINAMATH_CALUDE_right_angled_complex_roots_l3795_379516

open Complex

theorem right_angled_complex_roots (a b : ℂ) (z₁ z₂ : ℂ) : 
  z₁^2 + a*z₁ + b = 0 → 
  z₂^2 + a*z₂ + b = 0 → 
  z₁ ≠ 0 → 
  z₂ ≠ 0 → 
  z₁ ≠ z₂ → 
  (z₁.re * z₂.re + z₁.im * z₂.im = 0) → 
  a^2 / b = 2 := by
sorry

end NUMINAMATH_CALUDE_right_angled_complex_roots_l3795_379516


namespace NUMINAMATH_CALUDE_f_g_intersection_l3795_379554

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - a * Real.log x

/-- The function g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 - (a + 1) * x

/-- Theorem stating that f and g have exactly one intersection point when a ≥ 0 -/
theorem f_g_intersection (a : ℝ) (h : a ≥ 0) :
  ∃! x : ℝ, x > 0 ∧ f a x = g a x :=
sorry

end NUMINAMATH_CALUDE_f_g_intersection_l3795_379554


namespace NUMINAMATH_CALUDE_certain_number_proof_l3795_379565

theorem certain_number_proof : ∃ x : ℝ, (213 * 16 = 3408 ∧ 16 * x = 340.8) → x = 21.3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3795_379565


namespace NUMINAMATH_CALUDE_circle_rooted_polynomial_ab_neq_nine_l3795_379539

/-- A polynomial of degree 4 with four distinct roots on a circle in the complex plane -/
structure CircleRootedPolynomial where
  a : ℂ
  b : ℂ
  roots_distinct : True  -- Placeholder for the distinctness condition
  roots_on_circle : True -- Placeholder for the circle condition

/-- The theorem stating that for a polynomial with four distinct roots on a circle, ab ≠ 9 -/
theorem circle_rooted_polynomial_ab_neq_nine (P : CircleRootedPolynomial) : P.a * P.b ≠ 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_rooted_polynomial_ab_neq_nine_l3795_379539


namespace NUMINAMATH_CALUDE_levels_ratio_l3795_379506

def total_levels : ℕ := 32
def beaten_levels : ℕ := 24

theorem levels_ratio :
  let not_beaten := total_levels - beaten_levels
  (beaten_levels : ℚ) / not_beaten = 3 / 1 := by sorry

end NUMINAMATH_CALUDE_levels_ratio_l3795_379506


namespace NUMINAMATH_CALUDE_square_circle_area_fraction_l3795_379586

theorem square_circle_area_fraction (r : ℝ) (h : r > 0) :
  let square_area := (2 * r)^2
  let circle_area := π * r^2
  let outside_area := square_area - circle_area
  outside_area / square_area = 1 - π / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_circle_area_fraction_l3795_379586


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_l3795_379588

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℚ) 
                                  (set2_count : ℕ) (set2_mean : ℚ) :
  set1_count = 7 →
  set1_mean = 15 →
  set2_count = 8 →
  set2_mean = 30 →
  let total_count := set1_count + set2_count
  let total_sum := set1_count * set1_mean + set2_count * set2_mean
  (total_sum / total_count : ℚ) = 23 := by
  sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_l3795_379588


namespace NUMINAMATH_CALUDE_ball_drawing_theorem_l3795_379549

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat

/-- Represents the random variable ξ (number of yellow balls drawn) -/
def Xi := Nat

/-- The initial state of the box -/
def initialBox : BallCounts := { red := 1, green := 1, yellow := 2 }

/-- The probability of drawing no yellow balls before drawing the red ball -/
def probXiZero (box : BallCounts) : Real :=
  sorry

/-- The expected value of ξ -/
def expectedXi (box : BallCounts) : Real :=
  sorry

/-- The main theorem stating the probability and expectation results -/
theorem ball_drawing_theorem (box : BallCounts) 
  (h : box = initialBox) : 
  probXiZero box = 1/3 ∧ expectedXi box = 1 := by
  sorry

end NUMINAMATH_CALUDE_ball_drawing_theorem_l3795_379549


namespace NUMINAMATH_CALUDE_cycle_selling_price_l3795_379582

/-- Calculates the selling price of an item given its cost price and gain percent. -/
def sellingPrice (costPrice : ℕ) (gainPercent : ℕ) : ℕ :=
  costPrice + (costPrice * gainPercent) / 100

/-- Theorem stating that the selling price of a cycle with cost price 900 and gain percent 30 is 1170. -/
theorem cycle_selling_price :
  sellingPrice 900 30 = 1170 := by
  sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l3795_379582


namespace NUMINAMATH_CALUDE_benny_total_spent_l3795_379548

def soft_drink_quantity : ℕ := 2
def soft_drink_price : ℕ := 4
def candy_bar_quantity : ℕ := 5
def candy_bar_price : ℕ := 4

theorem benny_total_spent :
  soft_drink_quantity * soft_drink_price + candy_bar_quantity * candy_bar_price = 28 := by
  sorry

end NUMINAMATH_CALUDE_benny_total_spent_l3795_379548


namespace NUMINAMATH_CALUDE_book_club_picks_l3795_379501

theorem book_club_picks (total_members : ℕ) (meeting_weeks : ℕ) (guest_picks : ℕ) :
  total_members = 13 →
  meeting_weeks = 48 →
  guest_picks = 12 →
  (meeting_weeks - guest_picks) / total_members = 2 :=
by sorry

end NUMINAMATH_CALUDE_book_club_picks_l3795_379501


namespace NUMINAMATH_CALUDE_min_subset_size_for_acute_triangle_l3795_379574

def is_acute_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

theorem min_subset_size_for_acute_triangle :
  ∃ (k : ℕ), k = 29 ∧
  (∀ (S : Finset ℕ), S ⊆ Finset.range 2004 → S.card ≥ k →
    ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ is_acute_triangle a b c) ∧
  (∀ (k' : ℕ), k' < k →
    ∃ (S : Finset ℕ), S ⊆ Finset.range 2004 ∧ S.card = k' ∧
      ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → c ≠ a → ¬is_acute_triangle a b c) :=
by sorry

end NUMINAMATH_CALUDE_min_subset_size_for_acute_triangle_l3795_379574


namespace NUMINAMATH_CALUDE_tonys_initial_money_l3795_379591

/-- Represents the problem of calculating Tony's initial amount of money --/
theorem tonys_initial_money 
  (bucket_capacity : ℝ)
  (sandbox_depth sandbox_width sandbox_length : ℝ)
  (sand_density : ℝ)
  (water_per_break : ℝ)
  (trips_per_break : ℕ)
  (bottle_capacity : ℝ)
  (bottle_cost : ℝ)
  (change : ℝ)
  (h1 : bucket_capacity = 2)
  (h2 : sandbox_depth = 2)
  (h3 : sandbox_width = 4)
  (h4 : sandbox_length = 5)
  (h5 : sand_density = 3)
  (h6 : water_per_break = 3)
  (h7 : trips_per_break = 4)
  (h8 : bottle_capacity = 15)
  (h9 : bottle_cost = 2)
  (h10 : change = 4) :
  ∃ (initial_money : ℝ), initial_money = 10 := by
sorry

end NUMINAMATH_CALUDE_tonys_initial_money_l3795_379591
