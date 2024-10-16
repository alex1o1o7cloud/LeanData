import Mathlib

namespace NUMINAMATH_CALUDE_transport_cost_500g_l674_67433

/-- Calculates the transport cost for a scientific instrument to the International Space Station -/
def transport_cost (weight_g : ℕ) : ℚ :=
  let weight_kg : ℚ := weight_g / 1000
  let base_cost : ℚ := weight_kg * 18000
  let discount_rate : ℚ := if weight_kg < 1 then 1/10 else 0
  base_cost * (1 - discount_rate)

/-- The cost to transport a 500 g scientific instrument to the International Space Station is $8,100 -/
theorem transport_cost_500g : transport_cost 500 = 8100 := by
  sorry

end NUMINAMATH_CALUDE_transport_cost_500g_l674_67433


namespace NUMINAMATH_CALUDE_largest_number_l674_67483

theorem largest_number (a b c d e : ℚ) 
  (ha : a = 0.999) 
  (hb : b = 0.9099) 
  (hc : c = 0.9991) 
  (hd : d = 0.991) 
  (he : e = 0.9091) : 
  c ≥ a ∧ c ≥ b ∧ c ≥ d ∧ c ≥ e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l674_67483


namespace NUMINAMATH_CALUDE_congruence_solution_l674_67420

theorem congruence_solution (n : ℤ) : 13 * n ≡ 9 [ZMOD 53] → n ≡ 17 [ZMOD 53] := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l674_67420


namespace NUMINAMATH_CALUDE_females_without_daughters_l674_67418

/-- Represents Bertha's family structure -/
structure BerthaFamily where
  daughters : ℕ
  granddaughters : ℕ
  total_females : ℕ
  daughters_with_children : ℕ

/-- The actual family structure of Bertha -/
def berthas_family : BerthaFamily :=
  { daughters := 8
  , granddaughters := 40
  , total_females := 48
  , daughters_with_children := 5 }

/-- Theorem stating the number of females without daughters in Bertha's family -/
theorem females_without_daughters (b : BerthaFamily) (h1 : b = berthas_family) :
  b.daughters + b.granddaughters - b.daughters_with_children = 43 := by
  sorry

#check females_without_daughters

end NUMINAMATH_CALUDE_females_without_daughters_l674_67418


namespace NUMINAMATH_CALUDE_june_birth_percentage_l674_67439

theorem june_birth_percentage (total_scientists : ℕ) (june_born : ℕ) 
  (h1 : total_scientists = 200) (h2 : june_born = 18) :
  (june_born : ℚ) / total_scientists * 100 = 9 := by
  sorry

end NUMINAMATH_CALUDE_june_birth_percentage_l674_67439


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_of_fraction_l674_67432

/-- The number of zeros between the decimal point and the first non-zero digit when 7/5000 is written as a decimal -/
def zeros_before_first_nonzero : ℕ := 2

/-- The fraction we're considering -/
def fraction : ℚ := 7 / 5000

theorem zeros_before_first_nonzero_of_fraction :
  zeros_before_first_nonzero = 2 ∧ 
  fraction = 7 / 5000 ∧
  5000 = 5^3 * 2^3 := by
  sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_of_fraction_l674_67432


namespace NUMINAMATH_CALUDE_badminton_purchase_costs_l674_67400

/-- Represents the cost calculation for badminton equipment purchases --/
structure BadmintonPurchase where
  num_rackets : ℕ
  num_shuttlecocks : ℕ
  racket_price : ℕ
  shuttlecock_price : ℕ
  store_a_promotion : Bool
  store_b_discount : ℚ

/-- Calculates the cost at Store A --/
def cost_store_a (p : BadmintonPurchase) : ℕ :=
  p.num_rackets * p.racket_price + (p.num_shuttlecocks - p.num_rackets) * p.shuttlecock_price

/-- Calculates the cost at Store B --/
def cost_store_b (p : BadmintonPurchase) : ℚ :=
  ((p.num_rackets * p.racket_price + p.num_shuttlecocks * p.shuttlecock_price : ℚ) * (1 - p.store_b_discount))

/-- The main theorem to prove --/
theorem badminton_purchase_costs 
  (x : ℕ) 
  (h : x > 16) :
  let p : BadmintonPurchase := {
    num_rackets := 16,
    num_shuttlecocks := x,
    racket_price := 150,
    shuttlecock_price := 40,
    store_a_promotion := true,
    store_b_discount := 1/5
  }
  cost_store_a p = 1760 + 40 * x ∧ 
  cost_store_b p = 1920 + 32 * x := by
  sorry

#check badminton_purchase_costs

end NUMINAMATH_CALUDE_badminton_purchase_costs_l674_67400


namespace NUMINAMATH_CALUDE_trig_identity_l674_67476

theorem trig_identity : 
  Real.sin (160 * π / 180) * Real.sin (10 * π / 180) - 
  Real.cos (20 * π / 180) * Real.cos (10 * π / 180) = 
  - (Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_trig_identity_l674_67476


namespace NUMINAMATH_CALUDE_universal_set_determination_l674_67426

universe u

theorem universal_set_determination (U : Set ℕ) (A : Set ℕ) (h1 : A = {1, 3, 5})
  (h2 : Set.compl A = {2, 4, 6}) : U = {1, 2, 3, 4, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_universal_set_determination_l674_67426


namespace NUMINAMATH_CALUDE_shooter_conditional_probability_l674_67414

/-- Given a shooter with probabilities of hitting a target, prove the conditional probability of hitting the target in a subsequent shot. -/
theorem shooter_conditional_probability
  (p_single : ℝ)
  (p_twice : ℝ)
  (h_single : p_single = 0.7)
  (h_twice : p_twice = 0.4) :
  p_twice / p_single = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_shooter_conditional_probability_l674_67414


namespace NUMINAMATH_CALUDE_solution_set_inequality_l674_67469

theorem solution_set_inequality (x : ℝ) : 
  (x * |x - 1| > 0) ↔ (x ∈ Set.Ioo 0 1 ∪ Set.Ioi 1) := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l674_67469


namespace NUMINAMATH_CALUDE_magical_stack_size_l674_67466

/-- A magical stack of cards with the given properties. -/
structure MagicalStack :=
  (n : ℕ)  -- Total number of cards is 3n
  (card_101_position : ℕ)  -- Position of card 101 after restacking
  (is_magical : Prop)  -- The stack is magical

/-- The properties of a magical stack where card 101 retains its position. -/
def magical_stack_properties (stack : MagicalStack) : Prop :=
  stack.n > 0 ∧
  stack.card_101_position = 101 ∧
  stack.is_magical

/-- Theorem stating the number of cards in the magical stack. -/
theorem magical_stack_size (stack : MagicalStack) 
  (h : magical_stack_properties stack) : 
  3 * stack.n = 303 :=
sorry

end NUMINAMATH_CALUDE_magical_stack_size_l674_67466


namespace NUMINAMATH_CALUDE_dolphin_altitude_l674_67455

/-- Given a submarine at an altitude of -50 meters and a dolphin 10 meters above it,
    the altitude of the dolphin is -40 meters. -/
theorem dolphin_altitude (submarine_altitude dolphin_distance : ℝ) :
  submarine_altitude = -50 ∧ dolphin_distance = 10 →
  submarine_altitude + dolphin_distance = -40 :=
by sorry

end NUMINAMATH_CALUDE_dolphin_altitude_l674_67455


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l674_67462

/-- Given an original salary and a final salary after an increase followed by a decrease,
    calculate the initial percentage increase. -/
theorem salary_increase_percentage
  (original_salary : ℝ)
  (final_salary : ℝ)
  (decrease_percentage : ℝ)
  (h1 : original_salary = 6000)
  (h2 : final_salary = 6270)
  (h3 : decrease_percentage = 5)
  : ∃ x : ℝ,
    final_salary = original_salary * (1 + x / 100) * (1 - decrease_percentage / 100) ∧
    x = 10 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l674_67462


namespace NUMINAMATH_CALUDE_complete_square_with_integer_l674_67427

theorem complete_square_with_integer (y : ℝ) : ∃ k : ℤ, y^2 + 10*y + 33 = (y + 5)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_complete_square_with_integer_l674_67427


namespace NUMINAMATH_CALUDE_surrounding_circles_radius_l674_67409

theorem surrounding_circles_radius (r : ℝ) : r = 2 * (Real.sqrt 2 + 1) :=
  let central_radius := 2
  let square_side := 2 * r
  let square_diagonal := square_side * Real.sqrt 2
  let total_diagonal := 2 * central_radius + 2 * r
by
  sorry

end NUMINAMATH_CALUDE_surrounding_circles_radius_l674_67409


namespace NUMINAMATH_CALUDE_carver_school_earnings_l674_67491

/-- Represents a school with its student count and work days -/
structure School where
  name : String
  students : ℕ
  days : ℕ

/-- Calculates the total payment for all schools -/
def totalPayment (schools : List School) (basePayment dailyWage : ℚ) : ℚ :=
  (schools.map (fun s => s.students * s.days) |>.sum : ℕ) * dailyWage + 
  (schools.length : ℕ) * basePayment

/-- Calculates the earnings for a specific school -/
def schoolEarnings (school : School) (dailyWage : ℚ) : ℚ :=
  (school.students * school.days : ℕ) * dailyWage

theorem carver_school_earnings :
  let allen := School.mk "Allen" 7 3
  let balboa := School.mk "Balboa" 5 6
  let carver := School.mk "Carver" 4 10
  let schools := [allen, balboa, carver]
  let basePayment := 20
  ∃ dailyWage : ℚ,
    totalPayment schools basePayment dailyWage = 900 ∧
    schoolEarnings carver dailyWage = 369.60 := by
  sorry

end NUMINAMATH_CALUDE_carver_school_earnings_l674_67491


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l674_67480

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + |2*x - 1|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f x (-1) ≤ 2} = {x : ℝ | -1/2 ≤ x ∧ x ≤ 1/2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  (∀ x ∈ Set.Icc (1/2) 1, f x a ≤ |2*x + 1|) →
  (0 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l674_67480


namespace NUMINAMATH_CALUDE_max_x_squared_y_l674_67443

theorem max_x_squared_y (x y : ℕ+) (h : 7 * x.val + 4 * y.val = 140) :
  ∀ (a b : ℕ+), 7 * a.val + 4 * b.val = 140 → x.val^2 * y.val ≥ a.val^2 * b.val :=
by sorry

end NUMINAMATH_CALUDE_max_x_squared_y_l674_67443


namespace NUMINAMATH_CALUDE_number_thought_of_l674_67403

theorem number_thought_of (x : ℝ) : (x / 5 + 23 = 42) → x = 95 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l674_67403


namespace NUMINAMATH_CALUDE_fraction_simplest_form_l674_67419

theorem fraction_simplest_form (a b c : ℝ) :
  (a^2 - b^2 + c^2 + 2*b*c) / (a^2 - c^2 + b^2 + 2*a*b) =
  (a^2 - b^2 + c^2 + 2*b*c) / (a^2 - c^2 + b^2 + 2*a*b) := by sorry

end NUMINAMATH_CALUDE_fraction_simplest_form_l674_67419


namespace NUMINAMATH_CALUDE_belinda_passed_twenty_percent_l674_67482

-- Define the total number of flyers
def total_flyers : ℕ := 200

-- Define the number of flyers passed out by each person
def ryan_flyers : ℕ := 42
def alyssa_flyers : ℕ := 67
def scott_flyers : ℕ := 51

-- Define Belinda's flyers as the remaining flyers
def belinda_flyers : ℕ := total_flyers - (ryan_flyers + alyssa_flyers + scott_flyers)

-- Define the percentage of flyers Belinda passed out
def belinda_percentage : ℚ := (belinda_flyers : ℚ) / (total_flyers : ℚ) * 100

-- Theorem stating that Belinda passed out 20% of the flyers
theorem belinda_passed_twenty_percent : belinda_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_belinda_passed_twenty_percent_l674_67482


namespace NUMINAMATH_CALUDE_largest_group_size_l674_67417

def round_fraction (n : ℕ) (d : ℕ) (x : ℕ) : ℕ :=
  (2 * n * x + d) / (2 * d)

theorem largest_group_size :
  ∀ x : ℕ, x ≤ 37 ↔
    round_fraction 1 2 x + round_fraction 1 3 x + round_fraction 1 5 x ≤ x + 1 ∧
    (∀ y : ℕ, y > x →
      round_fraction 1 2 y + round_fraction 1 3 y + round_fraction 1 5 y > y + 1) :=
by sorry

end NUMINAMATH_CALUDE_largest_group_size_l674_67417


namespace NUMINAMATH_CALUDE_weight_difference_l674_67495

/-- Weights of different shapes in grams -/
def round_weight : ℕ := 200
def square_weight : ℕ := 300
def triangular_weight : ℕ := 150

/-- Number of weights on the left pan -/
def left_square : ℕ := 1
def left_triangular : ℕ := 2
def left_round : ℕ := 3

/-- Number of weights on the right pan -/
def right_triangular : ℕ := 1
def right_round : ℕ := 2
def right_square : ℕ := 3

/-- Total weight on the left pan -/
def left_total : ℕ := 
  left_square * square_weight + 
  left_triangular * triangular_weight + 
  left_round * round_weight

/-- Total weight on the right pan -/
def right_total : ℕ := 
  right_triangular * triangular_weight + 
  right_round * round_weight + 
  right_square * square_weight

/-- The difference in weight between the right and left pans -/
theorem weight_difference : right_total - left_total = 250 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l674_67495


namespace NUMINAMATH_CALUDE_even_function_sum_ab_eq_two_l674_67415

/-- A function f is even on an interval if f(-x) = f(x) for all x in the interval -/
def IsEvenOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x, x ∈ Set.Icc a b → f (-x) = f x

theorem even_function_sum_ab_eq_two (a b : ℝ) :
  let f := fun x => a * x^2 + (b - 1) * x + 3 * a
  let domain := Set.Icc (a - 3) (2 * a)
  IsEvenOn f (a - 3) (2 * a) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_sum_ab_eq_two_l674_67415


namespace NUMINAMATH_CALUDE_side_x_must_be_green_l674_67407

-- Define the possible colors
inductive Color
  | Red
  | Green
  | Blue

-- Define a triangle with three sides
structure Triangle where
  side1 : Color
  side2 : Color
  side3 : Color

-- Define the condition that each triangle must have one of each color
def validTriangle (t : Triangle) : Prop :=
  t.side1 ≠ t.side2 ∧ t.side2 ≠ t.side3 ∧ t.side1 ≠ t.side3

-- Define the configuration of five triangles
structure Configuration where
  t1 : Triangle
  t2 : Triangle
  t3 : Triangle
  t4 : Triangle
  t5 : Triangle

-- Define the given colored sides
def givenColoring (c : Configuration) : Prop :=
  c.t1.side1 = Color.Green ∧
  c.t2.side1 = Color.Blue ∧
  c.t3.side3 = Color.Green ∧
  c.t5.side2 = Color.Blue

-- Define the shared sides
def sharedSides (c : Configuration) : Prop :=
  c.t1.side2 = c.t2.side3 ∧
  c.t1.side3 = c.t3.side1 ∧
  c.t2.side2 = c.t3.side2 ∧
  c.t3.side3 = c.t4.side1 ∧
  c.t4.side2 = c.t5.side1 ∧
  c.t4.side3 = c.t5.side3

-- Theorem statement
theorem side_x_must_be_green (c : Configuration) 
  (h1 : givenColoring c)
  (h2 : sharedSides c)
  (h3 : ∀ t, t ∈ [c.t1, c.t2, c.t3, c.t4, c.t5] → validTriangle t) :
  c.t4.side3 = Color.Green :=
sorry

end NUMINAMATH_CALUDE_side_x_must_be_green_l674_67407


namespace NUMINAMATH_CALUDE_inequality_proof_l674_67447

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l674_67447


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_no_solution_for_2891_l674_67425

theorem cubic_equation_solutions (n : ℕ+) :
  (∃ (x y : ℤ), x^3 - 3*x*y^2 + y^3 = n) →
  (∃ (x y : ℤ), x^3 - 3*x*y^2 + y^3 = n ∧
                (y-x)^3 - 3*(y-x)*(-x)^2 + (-x)^3 = n ∧
                (-y)^3 - 3*(-y)*(x-y)^2 + (x-y)^3 = n) :=
sorry

theorem no_solution_for_2891 :
  ¬ ∃ (x y : ℤ), x^3 - 3*x*y^2 + y^3 = 2891 :=
sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_no_solution_for_2891_l674_67425


namespace NUMINAMATH_CALUDE_remaining_half_speed_l674_67430

-- Define the given conditions
def total_time : ℝ := 11
def total_distance : ℝ := 300
def first_half_speed : ℝ := 30

-- Define the theorem
theorem remaining_half_speed :
  let first_half_distance : ℝ := total_distance / 2
  let first_half_time : ℝ := first_half_distance / first_half_speed
  let remaining_time : ℝ := total_time - first_half_time
  let remaining_distance : ℝ := total_distance / 2
  (remaining_distance / remaining_time) = 25 := by
  sorry


end NUMINAMATH_CALUDE_remaining_half_speed_l674_67430


namespace NUMINAMATH_CALUDE_exponential_inequality_l674_67413

theorem exponential_inequality (a b : ℝ) (h : a < b) :
  let f := fun (x : ℝ) => Real.exp x
  let A := f b - f a
  let B := (1/2) * (b - a) * (f a + f b)
  A < B := by sorry

end NUMINAMATH_CALUDE_exponential_inequality_l674_67413


namespace NUMINAMATH_CALUDE_cookies_eaten_l674_67474

theorem cookies_eaten (charlie_cookies : ℕ) (father_cookies : ℕ) (mother_cookies : ℕ)
  (h1 : charlie_cookies = 15)
  (h2 : father_cookies = 10)
  (h3 : mother_cookies = 5) :
  charlie_cookies + father_cookies + mother_cookies = 30 := by
sorry

end NUMINAMATH_CALUDE_cookies_eaten_l674_67474


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l674_67437

/-- Given an arithmetic sequence {a_n} with common difference d ≠ 0,
    where a₁, a₃, a₉ form a geometric sequence,
    prove that (a₁ + a₃ + a₉) / (a₂ + a₄ + a₁₀) = 13/16. -/
theorem arithmetic_geometric_ratio 
  (a : ℕ → ℚ) 
  (d : ℚ) 
  (h1 : d ≠ 0) 
  (h2 : ∀ n, a (n + 1) = a n + d) 
  (h3 : (a 3) ^ 2 = a 1 * a 9) :
  (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13 / 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l674_67437


namespace NUMINAMATH_CALUDE_color_film_fraction_l674_67405

theorem color_film_fraction (x y : ℝ) (h : x ≠ 0) :
  let total_bw := 20 * x
  let total_color := 6 * y
  let selected_bw := (y / x) * (total_bw / 100)
  let selected_color := total_color
  let total_selected := selected_bw + selected_color
  (selected_color / total_selected) = 6 / 7 := by
sorry

end NUMINAMATH_CALUDE_color_film_fraction_l674_67405


namespace NUMINAMATH_CALUDE_venus_speed_mph_l674_67477

/-- The speed of Venus in miles per second -/
def venus_speed_mps : ℝ := 21.9

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Theorem: Venus's speed in miles per hour -/
theorem venus_speed_mph : ⌊venus_speed_mps * seconds_per_hour⌋ = 78840 := by
  sorry

end NUMINAMATH_CALUDE_venus_speed_mph_l674_67477


namespace NUMINAMATH_CALUDE_system_solution_l674_67441

theorem system_solution (x y : ℝ) (h1 : 2*x + 3*y = 5) (h2 : 3*x + 2*y = 10) : x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l674_67441


namespace NUMINAMATH_CALUDE_tempo_premium_calculation_l674_67475

/-- Calculate the premium amount for an insured tempo --/
theorem tempo_premium_calculation (original_value : ℝ) (insurance_ratio : ℝ) (premium_rate : ℝ) :
  original_value = 87500 →
  insurance_ratio = 4/5 →
  premium_rate = 0.013 →
  (original_value * insurance_ratio * premium_rate : ℝ) = 910 := by
  sorry

end NUMINAMATH_CALUDE_tempo_premium_calculation_l674_67475


namespace NUMINAMATH_CALUDE_find_divisor_l674_67446

theorem find_divisor (dividend quotient remainder : ℕ) (h1 : dividend = 217) (h2 : quotient = 54) (h3 : remainder = 1) :
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l674_67446


namespace NUMINAMATH_CALUDE_H_composition_equals_neg_one_l674_67435

/-- The function H defined as H(x) = x^2 - 2x - 1 -/
def H (x : ℝ) : ℝ := x^2 - 2*x - 1

/-- Theorem stating that H(H(H(H(H(2))))) = -1 -/
theorem H_composition_equals_neg_one : H (H (H (H (H 2)))) = -1 := by
  sorry

end NUMINAMATH_CALUDE_H_composition_equals_neg_one_l674_67435


namespace NUMINAMATH_CALUDE_circle_tangency_problem_l674_67416

theorem circle_tangency_problem :
  let max_radius : ℕ := 36
  let valid_radius (s : ℕ) : Prop := 1 ≤ s ∧ s < max_radius ∧ max_radius % s = 0
  (Finset.filter valid_radius (Finset.range max_radius)).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangency_problem_l674_67416


namespace NUMINAMATH_CALUDE_books_read_indeterminate_l674_67412

/-- Represents the 'crazy silly school' series --/
structure CrazySillySchool where
  total_movies : ℕ
  total_books : ℕ
  movies_watched : ℕ
  movies_left : ℕ

/-- Theorem stating that the number of books read cannot be uniquely determined --/
theorem books_read_indeterminate (series : CrazySillySchool)
  (h1 : series.total_movies = 8)
  (h2 : series.total_books = 21)
  (h3 : series.movies_watched = 4)
  (h4 : series.movies_left = 4) :
  ∀ n : ℕ, n ≤ series.total_books → ∃ m : ℕ, m ≠ n ∧ m ≤ series.total_books :=
by sorry

end NUMINAMATH_CALUDE_books_read_indeterminate_l674_67412


namespace NUMINAMATH_CALUDE_cat_puppy_weight_difference_l674_67479

/-- The weight difference between cats and puppies -/
theorem cat_puppy_weight_difference :
  let puppy_weights : List ℝ := [6.5, 7.2, 8, 9.5]
  let cat_weight : ℝ := 2.8
  let num_cats : ℕ := 16
  (num_cats : ℝ) * cat_weight - puppy_weights.sum = 13.6 := by
  sorry

end NUMINAMATH_CALUDE_cat_puppy_weight_difference_l674_67479


namespace NUMINAMATH_CALUDE_total_stickers_l674_67472

/-- Given 25 stickers on each page and 35 pages of stickers, 
    the total number of stickers is 875. -/
theorem total_stickers (stickers_per_page pages : ℕ) : 
  stickers_per_page = 25 → pages = 35 → stickers_per_page * pages = 875 := by
  sorry

end NUMINAMATH_CALUDE_total_stickers_l674_67472


namespace NUMINAMATH_CALUDE_fifteenth_digit_is_one_l674_67499

/-- The decimal representation of 1/9 as a sequence of digits after the decimal point -/
def decimal_1_9 : ℕ → ℕ
  | n => 1

/-- The decimal representation of 1/11 as a sequence of digits after the decimal point -/
def decimal_1_11 : ℕ → ℕ
  | n => if n % 3 = 0 then 0 else 9

/-- The sum of the decimal representations of 1/9 and 1/11 as a sequence of digits after the decimal point -/
def sum_decimals : ℕ → ℕ
  | n => (decimal_1_9 n + decimal_1_11 n) % 10

theorem fifteenth_digit_is_one :
  sum_decimals 14 = 1 := by sorry

end NUMINAMATH_CALUDE_fifteenth_digit_is_one_l674_67499


namespace NUMINAMATH_CALUDE_tim_initial_balls_l674_67456

theorem tim_initial_balls (robert_initial : ℕ) (robert_final : ℕ) (tim_initial : ℕ) : 
  robert_initial = 25 → 
  robert_final = 45 → 
  robert_final = robert_initial + tim_initial / 2 → 
  tim_initial = 40 := by
sorry

end NUMINAMATH_CALUDE_tim_initial_balls_l674_67456


namespace NUMINAMATH_CALUDE_train_final_speed_l674_67453

/-- The final speed of a train accelerating from rest -/
def final_speed (acceleration : ℝ) (time : ℝ) : ℝ :=
  acceleration * time

/-- Theorem: The final speed of a train accelerating from rest at 1.2 m/s² for 15 seconds is 18 m/s -/
theorem train_final_speed :
  final_speed 1.2 15 = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_final_speed_l674_67453


namespace NUMINAMATH_CALUDE_new_tax_rate_is_28_percent_l674_67429

/-- Calculates the new tax rate given the initial conditions --/
def calculate_new_tax_rate (initial_rate : ℚ) (income : ℚ) (savings : ℚ) : ℚ :=
  100 * (initial_rate * income - savings) / income

/-- Proves that the new tax rate is 28% given the initial conditions --/
theorem new_tax_rate_is_28_percent :
  let initial_rate : ℚ := 42 / 100
  let income : ℚ := 34500
  let savings : ℚ := 4830
  calculate_new_tax_rate initial_rate income savings = 28 := by
  sorry

#eval calculate_new_tax_rate (42/100) 34500 4830

end NUMINAMATH_CALUDE_new_tax_rate_is_28_percent_l674_67429


namespace NUMINAMATH_CALUDE_triangle_side_length_l674_67421

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  b = Real.sqrt 3 →
  c = 3 →
  B = 30 * π / 180 →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos B →
  a = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l674_67421


namespace NUMINAMATH_CALUDE_problem_solution_l674_67444

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + 1) / Real.log (1/2)

def g (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 6

theorem problem_solution :
  ∀ a : ℝ,
  (∀ x : ℝ, g a x = g a (-x)) →
  (a = 0 ∧ ∀ x > 0, ∀ y > x, g a y > g a x) ∧
  (({x : ℝ | g a x < 0} = {x : ℝ | 2 < x ∧ x < 3}) →
    (∀ x > 1, g a x / (x - 1) ≥ 2 * Real.sqrt 2 - 3) ∧
    (∃ x > 1, g a x / (x - 1) = 2 * Real.sqrt 2 - 3)) ∧
  ((∀ x₁ ≥ 1, ∀ x₂ ∈ Set.Icc (-2) 4, f x₁ ≤ g a x₂) →
    -11/2 ≤ a ∧ a ≤ 2 * Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l674_67444


namespace NUMINAMATH_CALUDE_sin_cos_identity_l674_67486

theorem sin_cos_identity : 
  Real.sin (50 * π / 180) * Real.cos (20 * π / 180) - 
  Real.sin (40 * π / 180) * Real.cos (70 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l674_67486


namespace NUMINAMATH_CALUDE_quadratic_root_range_l674_67410

theorem quadratic_root_range (m : ℝ) : 
  (∃ (α : ℂ), (α.re = 0 ∧ α.im ≠ 0) ∧ 
    (α ^ 2 - (2 * m - 1) * α + m ^ 2 + 1 = 0) ∧
    (Complex.abs α ≤ 2)) →
  (m > -3/4 ∧ m ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l674_67410


namespace NUMINAMATH_CALUDE_sum_difference_theorem_l674_67406

/-- Rounds a number to the nearest multiple of 5, rounding 5s up -/
def roundToNearestFive (n : ℕ) : ℕ :=
  ((n + 2) / 5) * 5

/-- Sums all integers from 1 to n -/
def sumToN (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Sums all integers from 1 to n after rounding each to the nearest multiple of 5 -/
def sumRoundedToN (n : ℕ) : ℕ :=
  List.sum (List.map roundToNearestFive (List.range n))

theorem sum_difference_theorem :
  sumToN 100 - sumRoundedToN 100 = 4750 :=
sorry

end NUMINAMATH_CALUDE_sum_difference_theorem_l674_67406


namespace NUMINAMATH_CALUDE_age_difference_l674_67467

theorem age_difference (A B : ℕ) : B = 37 → A + 10 = 2 * (B - 10) → A - B = 7 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l674_67467


namespace NUMINAMATH_CALUDE_larger_number_puzzle_l674_67428

theorem larger_number_puzzle (x y : ℕ) : 
  x * y = 18 → x + y = 13 → max x y = 9 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_puzzle_l674_67428


namespace NUMINAMATH_CALUDE_micks_to_macks_l674_67489

/-- Given the conversion rates between micks, mocks, and macks, 
    prove that 200/3 micks equal 30 macks. -/
theorem micks_to_macks 
  (h1 : (8 : ℚ) * mick = 3 * mock) 
  (h2 : (5 : ℚ) * mock = 6 * mack) : 
  (200 : ℚ) / 3 * mick = 30 * mack :=
by
  sorry


end NUMINAMATH_CALUDE_micks_to_macks_l674_67489


namespace NUMINAMATH_CALUDE_max_value_quadratic_inequality_l674_67494

/-- Given a quadratic inequality ax² + bx + c > 0 with solution set {x | -1 < x < 3},
    the maximum value of b - c + 1/a is -2 -/
theorem max_value_quadratic_inequality (a b c : ℝ) :
  (∀ x, ax^2 + b*x + c > 0 ↔ -1 < x ∧ x < 3) →
  (∃ m : ℝ, ∀ a' b' c' : ℝ, 
    (∀ x, a'*x^2 + b'*x + c' > 0 ↔ -1 < x ∧ x < 3) →
    b' - c' + 1/a' ≤ m) ∧
  (∃ a₀ b₀ c₀ : ℝ, 
    (∀ x, a₀*x^2 + b₀*x + c₀ > 0 ↔ -1 < x ∧ x < 3) ∧
    b₀ - c₀ + 1/a₀ = -2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_inequality_l674_67494


namespace NUMINAMATH_CALUDE_segment_sum_after_n_halvings_sum_after_million_halvings_l674_67459

/-- The sum of numbers on a segment after n halvings -/
def segmentSum (n : ℕ) : ℕ :=
  3^n + 1

/-- Theorem: The sum of numbers on a segment after n halvings is 3^n + 1 -/
theorem segment_sum_after_n_halvings (n : ℕ) :
  segmentSum n = 3^n + 1 := by
  sorry

/-- Corollary: The sum after one million halvings -/
theorem sum_after_million_halvings :
  segmentSum 1000000 = 3^1000000 + 1 := by
  sorry

end NUMINAMATH_CALUDE_segment_sum_after_n_halvings_sum_after_million_halvings_l674_67459


namespace NUMINAMATH_CALUDE_road_trip_days_l674_67460

/-- Proves that the number of days of the road trip is 3, given the driving hours of Jade and Krista and the total hours driven. -/
theorem road_trip_days (jade_hours krista_hours total_hours : ℕ) 
  (h1 : jade_hours = 8)
  (h2 : krista_hours = 6)
  (h3 : total_hours = 42) :
  (total_hours : ℚ) / (jade_hours + krista_hours : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_days_l674_67460


namespace NUMINAMATH_CALUDE_jack_shirts_per_kid_l674_67449

theorem jack_shirts_per_kid (num_kids : ℕ) (buttons_per_shirt : ℕ) (total_buttons : ℕ) 
  (h1 : num_kids = 3)
  (h2 : buttons_per_shirt = 7)
  (h3 : total_buttons = 63) :
  total_buttons / buttons_per_shirt / num_kids = 3 := by
sorry

end NUMINAMATH_CALUDE_jack_shirts_per_kid_l674_67449


namespace NUMINAMATH_CALUDE_min_value_expression_l674_67450

theorem min_value_expression (a b : ℝ) (ha : a > 1) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y, x > 1 ∧ y > 0 ∧ x + y = 2 → 1 / (a - 1) + 1 / (2 * b) ≤ 1 / (x - 1) + 1 / (2 * y)) ∧
  1 / (a - 1) + 1 / (2 * b) = 3 / 2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l674_67450


namespace NUMINAMATH_CALUDE_alpha_value_l674_67468

theorem alpha_value (α β : ℂ) 
  (h1 : (α + β).im = 0 ∧ (α + β).re > 0)
  (h2 : (Complex.I * (α - 3 * β)).im = 0 ∧ (Complex.I * (α - 3 * β)).re > 0)
  (h3 : β = 4 + 3 * Complex.I) : 
  α = 6 - 3 * Complex.I :=
sorry

end NUMINAMATH_CALUDE_alpha_value_l674_67468


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l674_67424

theorem smallest_base_perfect_square : 
  ∀ b : ℕ, b > 5 → (∀ k : ℕ, k > 5 ∧ k < b → ¬∃ n : ℕ, 4 * k + 5 = n^2) → 
  ∃ n : ℕ, 4 * b + 5 = n^2 → b = 11 := by
sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l674_67424


namespace NUMINAMATH_CALUDE_frame_price_ratio_l674_67431

/-- Calculates the ratio of the price of a smaller frame to the price of an initially intended frame given specific conditions --/
theorem frame_price_ratio (budget : ℚ) (initial_frame_markup : ℚ) (remaining : ℚ) : 
  budget = 60 →
  initial_frame_markup = 0.2 →
  remaining = 6 →
  let initial_frame_price := budget * (1 + initial_frame_markup)
  let smaller_frame_price := budget - remaining
  let ratio := smaller_frame_price / initial_frame_price
  ratio = 3/4 := by
    sorry


end NUMINAMATH_CALUDE_frame_price_ratio_l674_67431


namespace NUMINAMATH_CALUDE_two_problems_without_conditional_l674_67436

/-- Represents a mathematical problem that may or may not require conditional statements in its algorithm. -/
inductive Problem
| OppositeNumber
| SquarePerimeter
| MaximumOfThree
| FunctionValue

/-- Determines if a problem requires conditional statements in its algorithm. -/
def requiresConditional (p : Problem) : Bool :=
  match p with
  | Problem.OppositeNumber => false
  | Problem.SquarePerimeter => false
  | Problem.MaximumOfThree => true
  | Problem.FunctionValue => true

/-- The list of all problems given in the question. -/
def allProblems : List Problem :=
  [Problem.OppositeNumber, Problem.SquarePerimeter, Problem.MaximumOfThree, Problem.FunctionValue]

/-- Theorem stating that the number of problems not requiring conditional statements is 2. -/
theorem two_problems_without_conditional :
  (allProblems.filter (fun p => ¬requiresConditional p)).length = 2 := by
  sorry


end NUMINAMATH_CALUDE_two_problems_without_conditional_l674_67436


namespace NUMINAMATH_CALUDE_quiz_mistakes_l674_67448

theorem quiz_mistakes (total_items : ℕ) (score_percentage : ℚ) : 
  total_items = 25 → score_percentage = 80 / 100 → 
  total_items - (score_percentage * total_items).num = 5 := by
sorry

end NUMINAMATH_CALUDE_quiz_mistakes_l674_67448


namespace NUMINAMATH_CALUDE_movie_ticket_price_l674_67442

/-- The price of a movie ticket and nachos, where the nachos cost half the ticket price and the total is $24. -/
def MovieTheaterVisit : Type :=
  {ticket : ℚ // ∃ (nachos : ℚ), nachos = ticket / 2 ∧ ticket + nachos = 24}

/-- Theorem stating that the price of the movie ticket is $16. -/
theorem movie_ticket_price (visit : MovieTheaterVisit) : visit.val = 16 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_price_l674_67442


namespace NUMINAMATH_CALUDE_max_value_of_a_max_value_is_negative_two_l674_67490

theorem max_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x < a → |x| > 2) ∧ 
  (∃ x : ℝ, |x| > 2 ∧ x ≥ a) →
  a ≤ -2 :=
by sorry

theorem max_value_is_negative_two :
  ∃ a : ℝ, 
    (∀ x : ℝ, x < a → |x| > 2) ∧
    (∃ x : ℝ, |x| > 2 ∧ x ≥ a) ∧
    a = -2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_max_value_is_negative_two_l674_67490


namespace NUMINAMATH_CALUDE_ceiling_floor_product_l674_67481

theorem ceiling_floor_product (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_product_l674_67481


namespace NUMINAMATH_CALUDE_three_dollar_neg_one_l674_67497

def dollar_op (a b : ℤ) : ℤ := a * (b + 2) + a * b

theorem three_dollar_neg_one : dollar_op 3 (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_three_dollar_neg_one_l674_67497


namespace NUMINAMATH_CALUDE_above_x_axis_on_line_l674_67461

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 + 5*m + 6) (m^2 - 2*m - 15)

-- Statement for the first part of the problem
theorem above_x_axis (m : ℝ) : 
  Complex.im (z m) > 0 ↔ m < -3 ∨ m > 5 := by sorry

-- Statement for the second part of the problem
theorem on_line (m : ℝ) :
  Complex.re (z m) + Complex.im (z m) + 5 = 0 ↔ 
  m = (-3 + Real.sqrt 41) / 4 ∨ m = (-3 - Real.sqrt 41) / 4 := by sorry

end NUMINAMATH_CALUDE_above_x_axis_on_line_l674_67461


namespace NUMINAMATH_CALUDE_bus_seating_capacity_l674_67471

theorem bus_seating_capacity : 
  ∀ (total_students : ℕ) (bus_capacity : ℕ),
    (4 * bus_capacity + 30 = total_students) →
    (5 * bus_capacity = total_students + 10) →
    bus_capacity = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_bus_seating_capacity_l674_67471


namespace NUMINAMATH_CALUDE_tan_theta_value_l674_67452

theorem tan_theta_value (θ : Real) : 
  Real.tan (π / 4 + θ) = 1 / 2 → Real.tan θ = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l674_67452


namespace NUMINAMATH_CALUDE_line_slopes_problem_l674_67438

theorem line_slopes_problem (k₁ k₂ b : ℝ) : 
  (2 * k₁^2 - 3 * k₁ - b = 0) → 
  (2 * k₂^2 - 3 * k₂ - b = 0) → 
  ((k₁ * k₂ = -1) → b = 2) ∧ 
  ((k₁ = k₂) → b = -9/8) := by
sorry

end NUMINAMATH_CALUDE_line_slopes_problem_l674_67438


namespace NUMINAMATH_CALUDE_solve_equation_l674_67478

theorem solve_equation : ∃ r : ℝ, 5 * (r - 9) = 6 * (3 - 3 * r) + 6 ∧ r = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l674_67478


namespace NUMINAMATH_CALUDE_additional_pecks_needed_to_fill_barrel_l674_67496

-- Define the relationships
def peck_to_bushel : ℚ := 1/4
def bushel_to_barrel : ℚ := 1/9

-- Define the number of pecks already picked
def pecks_picked : ℕ := 1

-- Theorem statement
theorem additional_pecks_needed_to_fill_barrel : 
  ∀ (pecks_in_barrel : ℕ), 
    pecks_in_barrel = (1 / peck_to_bushel : ℚ) * (1 / bushel_to_barrel : ℚ) → 
    pecks_in_barrel - pecks_picked = 35 := by
  sorry

end NUMINAMATH_CALUDE_additional_pecks_needed_to_fill_barrel_l674_67496


namespace NUMINAMATH_CALUDE_factorization_proof_l674_67498

theorem factorization_proof (x : ℝ) : 18 * x^3 + 12 * x^2 = 6 * x^2 * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l674_67498


namespace NUMINAMATH_CALUDE_train_length_calculation_l674_67463

/-- Calculates the length of a train given the speeds of a jogger and the train, 
    the initial distance between them, and the time it takes for the train to pass the jogger. -/
def train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) : ℝ :=
  (train_speed - jogger_speed) * passing_time - initial_distance

/-- Theorem stating that given the specific conditions, the train length is 120 meters. -/
theorem train_length_calculation : 
  train_length (9 * (1000 / 3600)) (45 * (1000 / 3600)) 250 37 = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l674_67463


namespace NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l674_67488

theorem cubic_polynomials_common_roots :
  ∃! (c d : ℝ), ∃ (r s : ℝ),
    r ≠ s ∧
    (r^3 + c*r^2 + 17*r + 10 = 0) ∧
    (r^3 + d*r^2 + 22*r + 14 = 0) ∧
    (s^3 + c*s^2 + 17*s + 10 = 0) ∧
    (s^3 + d*s^2 + 22*s + 14 = 0) ∧
    (∀ (x : ℝ), x ≠ r ∧ x ≠ s →
      (x^3 + c*x^2 + 17*x + 10 ≠ 0) ∨
      (x^3 + d*x^2 + 22*x + 14 ≠ 0)) ∧
    c = 8 ∧
    d = 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l674_67488


namespace NUMINAMATH_CALUDE_janine_read_150_pages_l674_67484

/-- The number of pages Janine read in two months -/
def pages_read_in_two_months (books_last_month : ℕ) (books_this_month_factor : ℕ) (pages_per_book : ℕ) : ℕ :=
  (books_last_month + books_last_month * books_this_month_factor) * pages_per_book

/-- Theorem stating that Janine read 150 pages in two months -/
theorem janine_read_150_pages :
  pages_read_in_two_months 5 2 10 = 150 := by
  sorry

#eval pages_read_in_two_months 5 2 10

end NUMINAMATH_CALUDE_janine_read_150_pages_l674_67484


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l674_67492

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_second_term
  (a : ℕ → ℝ)
  (h_geo : IsGeometricSequence a)
  (h_first : a 1 = 2)
  (h_relation : 16 * a 3 * a 5 = 8 * a 4 - 1) :
  a 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l674_67492


namespace NUMINAMATH_CALUDE_min_additional_marbles_l674_67422

/-- The number of friends Tom has -/
def num_friends : ℕ := 12

/-- The initial number of marbles Tom has -/
def initial_marbles : ℕ := 40

/-- The sum of consecutive integers from 1 to n -/
def sum_consecutive (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating the minimum number of additional marbles Tom needs -/
theorem min_additional_marbles : 
  sum_consecutive num_friends - initial_marbles = 38 := by sorry

end NUMINAMATH_CALUDE_min_additional_marbles_l674_67422


namespace NUMINAMATH_CALUDE_inequality_solution_l674_67457

theorem inequality_solution (x : ℝ) : (2 - x) / 3 + 2 > x - (x - 2) / 2 → x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l674_67457


namespace NUMINAMATH_CALUDE_vector_coordinates_proof_l674_67493

theorem vector_coordinates_proof :
  ∀ (a b : ℝ × ℝ),
    (‖a‖ = 3) →
    (b = (1, 2)) →
    (a.1 * b.1 + a.2 * b.2 = 0) →
    ((a = (-6 * Real.sqrt 5 / 5, 3 * Real.sqrt 5 / 5)) ∨
     (a = (6 * Real.sqrt 5 / 5, -3 * Real.sqrt 5 / 5))) := by
  sorry

end NUMINAMATH_CALUDE_vector_coordinates_proof_l674_67493


namespace NUMINAMATH_CALUDE_triangle_area_angle_l674_67404

theorem triangle_area_angle (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < C → C < π →
  S = (1/4) * (a^2 + b^2 - c^2) →
  S = (1/2) * a * b * Real.sin C →
  C = π/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_angle_l674_67404


namespace NUMINAMATH_CALUDE_animal_lifespan_l674_67451

theorem animal_lifespan (bat_lifespan hamster_lifespan frog_lifespan : ℕ) : 
  bat_lifespan = 10 →
  hamster_lifespan = bat_lifespan - 6 →
  frog_lifespan = 4 * hamster_lifespan →
  bat_lifespan + hamster_lifespan + frog_lifespan = 30 := by
  sorry

end NUMINAMATH_CALUDE_animal_lifespan_l674_67451


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l674_67445

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l674_67445


namespace NUMINAMATH_CALUDE_ackermann_3_2_l674_67458

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem ackermann_3_2 : A 3 2 = 29 := by sorry

end NUMINAMATH_CALUDE_ackermann_3_2_l674_67458


namespace NUMINAMATH_CALUDE_complex_equation_solution_l674_67401

theorem complex_equation_solution (z : ℂ) : (3 + Complex.I) * z = 4 - 2 * Complex.I → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l674_67401


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l674_67465

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 4 * 2 = k) :
  x * (-5) = k → x = -8/5 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l674_67465


namespace NUMINAMATH_CALUDE_equation_solution_l674_67440

theorem equation_solution (m : ℝ) : 
  (∃ x : ℝ, x = 3 ∧ 4 * (x - 1) - m * x + 6 = 8) → 
  m^2 + 2*m - 3 = 5 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l674_67440


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l674_67411

theorem complex_modulus_problem (z : ℂ) : (1 + Complex.I) * z = 2 * Complex.I → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l674_67411


namespace NUMINAMATH_CALUDE_congruence_problem_l674_67402

theorem congruence_problem (c d : ℤ) (h1 : c ≡ 25 [ZMOD 53]) (h2 : d ≡ 88 [ZMOD 53]) :
  ∃ m : ℤ, m ≥ 150 ∧ c - d ≡ m [ZMOD 53] ∧ ∀ k : ℤ, k ≥ 150 ∧ c - d ≡ k [ZMOD 53] → m ≤ k :=
by sorry

end NUMINAMATH_CALUDE_congruence_problem_l674_67402


namespace NUMINAMATH_CALUDE_solution_set_f_positive_range_of_m_l674_67473

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| - |x + 2|

-- Theorem for part I
theorem solution_set_f_positive :
  {x : ℝ | f x > 0} = {x : ℝ | x < -1/3 ∨ x > 3} := by sorry

-- Theorem for part II
theorem range_of_m (h : ∃ x₀ : ℝ, f x₀ + 2*m^2 < 4*m) :
  -1/2 < m ∧ m < 5/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_positive_range_of_m_l674_67473


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l674_67423

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - 3 * y + 5 = 0

-- Define the point that the parallel line passes through
def point : ℝ × ℝ := (-2, 1)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 2 * x - 3 * y + 7 = 0

-- Theorem statement
theorem parallel_line_through_point :
  (∀ x y : ℝ, given_line x y ↔ 2 * x - 3 * y + 5 = 0) →
  parallel_line point.1 point.2 ∧
  (∀ x y : ℝ, parallel_line x y ↔ 2 * x - 3 * y + 7 = 0) ∧
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, given_line x y ↔ parallel_line x y) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l674_67423


namespace NUMINAMATH_CALUDE_time_difference_to_halfway_l674_67454

/-- The time difference to reach the halfway point between two runners --/
theorem time_difference_to_halfway (danny_time steve_time : ℝ) : 
  danny_time = 33 → steve_time = 2 * danny_time → 
  (steve_time / 2) - (danny_time / 2) = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_to_halfway_l674_67454


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l674_67485

theorem least_positive_integer_with_remainders : ∃ (M : ℕ), 
  (M > 0) ∧
  (M % 7 = 6) ∧
  (M % 8 = 7) ∧
  (M % 9 = 8) ∧
  (M % 10 = 9) ∧
  (M % 11 = 10) ∧
  (M % 12 = 11) ∧
  (∀ (N : ℕ), N > 0 ∧ 
    N % 7 = 6 ∧
    N % 8 = 7 ∧
    N % 9 = 8 ∧
    N % 10 = 9 ∧
    N % 11 = 10 ∧
    N % 12 = 11 → M ≤ N) ∧
  M = 27719 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l674_67485


namespace NUMINAMATH_CALUDE_triangle_side_length_l674_67464

theorem triangle_side_length (A B : Real) (a b : Real) :
  A = 30 * π / 180 →
  B = 45 * π / 180 →
  a = 1 →
  b = a * Real.sin B / Real.sin A →
  b = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l674_67464


namespace NUMINAMATH_CALUDE_sunset_time_proof_l674_67470

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Represents duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat

/-- Calculates the end time given a start time and duration -/
def calculateEndTime (start : Time) (duration : Duration) : Time :=
  let totalMinutes := start.hours * 60 + start.minutes + duration.hours * 60 + duration.minutes
  { hours := totalMinutes / 60,
    minutes := totalMinutes % 60 }

/-- Converts 24-hour time to 12-hour time -/
def to12HourFormat (time : Time) : Time :=
  if time.hours ≥ 12 then
    { hours := if time.hours = 12 then 12 else time.hours - 12,
      minutes := time.minutes }
  else
    time

theorem sunset_time_proof :
  let sunrise : Time := { hours := 6, minutes := 32 }
  let daylight : Duration := { hours := 11, minutes := 18 }
  let sunset := calculateEndTime sunrise daylight
  to12HourFormat sunset = { hours := 5, minutes := 50 } :=
by sorry

end NUMINAMATH_CALUDE_sunset_time_proof_l674_67470


namespace NUMINAMATH_CALUDE_rayden_vs_lily_birds_l674_67408

theorem rayden_vs_lily_birds (lily_ducks lily_geese lily_chickens lily_pigeons : ℕ)
  (rayden_ducks rayden_geese rayden_chickens rayden_pigeons : ℕ)
  (h1 : lily_ducks = 20)
  (h2 : lily_geese = 10)
  (h3 : lily_chickens = 5)
  (h4 : lily_pigeons = 30)
  (h5 : rayden_ducks = 3 * lily_ducks)
  (h6 : rayden_geese = 4 * lily_geese)
  (h7 : rayden_chickens = 5 * lily_chickens)
  (h8 : lily_pigeons = 2 * rayden_pigeons) :
  (rayden_ducks + rayden_geese + rayden_chickens + rayden_pigeons) -
  (lily_ducks + lily_geese + lily_chickens + lily_pigeons) = 75 :=
by sorry

end NUMINAMATH_CALUDE_rayden_vs_lily_birds_l674_67408


namespace NUMINAMATH_CALUDE_dividend_calculation_l674_67434

theorem dividend_calculation (quotient divisor remainder : ℕ) 
  (h_quotient : quotient = 36)
  (h_divisor : divisor = 85)
  (h_remainder : remainder = 26) :
  (divisor * quotient) + remainder = 3086 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l674_67434


namespace NUMINAMATH_CALUDE_figure_to_square_l674_67487

/-- A figure that can be cut into three parts -/
structure Figure where
  area : ℕ

/-- Proves that a figure with an area of 57 unit squares can be assembled into a square -/
theorem figure_to_square (f : Figure) (h : f.area = 57) : 
  ∃ (s : ℝ), s^2 = f.area := by
  sorry

end NUMINAMATH_CALUDE_figure_to_square_l674_67487
