import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_l183_18310

theorem system_solution :
  ∃ (x y : ℚ), 7 * x - 14 * y = 3 ∧ 3 * y - x = 5 ∧ x = 79 / 7 ∧ y = 38 / 7 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l183_18310


namespace NUMINAMATH_CALUDE_train_passing_time_l183_18361

/-- The time it takes for a train to pass a person moving in the opposite direction --/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (person_speed : ℝ) :
  train_length = 110 →
  train_speed = 84 * (5 / 18) →
  person_speed = 6 * (5 / 18) →
  (train_length / (train_speed + person_speed)) = 4.4 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l183_18361


namespace NUMINAMATH_CALUDE_madeline_max_distance_difference_l183_18387

-- Define the speeds and durations
def madeline_speed : ℝ := 12
def madeline_time : ℝ := 3
def max_speed : ℝ := 15
def max_time : ℝ := 2

-- Define the distance function
def distance (speed time : ℝ) : ℝ := speed * time

-- Theorem statement
theorem madeline_max_distance_difference :
  distance madeline_speed madeline_time - distance max_speed max_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_madeline_max_distance_difference_l183_18387


namespace NUMINAMATH_CALUDE_max_value_of_f_l183_18344

noncomputable def f (x : ℝ) : ℝ := (1 + Real.sqrt 3 * Real.tan x) * Real.cos x

theorem max_value_of_f :
  ∃ (M : ℝ), M = Real.sqrt 3 ∧
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 6) → f x ≤ M) ∧
  (∃ x, x ∈ Set.Icc 0 (Real.pi / 6) ∧ f x = M) := by
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l183_18344


namespace NUMINAMATH_CALUDE_planes_through_three_points_l183_18347

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define collinearity for three points
def collinear (p1 p2 p3 : Point3D) : Prop :=
  ∃ (t : ℝ), (p3.x - p1.x) = t * (p2.x - p1.x) ∧
             (p3.y - p1.y) = t * (p2.y - p1.y) ∧
             (p3.z - p1.z) = t * (p2.z - p1.z)

-- Define a function to count the number of planes through three points
def count_planes (p1 p2 p3 : Point3D) : Nat ⊕ Nat → Prop
  | Sum.inl 1 => ¬collinear p1 p2 p3
  | Sum.inr 0 => collinear p1 p2 p3
  | _ => False

-- Theorem statement
theorem planes_through_three_points (p1 p2 p3 : Point3D) :
  (count_planes p1 p2 p3 (Sum.inl 1)) ∨ (count_planes p1 p2 p3 (Sum.inr 0)) :=
sorry

end NUMINAMATH_CALUDE_planes_through_three_points_l183_18347


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l183_18332

theorem binomial_coefficient_two (n : ℕ) (h : n > 0) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l183_18332


namespace NUMINAMATH_CALUDE_hidden_numbers_average_l183_18314

/-- A card with two numbers -/
structure Card where
  visible : ℕ
  hidden : ℕ

/-- The problem setup -/
def problem_setup (cards : Fin 3 → Card) : Prop :=
  -- The sums of the numbers on each card are the same
  (∃ s : ℕ, ∀ i : Fin 3, (cards i).visible + (cards i).hidden = s) ∧
  -- Visible numbers are 81, 52, and 47
  (cards 0).visible = 81 ∧ (cards 1).visible = 52 ∧ (cards 2).visible = 47 ∧
  -- Hidden numbers are all prime
  (∀ i : Fin 3, Nat.Prime (cards i).hidden) ∧
  -- All numbers are different
  (∀ i j : Fin 3, i ≠ j → (cards i).visible ≠ (cards j).visible ∧ 
                         (cards i).hidden ≠ (cards j).hidden ∧
                         (cards i).visible ≠ (cards j).hidden)

/-- The theorem to prove -/
theorem hidden_numbers_average (cards : Fin 3 → Card) 
  (h : problem_setup cards) : 
  (cards 0).hidden + (cards 1).hidden + (cards 2).hidden = 119 := by
  sorry

#check hidden_numbers_average

end NUMINAMATH_CALUDE_hidden_numbers_average_l183_18314


namespace NUMINAMATH_CALUDE_a_2009_equals_7_l183_18391

/-- Defines the array structure as described in the problem -/
def array_element (n i : ℕ) : ℚ :=
  if i ≤ n then i / (n + 1 - i) else 0

/-- Defines the index of the last element in the nth array -/
def last_index (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The main theorem stating that the 2009th element of the array is 7 -/
theorem a_2009_equals_7 : array_element 63 56 = 7 := by sorry

end NUMINAMATH_CALUDE_a_2009_equals_7_l183_18391


namespace NUMINAMATH_CALUDE_f_seven_half_value_l183_18315

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_seven_half_value 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : ∀ x, f (x + 2) = -f x)
  (h_unit : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x) :
  f 7.5 = -0.5 := by
  sorry

end NUMINAMATH_CALUDE_f_seven_half_value_l183_18315


namespace NUMINAMATH_CALUDE_derivative_of_2ln_derivative_of_exp_div_x_l183_18349

-- Function 1: f(x) = 2ln(x)
theorem derivative_of_2ln (x : ℝ) (h : x > 0) : 
  deriv (fun x => 2 * Real.log x) x = 2 / x := by sorry

-- Function 2: f(x) = e^x / x
theorem derivative_of_exp_div_x (x : ℝ) (h : x ≠ 0) : 
  deriv (fun x => Real.exp x / x) x = (Real.exp x * x - Real.exp x) / x^2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_2ln_derivative_of_exp_div_x_l183_18349


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l183_18359

theorem trigonometric_equation_solution (x : ℝ) : 
  (∃ (n : ℤ), x = Real.pi / 2 * (2 * ↑n + 1)) ∨ 
  (∃ (k : ℤ), x = Real.pi / 18 * (4 * ↑k + 1)) ↔ 
  Real.sin (3 * x) + Real.sin (5 * x) = 2 * (Real.cos (2 * x))^2 - 2 * (Real.sin (3 * x))^2 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l183_18359


namespace NUMINAMATH_CALUDE_video_game_spend_l183_18331

/-- Calculates the amount spent on video games given total pocket money and fractions spent on other items --/
def video_game_expenditure (total : ℚ) (books : ℚ) (snacks : ℚ) (toys : ℚ) : ℚ :=
  total - (books * total + snacks * total + toys * total)

/-- Theorem stating that the amount spent on video games is 6 dollars --/
theorem video_game_spend :
  let total : ℚ := 40
  let books : ℚ := 2 / 5
  let snacks : ℚ := 1 / 4
  let toys : ℚ := 1 / 5
  video_game_expenditure total books snacks toys = 6 := by
  sorry

end NUMINAMATH_CALUDE_video_game_spend_l183_18331


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l183_18365

theorem cubic_sum_theorem (a b c d : ℕ) 
  (h : (a + b + c + d) * (a^2 + b^2 + c^2 + d^2)^2 = 2023) : 
  a^3 + b^3 + c^3 + d^3 = 43 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l183_18365


namespace NUMINAMATH_CALUDE_pet_store_dogs_l183_18373

theorem pet_store_dogs (dogs : ℕ) : 
  dogs + (dogs / 2) + (2 * dogs) + (3 * dogs) = 39 → dogs = 6 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_dogs_l183_18373


namespace NUMINAMATH_CALUDE_art_students_count_l183_18374

/-- Given a high school with the following student enrollment:
  * 500 total students
  * 50 students taking music
  * 10 students taking both music and art
  * 440 students taking neither music nor art
  Prove that the number of students taking art is 20. -/
theorem art_students_count (total : ℕ) (music : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 500)
  (h2 : music = 50)
  (h3 : both = 10)
  (h4 : neither = 440) :
  total - music - neither + both = 20 := by
  sorry

#check art_students_count

end NUMINAMATH_CALUDE_art_students_count_l183_18374


namespace NUMINAMATH_CALUDE_final_card_values_card_game_2004_l183_18300

def card_game (n : ℕ) : ℕ :=
  3^(2*n) - 2 * 3^n + 2

theorem final_card_values (n : ℕ) :
  let initial_cards := 3^(2*n)
  let final_values := card_game n
  ∀ c : ℕ, c ≥ 3^n ∧ c ≤ 3^(2*n) - 3^n + 1 →
    c ∈ Finset.range final_values :=
by sorry

theorem card_game_2004 :
  card_game 1002 = 3^2004 - 2 * 3^1002 + 2 :=
by sorry

end NUMINAMATH_CALUDE_final_card_values_card_game_2004_l183_18300


namespace NUMINAMATH_CALUDE_orchestra_admission_l183_18333

theorem orchestra_admission (initial_ratio_violinists : ℝ) (initial_ratio_cellists : ℝ) (initial_ratio_trumpeters : ℝ)
  (violinist_increase : ℝ) (cellist_decrease : ℝ) (total_admitted : ℕ) :
  initial_ratio_violinists = 1.6 →
  initial_ratio_cellists = 1 →
  initial_ratio_trumpeters = 0.4 →
  violinist_increase = 0.25 →
  cellist_decrease = 0.2 →
  total_admitted = 32 →
  ∃ (violinists cellists trumpeters : ℕ),
    violinists = 20 ∧
    cellists = 8 ∧
    trumpeters = 4 ∧
    violinists + cellists + trumpeters = total_admitted :=
by sorry

end NUMINAMATH_CALUDE_orchestra_admission_l183_18333


namespace NUMINAMATH_CALUDE_mod_equivalence_solution_l183_18358

theorem mod_equivalence_solution : ∃ (n : ℕ), n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_solution_l183_18358


namespace NUMINAMATH_CALUDE_production_days_l183_18368

theorem production_days (n : ℕ) 
  (h1 : (40 * n) / n = 40)  -- Average daily production for past n days
  (h2 : ((40 * n + 90) : ℝ) / (n + 1) = 45) : n = 9 :=
by sorry

end NUMINAMATH_CALUDE_production_days_l183_18368


namespace NUMINAMATH_CALUDE_initial_flower_plates_is_four_l183_18339

/-- Represents the initial number of flower pattern plates Jack has. -/
def initial_flower_plates : ℕ := sorry

/-- Represents the number of checked pattern plates Jack has. -/
def checked_plates : ℕ := 8

/-- Represents the number of polka dotted plates Jack buys. -/
def polka_dotted_plates : ℕ := 2 * checked_plates

/-- Represents the total number of plates Jack has after buying polka dotted plates and smashing one flower plate. -/
def total_plates : ℕ := 27

/-- Theorem stating that the initial number of flower pattern plates is 4. -/
theorem initial_flower_plates_is_four :
  initial_flower_plates = 4 :=
by
  have h1 : initial_flower_plates + checked_plates + polka_dotted_plates - 1 = total_plates := by sorry
  sorry

end NUMINAMATH_CALUDE_initial_flower_plates_is_four_l183_18339


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l183_18392

/-- The atomic weight of Barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of Sulphur in g/mol -/
def atomic_weight_S : ℝ := 32.07

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Barium atoms in the compound -/
def num_Ba : ℕ := 1

/-- The number of Sulphur atoms in the compound -/
def num_S : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 4

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 
  num_Ba * atomic_weight_Ba + num_S * atomic_weight_S + num_O * atomic_weight_O

theorem compound_molecular_weight : 
  molecular_weight = 233.40 :=
by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l183_18392


namespace NUMINAMATH_CALUDE_integer_sum_problem_l183_18346

theorem integer_sum_problem (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 180) : x + y = 28 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l183_18346


namespace NUMINAMATH_CALUDE_apple_ratio_l183_18325

/-- Proves the ratio of wormy apples to total apples given specific conditions -/
theorem apple_ratio (total : ℕ) (raw : ℕ) (bruised : ℕ) (wormy : ℕ)
  (h1 : total = 85)
  (h2 : raw = 42)
  (h3 : bruised = total / 5 + 9)
  (h4 : wormy = total - bruised - raw) :
  (wormy : ℚ) / total = 17 / 85 := by
  sorry

end NUMINAMATH_CALUDE_apple_ratio_l183_18325


namespace NUMINAMATH_CALUDE_least_possible_smallest_integer_l183_18340

theorem least_possible_smallest_integer
  (a b c d : ℤ) -- Four different integers
  (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) -- Integers are different
  (h_avg : (a + b + c + d) / 4 = 74) -- Average is 74
  (h_max : d = 90) -- Largest integer is 90
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d) -- Ordering of integers
  : a ≥ 31 := by
sorry

end NUMINAMATH_CALUDE_least_possible_smallest_integer_l183_18340


namespace NUMINAMATH_CALUDE_orange_sum_l183_18388

theorem orange_sum : 
  let tree1 : ℕ := 80
  let tree2 : ℕ := 60
  let tree3 : ℕ := 120
  let tree4 : ℕ := 45
  let tree5 : ℕ := 25
  let tree6 : ℕ := 97
  tree1 + tree2 + tree3 + tree4 + tree5 + tree6 = 427 := by
sorry

end NUMINAMATH_CALUDE_orange_sum_l183_18388


namespace NUMINAMATH_CALUDE_max_c_value_l183_18323

theorem max_c_value (c d : ℝ) (h : 5 * c + (d - 12)^2 = 235) :
  c ≤ 47 ∧ ∃ d', 5 * 47 + (d' - 12)^2 = 235 := by
  sorry

end NUMINAMATH_CALUDE_max_c_value_l183_18323


namespace NUMINAMATH_CALUDE_smallest_third_altitude_nine_is_achievable_l183_18301

/-- Represents a triangle with altitudes --/
structure TriangleWithAltitudes where
  /-- The lengths of the three altitudes --/
  altitudes : Fin 3 → ℝ
  /-- At least two altitudes are positive --/
  two_positive : ∃ (i j : Fin 3), i ≠ j ∧ altitudes i > 0 ∧ altitudes j > 0

/-- The proposition to be proved --/
theorem smallest_third_altitude 
  (t : TriangleWithAltitudes) 
  (h1 : t.altitudes 0 = 6) 
  (h2 : t.altitudes 1 = 18) 
  (h3 : ∃ (n : ℕ), t.altitudes 2 = n) :
  t.altitudes 2 ≥ 9 := by
sorry

/-- The proposition that 9 is achievable --/
theorem nine_is_achievable : 
  ∃ (t : TriangleWithAltitudes), 
    t.altitudes 0 = 6 ∧ 
    t.altitudes 1 = 18 ∧ 
    t.altitudes 2 = 9 := by
sorry

end NUMINAMATH_CALUDE_smallest_third_altitude_nine_is_achievable_l183_18301


namespace NUMINAMATH_CALUDE_f_zero_eq_five_l183_18397

/-- Given a function f such that f(x-2) = 2^x - x + 3 for all x, prove that f(0) = 5 -/
theorem f_zero_eq_five (f : ℝ → ℝ) (h : ∀ x, f (x - 2) = 2^x - x + 3) : f 0 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_eq_five_l183_18397


namespace NUMINAMATH_CALUDE_line_through_origin_and_third_quadrant_l183_18307

/-- A line in 2D space represented by the equation Ax - By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Predicate to check if a point (x, y) lies on a given line -/
def Line.contains (L : Line) (x y : ℝ) : Prop :=
  L.A * x - L.B * y + L.C = 0

/-- Predicate to check if a line passes through the origin -/
def Line.passes_through_origin (L : Line) : Prop :=
  L.contains 0 0

/-- Predicate to check if a line passes through the third quadrant -/
def Line.passes_through_third_quadrant (L : Line) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ L.contains x y

/-- Theorem stating the properties of a line passing through the origin and third quadrant -/
theorem line_through_origin_and_third_quadrant (L : Line) :
  L.passes_through_origin ∧ L.passes_through_third_quadrant →
  L.A * L.B < 0 ∧ L.C = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_through_origin_and_third_quadrant_l183_18307


namespace NUMINAMATH_CALUDE_greatest_b_value_l183_18311

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 8*x - 15 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 8*5 - 15 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_value_l183_18311


namespace NUMINAMATH_CALUDE_coffee_bean_price_proof_l183_18394

/-- The price of the first type of coffee bean -/
def first_bean_price : ℝ := 33

/-- The price of the second type of coffee bean -/
def second_bean_price : ℝ := 12

/-- The total weight of the mixture in pounds -/
def total_mixture_weight : ℝ := 100

/-- The selling price of the mixture per pound -/
def mixture_price_per_pound : ℝ := 11.25

/-- The weight of each type of bean used in the mixture -/
def each_bean_weight : ℝ := 25

theorem coffee_bean_price_proof : 
  first_bean_price * each_bean_weight + 
  second_bean_price * each_bean_weight = 
  total_mixture_weight * mixture_price_per_pound :=
by sorry

end NUMINAMATH_CALUDE_coffee_bean_price_proof_l183_18394


namespace NUMINAMATH_CALUDE_journey_mpg_approx_30_3_l183_18348

/-- Calculates the average miles per gallon for a car journey -/
def average_mpg (initial_odometer final_odometer : ℕ) (gas_fills : List ℕ) : ℚ :=
  let total_distance := final_odometer - initial_odometer
  let total_gas := gas_fills.sum
  (total_distance : ℚ) / total_gas

/-- The average miles per gallon for the given journey is approximately 30.3 -/
theorem journey_mpg_approx_30_3 :
  let initial_odometer := 34650
  let final_odometer := 35800
  let gas_fills := [8, 10, 15, 5]
  let mpg := average_mpg initial_odometer final_odometer gas_fills
  ∃ ε > 0, abs (mpg - 30.3) < ε ∧ ε < 0.1 := by
  sorry

#eval average_mpg 34650 35800 [8, 10, 15, 5]

end NUMINAMATH_CALUDE_journey_mpg_approx_30_3_l183_18348


namespace NUMINAMATH_CALUDE_digit_150_of_17_150_l183_18308

/-- The decimal representation of 17/150 -/
def decimal_rep : ℚ := 17 / 150

/-- The nth digit after the decimal point in a rational number -/
def nth_digit (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem digit_150_of_17_150 :
  nth_digit decimal_rep 150 = 3 :=
sorry

end NUMINAMATH_CALUDE_digit_150_of_17_150_l183_18308


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l183_18376

theorem least_positive_integer_with_remainder_one (n : ℕ) : n = 2311 ↔ 
  (n > 1) ∧ 
  (∀ d ∈ ({2, 3, 5, 7, 11} : Set ℕ), n % d = 1) ∧ 
  (∀ m : ℕ, m > 1 → (∀ d ∈ ({2, 3, 5, 7, 11} : Set ℕ), m % d = 1) → m ≥ n) := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainder_one_l183_18376


namespace NUMINAMATH_CALUDE_inequality_proof_l183_18324

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2*a + b + c)^2 / (2*a^2 + (b + c)^2) +
  (2*b + a + c)^2 / (2*b^2 + (c + a)^2) +
  (2*c + a + b)^2 / (2*c^2 + (a + b)^2) ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l183_18324


namespace NUMINAMATH_CALUDE_inequality_proof_l183_18367

/-- Given a function f: ℝ → ℝ with derivative f', such that ∀ x ∈ ℝ, f x > f' x,
    prove that 2023 * f (Real.log 2022) > 2022 * f (Real.log 2023) -/
theorem inequality_proof (f : ℝ → ℝ) (f' : ℝ → ℝ) (hf : ∀ x : ℝ, HasDerivAt f (f' x) x)
    (h : ∀ x : ℝ, f x > f' x) :
  2023 * f (Real.log 2022) > 2022 * f (Real.log 2023) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l183_18367


namespace NUMINAMATH_CALUDE_calculate_expression_l183_18312

theorem calculate_expression : 24 / (-6) * (3/2) / (-4/3) = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l183_18312


namespace NUMINAMATH_CALUDE_window_width_is_20_inches_l183_18316

/-- Represents the dimensions of a glass pane -/
structure PaneDimensions where
  width : ℝ
  height : ℝ

/-- Represents the configuration of a window -/
structure WindowConfig where
  pane : PaneDimensions
  columns : ℕ
  rows : ℕ
  borderWidth : ℝ

/-- Calculates the total width of a window given its configuration -/
def totalWidth (config : WindowConfig) : ℝ :=
  config.columns * config.pane.width + (config.columns + 1) * config.borderWidth

/-- Theorem stating the total width of the window is 20 inches -/
theorem window_width_is_20_inches (config : WindowConfig) 
  (h1 : config.columns = 3)
  (h2 : config.rows = 2)
  (h3 : config.pane.height = 3 * config.pane.width)
  (h4 : config.borderWidth = 2) :
  totalWidth config = 20 := by
  sorry

#check window_width_is_20_inches

end NUMINAMATH_CALUDE_window_width_is_20_inches_l183_18316


namespace NUMINAMATH_CALUDE_remainder_17_pow_63_mod_7_l183_18352

theorem remainder_17_pow_63_mod_7 : 17^63 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_pow_63_mod_7_l183_18352


namespace NUMINAMATH_CALUDE_range_of_b_minus_a_l183_18305

theorem range_of_b_minus_a (a b : ℝ) : 
  (a < b) →
  (∀ x : ℝ, (a ≤ x ∧ x ≤ b) → (x^2 + x - 2 ≤ 0)) →
  (∃ x : ℝ, (x^2 + x - 2 ≤ 0) ∧ ¬(a ≤ x ∧ x ≤ b)) →
  (0 < b - a) ∧ (b - a < 3) := by
sorry

end NUMINAMATH_CALUDE_range_of_b_minus_a_l183_18305


namespace NUMINAMATH_CALUDE_dihedral_angle_range_l183_18357

/-- The dihedral angle between adjacent faces in a regular n-prism -/
def dihedral_angle (n : ℕ) (θ : ℝ) : Prop :=
  n > 2 ∧ ((n - 2 : ℝ) / n) * Real.pi < θ ∧ θ < Real.pi

/-- Theorem stating the range of dihedral angles in a regular n-prism -/
theorem dihedral_angle_range (n : ℕ) :
  ∃ θ : ℝ, dihedral_angle n θ :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_range_l183_18357


namespace NUMINAMATH_CALUDE_intersection_implies_a_equals_two_l183_18398

def A (a : ℝ) : Set ℝ := {2, a^2 - a + 1}
def B (a : ℝ) : Set ℝ := {3, a + 3}

theorem intersection_implies_a_equals_two (a : ℝ) :
  A a ∩ B a = {3} → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_equals_two_l183_18398


namespace NUMINAMATH_CALUDE_work_completion_time_l183_18309

theorem work_completion_time (W D : ℝ) (h1 : W > 0) (h2 : D > 0) : 
  (3 * (W / D) + 3 * (W / D + W / 6) = W) → D = 12 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l183_18309


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_4x4x4_cube_l183_18355

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : ℕ
  painted_size : ℕ
  total_cubes : ℕ
  painted_cubes : ℕ

/-- Theorem: In a 4x4x4 cube with 2x2 squares painted on each face, 56 unit cubes are unpainted -/
theorem unpainted_cubes_in_4x4x4_cube (c : PaintedCube) 
  (h_size : c.size = 4)
  (h_painted : c.painted_size = 2)
  (h_total : c.total_cubes = c.size ^ 3)
  (h_painted_count : c.painted_cubes = 8) :
  c.total_cubes - c.painted_cubes = 56 := by
  sorry

#check unpainted_cubes_in_4x4x4_cube

end NUMINAMATH_CALUDE_unpainted_cubes_in_4x4x4_cube_l183_18355


namespace NUMINAMATH_CALUDE_solution_set_f_positive_solution_set_f_leq_g_l183_18319

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := 3 * x^2 + (4 - m) * x - 6 * m
def g (m : ℝ) (x : ℝ) : ℝ := 2 * x^2 - x - m

-- Part 1: Solution set of f(x) > 0 when m = 1
theorem solution_set_f_positive (x : ℝ) :
  f 1 x > 0 ↔ x < -2 ∨ x > 1 := by sorry

-- Part 2: Solution set of f(x) ≤ g(x) when m > 0
theorem solution_set_f_leq_g (m : ℝ) (x : ℝ) (h : m > 0) :
  f m x ≤ g m x ↔ -5 ≤ x ∧ x ≤ m := by sorry

end NUMINAMATH_CALUDE_solution_set_f_positive_solution_set_f_leq_g_l183_18319


namespace NUMINAMATH_CALUDE_percy_swims_52_hours_l183_18360

/-- Represents Percy's swimming schedule and calculates total swimming hours --/
def percy_swimming_hours : ℕ :=
  let weekday_hours := 2  -- 1 hour before school + 1 hour after school
  let weekdays_per_week := 5
  let weekend_hours := 3
  let weeks := 4
  let weekly_hours := weekday_hours * weekdays_per_week + weekend_hours
  weekly_hours * weeks

/-- Theorem stating that Percy swims 52 hours over 4 weeks --/
theorem percy_swims_52_hours : percy_swimming_hours = 52 := by
  sorry

end NUMINAMATH_CALUDE_percy_swims_52_hours_l183_18360


namespace NUMINAMATH_CALUDE_mike_debt_proof_l183_18372

/-- Calculates the final amount owed after compound interest is applied -/
def final_amount (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proves that the final amount owed is approximately $530.604 -/
theorem mike_debt_proof (ε : ℝ) (h_ε : ε > 0) :
  ∃ (result : ℝ), 
    final_amount 500 0.02 3 = result ∧ 
    abs (result - 530.604) < ε :=
by
  sorry

#eval final_amount 500 0.02 3

end NUMINAMATH_CALUDE_mike_debt_proof_l183_18372


namespace NUMINAMATH_CALUDE_triangle_theorem_l183_18313

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a^2 + t.b^2 - t.c^2 = Real.sqrt 3 * t.a * t.b) 
  (h2 : 0 < t.A ∧ t.A ≤ 2 * Real.pi / 3) :
  t.C = Real.pi / 6 ∧ 
  let m := 2 * (Real.cos (t.A / 2))^2 - Real.sin t.B - 1
  ∀ x, m = x → -1 ≤ x ∧ x < 1/2 := by
sorry

end NUMINAMATH_CALUDE_triangle_theorem_l183_18313


namespace NUMINAMATH_CALUDE_equal_distance_travel_l183_18318

theorem equal_distance_travel (v1 v2 v3 : ℝ) (t : ℝ) (h1 : v1 = 3) (h2 : v2 = 6) (h3 : v3 = 9) (ht : t = 11/60) :
  let d := t / (1/v1 + 1/v2 + 1/v3)
  3 * d = 0.9 := by sorry

end NUMINAMATH_CALUDE_equal_distance_travel_l183_18318


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l183_18385

theorem arithmetic_sequence_terms (a₁ a₂ aₙ : ℕ) (h1 : a₁ = 6) (h2 : a₂ = 9) (h3 : aₙ = 300) :
  ∃ n : ℕ, n = 99 ∧ aₙ = a₁ + (n - 1) * (a₂ - a₁) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l183_18385


namespace NUMINAMATH_CALUDE_mary_eggs_count_l183_18327

/-- Given that Mary starts with 27 eggs and finds 4 more eggs, prove that she ends up with 31 eggs in total. -/
theorem mary_eggs_count (initial_eggs found_eggs : ℕ) : 
  initial_eggs = 27 → found_eggs = 4 → initial_eggs + found_eggs = 31 := by
  sorry

end NUMINAMATH_CALUDE_mary_eggs_count_l183_18327


namespace NUMINAMATH_CALUDE_total_sheets_l183_18362

def sheets_in_desk : ℕ := 50
def sheets_in_backpack : ℕ := 41

theorem total_sheets : sheets_in_desk + sheets_in_backpack = 91 := by
  sorry

end NUMINAMATH_CALUDE_total_sheets_l183_18362


namespace NUMINAMATH_CALUDE_inequality_proof_l183_18328

theorem inequality_proof (x : ℝ) (h : 1 ≤ x ∧ x ≤ 5) : 2*x + 1/x + 1/(x+1) < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l183_18328


namespace NUMINAMATH_CALUDE_find_original_number_l183_18330

theorem find_original_number : 
  ∃ x : ℝ, 3 * (2 * x + 5) = 135 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_find_original_number_l183_18330


namespace NUMINAMATH_CALUDE_exist_three_digits_for_infinite_square_representations_l183_18343

/-- A type representing a digit (0-9) -/
def Digit := Fin 10

/-- A function that checks if a digit is nonzero -/
def isNonzeroDigit (d : Digit) : Prop := d.val ≠ 0

/-- A function that checks if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A function that represents a natural number as a sequence of digits -/
def toDigitSequence (n : ℕ) : List Digit := sorry

/-- A function that checks if a list of digits contains only the given three digits -/
def containsOnlyGivenDigits (seq : List Digit) (d1 d2 d3 : Digit) : Prop :=
  ∀ d ∈ seq, d = d1 ∨ d = d2 ∨ d = d3

/-- The main theorem -/
theorem exist_three_digits_for_infinite_square_representations :
  ∃ (d1 d2 d3 : Digit),
    isNonzeroDigit d1 ∧ isNonzeroDigit d2 ∧ isNonzeroDigit d3 ∧
    ∀ n : ℕ, ∃ m : ℕ, 
      isPerfectSquare m ∧ 
      containsOnlyGivenDigits (toDigitSequence m) d1 d2 d3 := by
  sorry

end NUMINAMATH_CALUDE_exist_three_digits_for_infinite_square_representations_l183_18343


namespace NUMINAMATH_CALUDE_racket_carton_problem_l183_18326

/-- Given two types of tennis racket cartons, one holding 2 rackets and the other
    holding an unknown number x, prove that x = 1 when 38 cartons of the first type
    and 24 cartons of the second type are used to pack a total of 100 rackets. -/
theorem racket_carton_problem (x : ℕ) : 
  (38 * 2 + 24 * x = 100) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_racket_carton_problem_l183_18326


namespace NUMINAMATH_CALUDE_union_P_Q_l183_18371

def P : Set ℝ := { x | -1 < x ∧ x < 1 }
def Q : Set ℝ := { x | x^2 - 2*x < 0 }

theorem union_P_Q : P ∪ Q = { x | -1 < x ∧ x < 2 } := by sorry

end NUMINAMATH_CALUDE_union_P_Q_l183_18371


namespace NUMINAMATH_CALUDE_three_fish_added_l183_18353

/-- The number of fish added to a barrel -/
def fish_added (initial_a initial_b final_total : ℕ) : ℕ :=
  final_total - (initial_a + initial_b)

/-- Theorem: Given the initial numbers of fish and the final total, prove that 3 fish were added -/
theorem three_fish_added : fish_added 4 3 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_fish_added_l183_18353


namespace NUMINAMATH_CALUDE_rhombus_area_l183_18366

/-- The area of a rhombus with side length 4 cm and an interior angle of 45 degrees is 8√2 square centimeters. -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = π / 4) : 
  s * s * Real.sin θ = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l183_18366


namespace NUMINAMATH_CALUDE_mika_stickers_left_l183_18342

/-- The number of stickers Mika has left after various changes -/
def stickers_left (initial bought birthday given_away used : ℕ) : ℕ :=
  initial + bought + birthday - given_away - used

/-- Theorem stating that Mika has 2 stickers left -/
theorem mika_stickers_left :
  stickers_left 20 26 20 6 58 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mika_stickers_left_l183_18342


namespace NUMINAMATH_CALUDE_power_product_rule_l183_18386

theorem power_product_rule (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_rule_l183_18386


namespace NUMINAMATH_CALUDE_candy_distribution_l183_18335

theorem candy_distribution (total_candies : ℕ) (candies_per_student : ℕ) (leftover_candies : ℕ) :
  total_candies = 67 →
  candies_per_student = 4 →
  leftover_candies = 3 →
  (total_candies - leftover_candies) / candies_per_student = 16 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l183_18335


namespace NUMINAMATH_CALUDE_polygon_sides_l183_18390

/-- A polygon with side length 7 and perimeter 42 has 6 sides -/
theorem polygon_sides (side_length : ℕ) (perimeter : ℕ) (h1 : side_length = 7) (h2 : perimeter = 42) :
  perimeter / side_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l183_18390


namespace NUMINAMATH_CALUDE_complex_power_215_36_l183_18304

theorem complex_power_215_36 :
  (Complex.exp (215 * π / 180 * Complex.I)) ^ 36 = 1/2 - Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_215_36_l183_18304


namespace NUMINAMATH_CALUDE_prob_not_late_prob_late_and_miss_bus_l183_18395

-- Define the probability of Sam being late
def prob_late : ℚ := 5/9

-- Define the probability of Sam missing the bus if late
def prob_miss_bus_if_late : ℚ := 1/3

-- Theorem 1: Probability that Sam is not late
theorem prob_not_late : 1 - prob_late = 4/9 := by sorry

-- Theorem 2: Probability that Sam is late and misses the bus
theorem prob_late_and_miss_bus : prob_late * prob_miss_bus_if_late = 5/27 := by sorry

end NUMINAMATH_CALUDE_prob_not_late_prob_late_and_miss_bus_l183_18395


namespace NUMINAMATH_CALUDE_joels_board_games_l183_18399

theorem joels_board_games (stuffed_animals action_figures puzzles total_toys joels_toys : ℕ)
  (h1 : stuffed_animals = 18)
  (h2 : action_figures = 42)
  (h3 : puzzles = 13)
  (h4 : total_toys = 108)
  (h5 : joels_toys = 22) :
  ∃ (board_games sisters_toys : ℕ),
    sisters_toys * 3 = joels_toys ∧
    stuffed_animals + action_figures + board_games + puzzles + sisters_toys * 3 = total_toys ∧
    board_games = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_joels_board_games_l183_18399


namespace NUMINAMATH_CALUDE_square_width_proof_l183_18382

theorem square_width_proof (rectangle_length : ℝ) (rectangle_width : ℝ) (area_difference : ℝ) :
  rectangle_length = 3 →
  rectangle_width = 6 →
  area_difference = 7 →
  ∃ (square_width : ℝ), square_width^2 = rectangle_length * rectangle_width - area_difference :=
by
  sorry

end NUMINAMATH_CALUDE_square_width_proof_l183_18382


namespace NUMINAMATH_CALUDE_multiply_24_to_get_2376_l183_18383

theorem multiply_24_to_get_2376 (x : ℚ) : 24 * x = 2376 → x = 99 := by
  sorry

end NUMINAMATH_CALUDE_multiply_24_to_get_2376_l183_18383


namespace NUMINAMATH_CALUDE_tangent_line_equation_l183_18379

/-- The function f(x) = x^2 -/
def f (x : ℝ) : ℝ := x^2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 2 * x

/-- The point through which the line passes -/
def point : ℝ × ℝ := (1, 1)

/-- The equation of the line: 2x - y - 1 = 0 -/
def line_equation (x y : ℝ) : Prop := 2 * x - y - 1 = 0

theorem tangent_line_equation :
  (∀ x y, line_equation x y ↔ 
    (y - point.2 = f' point.1 * (x - point.1) ∧
     f point.1 = point.2)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l183_18379


namespace NUMINAMATH_CALUDE_probability_not_greater_than_2_78_l183_18377

def digits : Finset ℕ := {7, 1, 8}

def valid_combinations : Finset (ℕ × ℕ) :=
  {(1, 7), (1, 8), (7, 1), (7, 8)}

theorem probability_not_greater_than_2_78 :
  (Finset.card valid_combinations) / (Finset.card (digits.product digits)) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_greater_than_2_78_l183_18377


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l183_18345

theorem right_triangle_leg_length 
  (A B C : ℝ × ℝ) 
  (is_right_triangle : (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0) 
  (leg_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 1) 
  (hypotenuse_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = Real.sqrt 5) :
  Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l183_18345


namespace NUMINAMATH_CALUDE_bobs_family_adults_l183_18329

theorem bobs_family_adults (total_apples : ℕ) (num_children : ℕ) (apples_per_child : ℕ) (apples_per_adult : ℕ) 
  (h1 : total_apples = 450)
  (h2 : num_children = 33)
  (h3 : apples_per_child = 10)
  (h4 : apples_per_adult = 3) :
  (total_apples - num_children * apples_per_child) / apples_per_adult = 40 := by
  sorry

end NUMINAMATH_CALUDE_bobs_family_adults_l183_18329


namespace NUMINAMATH_CALUDE_dragon_rope_problem_l183_18351

-- Define the constants
def rope_length : ℝ := 25
def castle_radius : ℝ := 5
def dragon_height : ℝ := 3
def rope_end_distance : ℝ := 3

-- Define the variables
variable (p q r : ℕ)

-- Define the conditions
axiom p_positive : p > 0
axiom q_positive : q > 0
axiom r_positive : r > 0
axiom r_prime : Nat.Prime r

-- Define the relationship between p, q, r and the rope length touching the castle
axiom rope_touching_castle : (p - Real.sqrt q) / r = (75 - Real.sqrt 450) / 3

-- Theorem to prove
theorem dragon_rope_problem : p + q + r = 528 := by sorry

end NUMINAMATH_CALUDE_dragon_rope_problem_l183_18351


namespace NUMINAMATH_CALUDE_alphabet_sum_theorem_l183_18369

/-- Represents a letter in the English alphabet -/
def Letter := Fin 26

/-- Represents a sequence of 26 letters -/
def Sequence := Fin 26 → Letter

/-- The sum operation for letters -/
def letter_sum (a b : Letter) : Letter :=
  ⟨(a.val + b.val) % 26, by sorry⟩

/-- The sum operation for sequences -/
def sequence_sum (s1 s2 : Sequence) : Sequence :=
  λ i => letter_sum (s1 i) (s2 i)

/-- The standard alphabet sequence -/
def alphabet_sequence : Sequence :=
  λ i => i

/-- A permutation of the alphabet -/
def is_permutation (s : Sequence) : Prop :=
  Function.Injective s

theorem alphabet_sum_theorem (s : Sequence) (h : is_permutation s) :
  ∃ i j : Fin 26, i ≠ j ∧ sequence_sum s alphabet_sequence i = sequence_sum s alphabet_sequence j :=
sorry

end NUMINAMATH_CALUDE_alphabet_sum_theorem_l183_18369


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l183_18378

/-- Given the conditions of a student's exam performance, 
    prove that the maximum marks are 500. -/
theorem exam_maximum_marks :
  let pass_percentage : ℚ := 33 / 100
  let student_marks : ℕ := 125
  let fail_margin : ℕ := 40
  ∃ (max_marks : ℕ), 
    (pass_percentage * max_marks : ℚ) = (student_marks + fail_margin : ℕ) ∧ 
    max_marks = 500 := by
  sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l183_18378


namespace NUMINAMATH_CALUDE_quadratic_root_form_n_l183_18384

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Represents the form (m ± √n) / p for roots of a quadratic equation -/
structure RootForm where
  m : ℤ
  n : ℕ
  p : ℤ

/-- Check if the given RootForm satisfies the conditions for the quadratic equation -/
def isValidRootForm (eq : QuadraticEquation) (rf : RootForm) : Prop :=
  ∃ (x : ℚ), (eq.a * x^2 + eq.b * x + eq.c = 0) ∧
              (x = (rf.m + Real.sqrt rf.n) / rf.p ∨ x = (rf.m - Real.sqrt rf.n) / rf.p) ∧
              Nat.gcd (Nat.gcd rf.m.natAbs rf.n) rf.p.natAbs = 1

theorem quadratic_root_form_n (eq : QuadraticEquation) (rf : RootForm) :
  eq = QuadraticEquation.mk 3 (-7) 2 →
  isValidRootForm eq rf →
  rf.n = 25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_form_n_l183_18384


namespace NUMINAMATH_CALUDE_three_Z_seven_l183_18322

-- Define the operation Z
def Z (a b : ℝ) : ℝ := b + 5 * a - 2 * a^2

-- Theorem to prove
theorem three_Z_seven : Z 3 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_three_Z_seven_l183_18322


namespace NUMINAMATH_CALUDE_solution_equation_one_solution_equation_two_l183_18334

-- First equation
theorem solution_equation_one : 
  ∃ x : ℝ, (2 - x) / (x - 3) = 3 / (3 - x) ↔ x = 5 := by sorry

-- Second equation
theorem solution_equation_two : 
  ∃ x : ℝ, 4 / (x^2 - 1) + 1 = (x - 1) / (x + 1) ↔ x = -1 := by sorry

end NUMINAMATH_CALUDE_solution_equation_one_solution_equation_two_l183_18334


namespace NUMINAMATH_CALUDE_present_expenditure_l183_18375

theorem present_expenditure (P : ℝ) : 
  P * (1 + 0.1)^2 = 24200.000000000004 → P = 20000 := by
  sorry

end NUMINAMATH_CALUDE_present_expenditure_l183_18375


namespace NUMINAMATH_CALUDE_probability_theorem_l183_18302

def total_containers : ℕ := 14
def dry_soil_containers : ℕ := 6
def selected_containers : ℕ := 5
def desired_dry_containers : ℕ := 3

def probability_dry_soil : ℚ :=
  (Nat.choose dry_soil_containers desired_dry_containers *
   Nat.choose (total_containers - dry_soil_containers) (selected_containers - desired_dry_containers)) /
  Nat.choose total_containers selected_containers

theorem probability_theorem :
  probability_dry_soil = 560 / 2002 :=
sorry

end NUMINAMATH_CALUDE_probability_theorem_l183_18302


namespace NUMINAMATH_CALUDE_manolo_total_masks_l183_18370

/-- Represents the number of face-masks Manolo can make in a given time period -/
def masks_made (rate : ℕ) (duration : ℕ) : ℕ :=
  (duration * 60) / rate

/-- Represents Manolo's six-hour shift face-mask production -/
def manolo_shift_production : ℕ :=
  masks_made 4 1 + masks_made 6 2 + masks_made 8 2

theorem manolo_total_masks :
  manolo_shift_production = 50 := by
  sorry

end NUMINAMATH_CALUDE_manolo_total_masks_l183_18370


namespace NUMINAMATH_CALUDE_fraction_of_foreign_male_students_l183_18341

theorem fraction_of_foreign_male_students 
  (total_students : ℕ) 
  (female_fraction : ℚ) 
  (non_foreign_male_students : ℕ) 
  (h1 : total_students = 300)
  (h2 : female_fraction = 2/3)
  (h3 : non_foreign_male_students = 90) :
  (total_students : ℚ) * (1 - female_fraction) - non_foreign_male_students = 
  (1/10) * (total_students * (1 - female_fraction)) := by
sorry

end NUMINAMATH_CALUDE_fraction_of_foreign_male_students_l183_18341


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l183_18354

theorem arithmetic_mean_problem (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 17 →
  a = 12 →
  b = 20 →
  c = d →
  c * d = 324 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l183_18354


namespace NUMINAMATH_CALUDE_composite_probability_l183_18356

/-- The number of sides on a standard die -/
def standard_die_sides : ℕ := 6

/-- The number of sides on the special die -/
def special_die_sides : ℕ := 10

/-- The number of standard dice -/
def num_standard_dice : ℕ := 5

/-- The total number of dice -/
def total_dice : ℕ := num_standard_dice + 1

/-- The total number of possible outcomes -/
def total_outcomes : ℕ := standard_die_sides ^ num_standard_dice * special_die_sides

/-- The number of outcomes where the product is not composite -/
def non_composite_outcomes : ℕ := 25

/-- The probability of rolling a composite product -/
def prob_composite : ℚ := 1 - (non_composite_outcomes : ℚ) / total_outcomes

theorem composite_probability : prob_composite = 77735 / 77760 := by
  sorry

end NUMINAMATH_CALUDE_composite_probability_l183_18356


namespace NUMINAMATH_CALUDE_balls_per_package_l183_18364

theorem balls_per_package (total_packages : Nat) (total_balls : Nat) 
  (h1 : total_packages = 21) 
  (h2 : total_balls = 399) : 
  (total_balls / total_packages : Nat) = 19 := by
  sorry

end NUMINAMATH_CALUDE_balls_per_package_l183_18364


namespace NUMINAMATH_CALUDE_contest_finish_orders_l183_18337

def number_of_participants : ℕ := 3

theorem contest_finish_orders :
  (Nat.factorial number_of_participants) = 6 := by
  sorry

end NUMINAMATH_CALUDE_contest_finish_orders_l183_18337


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l183_18393

/-- The number of diagonals in a convex n-gon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  num_diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l183_18393


namespace NUMINAMATH_CALUDE_mary_sheep_problem_l183_18303

theorem mary_sheep_problem (initial_sheep : ℕ) : 
  (initial_sheep : ℚ) * (3/4) * (1/2) = 150 → initial_sheep = 400 := by
  sorry

end NUMINAMATH_CALUDE_mary_sheep_problem_l183_18303


namespace NUMINAMATH_CALUDE_green_pill_cost_proof_l183_18389

/-- The cost of a green pill in dollars -/
def green_pill_cost : ℚ := 43 / 3

/-- The cost of a pink pill in dollars -/
def pink_pill_cost : ℚ := green_pill_cost - 1

/-- The cost of a blue pill in dollars -/
def blue_pill_cost : ℚ := pink_pill_cost - 2

/-- The number of days in the treatment -/
def treatment_days : ℕ := 21

/-- The total cost of the treatment in dollars -/
def total_cost : ℚ := 819

theorem green_pill_cost_proof :
  (green_pill_cost + pink_pill_cost + blue_pill_cost) * treatment_days = total_cost ∧
  green_pill_cost = 43 / 3 := by
  sorry

#eval green_pill_cost -- To check the value

end NUMINAMATH_CALUDE_green_pill_cost_proof_l183_18389


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_l183_18317

theorem pure_imaginary_solutions (x : ℂ) : 
  (x^5 - 4*x^4 + 6*x^3 - 50*x^2 - 100*x - 120 = 0 ∧ ∃ k : ℝ, x = k*I) ↔ 
  (x = I*Real.sqrt 14 ∨ x = -I*Real.sqrt 14) := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_l183_18317


namespace NUMINAMATH_CALUDE_consecutive_numbers_guarantee_l183_18363

theorem consecutive_numbers_guarantee (n : ℕ) (h : n = 150) :
  ∃ m : ℕ, m = 101 ∧
  (∀ S : Finset ℕ, S.card = m → S ⊆ Finset.range n →
    ∃ a b c : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ b = a + 1 ∧ c = b + 1) ∧
  (∀ k : ℕ, k < m →
    ∃ T : Finset ℕ, T.card = k ∧ T ⊆ Finset.range n ∧
      ∀ a b c : ℕ, (a ∈ T ∧ b ∈ T ∧ c ∈ T) → (b ≠ a + 1 ∨ c ≠ b + 1)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_guarantee_l183_18363


namespace NUMINAMATH_CALUDE_triangle_properties_l183_18336

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The main theorem -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a * Real.sin (2 * t.B) = Real.sqrt 3 * t.b * Real.sin t.A)
  (h2 : Real.cos t.A = 1/3) :
  t.B = π/6 ∧ Real.sin t.C = (2 * Real.sqrt 6 + 1) / 6 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_properties_l183_18336


namespace NUMINAMATH_CALUDE_circular_fields_area_comparison_l183_18320

theorem circular_fields_area_comparison :
  ∀ (r1 r2 : ℝ),
  r1 > 0 → r2 > 0 →
  r2 / r1 = 10 / 4 →
  (π * r2^2 - π * r1^2) / (π * r1^2) * 100 = 525 :=
by
  sorry

end NUMINAMATH_CALUDE_circular_fields_area_comparison_l183_18320


namespace NUMINAMATH_CALUDE_sine_cosine_shift_l183_18381

open Real

theorem sine_cosine_shift (ω : ℝ) :
  (∀ x, sin (ω * (x + π / 3)) = cos (ω * x)) → ω = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_shift_l183_18381


namespace NUMINAMATH_CALUDE_function_inequality_solution_set_l183_18350

/-- Given a function f(x) = (ax-1)/(x+b) where the solution set of f(x) > 0 is (-1, 3),
    prove that the solution set of f(-2x) < 0 is (-∞, -3/2) ∪ (1/2, +∞) -/
theorem function_inequality_solution_set 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h₁ : ∀ x, f x = (a * x - 1) / (x + b))
  (h₂ : Set.Ioo (-1 : ℝ) 3 = {x | f x > 0}) :
  {x : ℝ | f (-2 * x) < 0} = Set.Iic (-3/2) ∪ Set.Ioi (1/2) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_solution_set_l183_18350


namespace NUMINAMATH_CALUDE_f_unique_non_monotonic_range_l183_18321

/-- A quadratic function f(x) with specific properties -/
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

/-- The minimum value of f(x) is 1 -/
axiom min_value : ∃ (x : ℝ), f x = 1 ∧ ∀ (y : ℝ), f y ≥ f x

/-- f(0) = f(2) = 3 -/
axiom f_values : f 0 = 3 ∧ f 2 = 3

/-- Theorem: f(x) is the unique quadratic function satisfying the given conditions -/
theorem f_unique : ∀ (g : ℝ → ℝ), (∃ (a b c : ℝ), ∀ (x : ℝ), g x = a * x^2 + b * x + c) →
  (∃ (x : ℝ), g x = 1 ∧ ∀ (y : ℝ), g y ≥ g x) →
  (g 0 = 3 ∧ g 2 = 3) →
  (∀ (x : ℝ), g x = f x) :=
sorry

/-- Theorem: The range of a for which f(x) is not monotonic in [2a, a + 1] is 0 < a < 0.5 -/
theorem non_monotonic_range : ∀ (a : ℝ), 
  (∃ (x y : ℝ), 2 * a ≤ x ∧ x < y ∧ y ≤ a + 1 ∧ f x > f y) ∧
  (∃ (x y : ℝ), 2 * a ≤ x ∧ x < y ∧ y ≤ a + 1 ∧ f x < f y) ↔
  (0 < a ∧ a < 0.5) :=
sorry

end NUMINAMATH_CALUDE_f_unique_non_monotonic_range_l183_18321


namespace NUMINAMATH_CALUDE_frame_width_is_five_l183_18380

/-- Represents a frame with square openings -/
structure SquareFrame where
  numOpenings : ℕ
  openingPerimeter : ℝ
  totalPerimeter : ℝ

/-- Calculates the width of the frame -/
def frameWidth (frame : SquareFrame) : ℝ :=
  sorry

/-- Theorem stating that for a frame with 3 square openings, 
    an opening perimeter of 60 cm, and a total perimeter of 180 cm, 
    the frame width is 5 cm -/
theorem frame_width_is_five :
  let frame : SquareFrame := {
    numOpenings := 3,
    openingPerimeter := 60,
    totalPerimeter := 180
  }
  frameWidth frame = 5 := by sorry

end NUMINAMATH_CALUDE_frame_width_is_five_l183_18380


namespace NUMINAMATH_CALUDE_right_triangle_point_condition_l183_18306

theorem right_triangle_point_condition (a b c x : ℝ) :
  a > 0 → b > 0 → c > 0 →
  c^2 = a^2 + b^2 →
  0 ≤ x → x ≤ b →
  let s := x^2 + (b - x)^2 + (a * x / b)^2
  s = 2 * (b - x)^2 ↔ x = b^2 / Real.sqrt (a^2 + 2 * b^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_point_condition_l183_18306


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l183_18396

theorem greatest_integer_difference (x y : ℝ) (hx : 4 < x ∧ x < 8) (hy : 8 < y ∧ y < 12) :
  (⌊y⌋ - ⌈x⌉ : ℤ) ≤ 2 ∧ ∃ (x' y' : ℝ), 4 < x' ∧ x' < 8 ∧ 8 < y' ∧ y' < 12 ∧ (⌊y'⌋ - ⌈x'⌉ : ℤ) = 2 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l183_18396


namespace NUMINAMATH_CALUDE_power_function_value_l183_18338

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop := ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) (h1 : isPowerFunction f) (h2 : f (1/2) = 8) : f 2 = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_power_function_value_l183_18338
