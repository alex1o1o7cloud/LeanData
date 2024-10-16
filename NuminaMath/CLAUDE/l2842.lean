import Mathlib

namespace NUMINAMATH_CALUDE_max_sets_production_l2842_284216

/-- Represents the number of sets produced given the number of workers assigned to bolts and nuts -/
def sets_produced (bolt_workers : ℕ) (nut_workers : ℕ) : ℕ :=
  min (25 * bolt_workers) ((20 * nut_workers) / 2)

/-- Theorem stating that 40 bolt workers and 100 nut workers maximize set production -/
theorem max_sets_production :
  ∀ (b n : ℕ),
    b + n = 140 →
    sets_produced b n ≤ sets_produced 40 100 :=
by sorry

end NUMINAMATH_CALUDE_max_sets_production_l2842_284216


namespace NUMINAMATH_CALUDE_sin_half_and_third_max_solutions_l2842_284213

open Real

theorem sin_half_and_third_max_solutions (α : ℝ) : 
  (∃ (s : Finset ℝ), (∀ x ∈ s, ∃ k : ℤ, (x = α/2 + k*π ∨ x = (π - α)/2 + k*π) ∧ sin x = sin α) ∧ s.card ≤ 4) ∧
  (∃ (t : Finset ℝ), (∀ x ∈ t, ∃ k : ℤ, x = α/3 + 2*k*π/3 ∧ sin x = sin α) ∧ t.card ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_sin_half_and_third_max_solutions_l2842_284213


namespace NUMINAMATH_CALUDE_special_triangle_area_special_triangle_area_is_48_l2842_284251

/-- A triangle with two sides of length 10 and 12, and a median to the third side of length 5 -/
structure SpecialTriangle where
  side1 : ℝ
  side2 : ℝ
  median : ℝ
  h_side1 : side1 = 10
  h_side2 : side2 = 12
  h_median : median = 5

/-- The area of a SpecialTriangle is 48 -/
theorem special_triangle_area (t : SpecialTriangle) : ℝ :=
  48

/-- The area of a SpecialTriangle is indeed 48 -/
theorem special_triangle_area_is_48 (t : SpecialTriangle) :
  special_triangle_area t = 48 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_area_special_triangle_area_is_48_l2842_284251


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2842_284243

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ) :
  (∀ x : ℝ, (x - 1)^4 * (x + 2)^8 = a*x^12 + a₁*x^11 + a₂*x^10 + a₃*x^9 + a₄*x^8 + 
    a₅*x^7 + a₆*x^6 + a₇*x^5 + a₈*x^4 + a₉*x^3 + a₁₀*x^2 + a₁₁*x + a₁₂) →
  a + a₂ + a₄ + a₆ + a₈ + a₁₀ + a₁₂ = 8 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2842_284243


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l2842_284211

theorem reciprocal_inequality (a b : ℝ) (ha : a > 0) (hb : b < 0) : 1/a > 1/b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l2842_284211


namespace NUMINAMATH_CALUDE_f_at_one_l2842_284297

/-- Given a polynomial g(x) with three distinct roots, where each root is also a root of f(x),
    prove that f(1) = -217 -/
theorem f_at_one (a b c : ℝ) : 
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    (∀ x : ℝ, x^3 + a*x^2 + x + 20 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃)) →
  (∀ x : ℝ, x^3 + a*x^2 + x + 20 = 0 → x^4 + x^3 + b*x^2 + 50*x + c = 0) →
  (1 : ℝ)^4 + (1 : ℝ)^3 + b*(1 : ℝ)^2 + 50*(1 : ℝ) + c = -217 :=
by sorry

end NUMINAMATH_CALUDE_f_at_one_l2842_284297


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2842_284201

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence where the 3rd term is 5 and the 7th term is 29, the 10th term is 47. -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h_arith : ArithmeticSequence a) 
  (h_3rd : a 3 = 5)
  (h_7th : a 7 = 29) : 
  a 10 = 47 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2842_284201


namespace NUMINAMATH_CALUDE_henry_money_left_l2842_284220

/-- Calculates Henry's final amount of money after various transactions. -/
def henrys_final_amount (initial : ℚ) (from_relatives : ℚ) (from_friend : ℚ) (spent_on_game : ℚ) (donated_to_charity : ℚ) : ℚ :=
  initial + from_relatives + from_friend - spent_on_game - donated_to_charity

/-- Proves that Henry's final amount is $21.75 given the specified transactions. -/
theorem henry_money_left : 
  henrys_final_amount 11.75 18.50 5.25 10.60 3.15 = 21.75 := by
  sorry

end NUMINAMATH_CALUDE_henry_money_left_l2842_284220


namespace NUMINAMATH_CALUDE_sine_cosine_ratio_simplification_l2842_284200

theorem sine_cosine_ratio_simplification :
  (Real.sin (30 * π / 180) + Real.sin (60 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (60 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_ratio_simplification_l2842_284200


namespace NUMINAMATH_CALUDE_least_positive_integer_to_make_multiple_of_five_l2842_284299

theorem least_positive_integer_to_make_multiple_of_five (n : ℕ) : 
  (∃ k : ℕ, (789 + n) = 5 * k) ∧ (∀ m : ℕ, m < n → ¬∃ k : ℕ, (789 + m) = 5 * k) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_to_make_multiple_of_five_l2842_284299


namespace NUMINAMATH_CALUDE_f_negative_solution_l2842_284236

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain : Set ℝ := {x : ℝ | -5 ≤ x ∧ x ≤ 5}

-- State that f is odd
axiom f_odd : ∀ x ∈ domain, f (-x) = -f x

-- Define the set where f(x) < 0 for x ∈ [0, 5]
def negative_range : Set ℝ := {x : ℝ | ((-2 < x ∧ x < 0) ∨ (2 < x ∧ x ≤ 5)) ∧ f x < 0}

-- Theorem to prove
theorem f_negative_solution :
  {x ∈ domain | f x < 0} = {x : ℝ | (-5 < x ∧ x < -2) ∨ (-2 < x ∧ x < 0) ∨ (2 < x ∧ x ≤ 5)} :=
sorry

end NUMINAMATH_CALUDE_f_negative_solution_l2842_284236


namespace NUMINAMATH_CALUDE_steps_in_flight_l2842_284255

/-- The number of steps in each flight of stairs -/
def steps_per_flight : ℕ := sorry

/-- The height of each step in inches -/
def step_height : ℕ := 8

/-- The number of flights Jack goes up -/
def flights_up : ℕ := 3

/-- The number of flights Jack goes down -/
def flights_down : ℕ := 6

/-- The total vertical distance traveled in inches -/
def total_distance : ℕ := 24 * 12

theorem steps_in_flight :
  steps_per_flight * step_height * (flights_down - flights_up) = total_distance ∧
  steps_per_flight = 108 := by sorry

end NUMINAMATH_CALUDE_steps_in_flight_l2842_284255


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_sqrt_three_over_six_l2842_284231

theorem sqrt_difference_equals_sqrt_three_over_six :
  Real.sqrt (4 / 3) - Real.sqrt (3 / 4) = Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_sqrt_three_over_six_l2842_284231


namespace NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l2842_284218

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1 -/
theorem mans_age_to_sons_age_ratio :
  ∀ (man_age son_age : ℕ),
    son_age = 18 →
    man_age = son_age + 20 →
    ∃ (k : ℕ), (man_age + 2) = k * (son_age + 2) →
    (man_age + 2) / (son_age + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l2842_284218


namespace NUMINAMATH_CALUDE_science_club_enrollment_l2842_284277

theorem science_club_enrollment (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ) 
  (h1 : total = 120) 
  (h2 : math = 80) 
  (h3 : physics = 50) 
  (h4 : both = 15) : 
  total - (math + physics - both) = 5 := by
  sorry

end NUMINAMATH_CALUDE_science_club_enrollment_l2842_284277


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_seven_l2842_284282

theorem sum_of_fractions_equals_seven : 
  1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 
  1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - Real.sqrt 12) + 
  1 / (Real.sqrt 12 - 3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_seven_l2842_284282


namespace NUMINAMATH_CALUDE_problem_solution_l2842_284233

theorem problem_solution (x y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 20) : y = 100 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2842_284233


namespace NUMINAMATH_CALUDE_min_value_of_u_l2842_284293

theorem min_value_of_u (a b : ℝ) (h : 3*a^2 - 10*a*b + 8*b^2 + 5*a - 10*b = 0) :
  ∃ (u_min : ℝ), u_min = -34 ∧ ∀ (u : ℝ), u = 9*a^2 + 72*b + 2 → u ≥ u_min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_u_l2842_284293


namespace NUMINAMATH_CALUDE_simplify_inverse_sum_l2842_284222

theorem simplify_inverse_sum (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_simplify_inverse_sum_l2842_284222


namespace NUMINAMATH_CALUDE_fraction_product_equivalence_l2842_284292

theorem fraction_product_equivalence (f g : ℝ → ℝ) :
  ∀ x : ℝ, g x ≠ 0 → (f x / g x > 0 ↔ f x * g x > 0) := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equivalence_l2842_284292


namespace NUMINAMATH_CALUDE_students_going_to_zoo_l2842_284224

theorem students_going_to_zoo (teachers : ℕ) (students_per_group : ℕ) 
  (h1 : teachers = 8) 
  (h2 : students_per_group = 32) : 
  teachers * students_per_group = 256 := by
  sorry

end NUMINAMATH_CALUDE_students_going_to_zoo_l2842_284224


namespace NUMINAMATH_CALUDE_penny_species_count_l2842_284204

theorem penny_species_count :
  let sharks : ℕ := 35
  let eels : ℕ := 15
  let whales : ℕ := 5
  sharks + eels + whales = 55 := by
  sorry

end NUMINAMATH_CALUDE_penny_species_count_l2842_284204


namespace NUMINAMATH_CALUDE_impossibleAllFaceSumsDifferent_l2842_284221

/-- Represents the possible values that can be assigned to an edge -/
inductive EdgeValue
  | Positive
  | Negative

/-- Represents a cube with assigned edge values -/
def Cube := Fin 12 → EdgeValue

/-- Converts an EdgeValue to an integer -/
def edgeValueToInt (v : EdgeValue) : Int :=
  match v with
  | EdgeValue.Positive => 1
  | EdgeValue.Negative => -1

/-- Calculates the sum of a face given the indices of its edges -/
def faceSum (cube : Cube) (e1 e2 e3 e4 : Fin 12) : Int :=
  edgeValueToInt (cube e1) + edgeValueToInt (cube e2) + 
  edgeValueToInt (cube e3) + edgeValueToInt (cube e4)

/-- Represents the faces of a cube by the indices of their edges -/
def cubeFaces : List (Fin 12 × Fin 12 × Fin 12 × Fin 12) := sorry

/-- The main theorem stating that it's impossible to have all face sums different -/
theorem impossibleAllFaceSumsDifferent : 
  ¬ ∃ (cube : Cube), ∀ (face1 face2 : Fin 12 × Fin 12 × Fin 12 × Fin 12), 
    face1 ∈ cubeFaces → face2 ∈ cubeFaces → face1 ≠ face2 → 
    let (e1, e2, e3, e4) := face1
    let (f1, f2, f3, f4) := face2
    faceSum cube e1 e2 e3 e4 ≠ faceSum cube f1 f2 f3 f4 := by
  sorry

#check impossibleAllFaceSumsDifferent

end NUMINAMATH_CALUDE_impossibleAllFaceSumsDifferent_l2842_284221


namespace NUMINAMATH_CALUDE_equation_system_solution_l2842_284232

theorem equation_system_solution (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ + 49*x₇ + 64*x₈ = 2)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ + 64*x₇ + 81*x₈ = 15)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ + 81*x₇ + 100*x₈ = 136)
  (eq4 : 16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ + 100*x₇ + 121*x₈ = 1234) :
  25*x₁ + 36*x₂ + 49*x₃ + 64*x₄ + 81*x₅ + 100*x₆ + 121*x₇ + 144*x₈ = 1242 :=
by sorry


end NUMINAMATH_CALUDE_equation_system_solution_l2842_284232


namespace NUMINAMATH_CALUDE_race_problem_l2842_284257

/-- Race problem statement -/
theorem race_problem (race_length : ℕ) (distance_between : ℕ) (jack_distance : ℕ) :
  race_length = 1000 →
  distance_between = 848 →
  jack_distance = race_length - distance_between →
  jack_distance = 152 :=
by sorry

end NUMINAMATH_CALUDE_race_problem_l2842_284257


namespace NUMINAMATH_CALUDE_sequence_inequality_l2842_284214

theorem sequence_inequality (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0) 
  (hn1 : a (n + 1) = 0) 
  (h_ineq : ∀ k : ℕ, k ≥ 1 → k ≤ n → a (k - 1) - 2 * a k + a (k + 1) ≤ 1) :
  ∀ k : ℕ, k ≤ n + 1 → a k ≤ k * (n + 1 - k) / 2 :=
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l2842_284214


namespace NUMINAMATH_CALUDE_boat_distance_along_stream_l2842_284296

-- Define the speed of the boat in still water
def boat_speed : ℝ := 10

-- Define the distance traveled against the stream in one hour
def distance_against_stream : ℝ := 5

-- Define the time of travel
def travel_time : ℝ := 1

-- Theorem statement
theorem boat_distance_along_stream :
  let stream_speed := boat_speed - distance_against_stream / travel_time
  (boat_speed + stream_speed) * travel_time = 15 := by sorry

end NUMINAMATH_CALUDE_boat_distance_along_stream_l2842_284296


namespace NUMINAMATH_CALUDE_julio_lime_cost_l2842_284283

/-- Represents the cost of limes for Julio's mocktails over 30 days -/
def lime_cost (mocktails_per_day : ℕ) (tbsp_per_mocktail : ℕ) (tbsp_per_lime : ℕ) (limes_per_dollar : ℕ) (days : ℕ) : ℚ :=
  let limes_needed := (mocktails_per_day * tbsp_per_mocktail * days) / tbsp_per_lime
  let lime_sets := (limes_needed + limes_per_dollar - 1) / limes_per_dollar
  lime_sets

theorem julio_lime_cost : 
  lime_cost 1 1 2 3 30 = 5 := by
  sorry

end NUMINAMATH_CALUDE_julio_lime_cost_l2842_284283


namespace NUMINAMATH_CALUDE_solution_interval_l2842_284210

theorem solution_interval (f : ℝ → ℝ) (k : ℝ) : 
  (∃ x, f x = 0 ∧ k < x ∧ x < k + 1/2) →
  (∃ n : ℤ, k = n * 1/2) →
  (∀ x, f x = x^3 - 4 + x) →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_interval_l2842_284210


namespace NUMINAMATH_CALUDE_jake_earnings_l2842_284269

def calculate_earnings (viper_count cobra_count python_count : ℕ)
                       (viper_eggs cobra_eggs python_eggs : ℕ)
                       (viper_price cobra_price python_price : ℚ)
                       (viper_discount cobra_discount : ℚ) : ℚ :=
  let viper_babies := viper_count * viper_eggs
  let cobra_babies := cobra_count * cobra_eggs
  let python_babies := python_count * python_eggs
  let viper_earnings := viper_babies * (viper_price * (1 - viper_discount))
  let cobra_earnings := cobra_babies * (cobra_price * (1 - cobra_discount))
  let python_earnings := python_babies * python_price
  viper_earnings + cobra_earnings + python_earnings

theorem jake_earnings :
  calculate_earnings 2 3 1 3 2 4 300 250 450 (1/10) (1/20) = 4845 := by
  sorry

end NUMINAMATH_CALUDE_jake_earnings_l2842_284269


namespace NUMINAMATH_CALUDE_power_function_theorem_l2842_284266

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℝ, ∀ x : ℝ, f x = x ^ n

-- Define the theorem
theorem power_function_theorem (f : ℝ → ℝ) (h : isPowerFunction f) :
  f 2 = 1/4 → f (1/2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_function_theorem_l2842_284266


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_46_l2842_284281

theorem consecutive_integers_sum_46 :
  ∃ (x : ℕ), x > 0 ∧ x + (x + 1) + (x + 2) + (x + 3) = 46 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_46_l2842_284281


namespace NUMINAMATH_CALUDE_highest_power_of_three_dividing_M_l2842_284264

def M : ℕ := sorry

theorem highest_power_of_three_dividing_M : 
  ∃ (k : ℕ), (3^1 ∣ M) ∧ ¬(3^(k+2) ∣ M) :=
sorry

end NUMINAMATH_CALUDE_highest_power_of_three_dividing_M_l2842_284264


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l2842_284230

-- Define a function to convert a number from any base to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

-- Define the given numbers in their respective bases
def num1 : List Nat := [3, 2, 4]
def base1 : Nat := 9

def num2 : List Nat := [2, 1, 5]
def base2 : Nat := 6

-- State the theorem
theorem base_conversion_subtraction :
  (to_base_10 num1 base1) - (to_base_10 num2 base2) = 182 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l2842_284230


namespace NUMINAMATH_CALUDE_work_payment_proof_l2842_284272

/-- Calculates the total payment for a bricklayer and an electrician's work -/
def total_payment (total_hours : ℝ) (bricklayer_hours : ℝ) (bricklayer_rate : ℝ) (electrician_rate : ℝ) : ℝ :=
  let electrician_hours := total_hours - bricklayer_hours
  bricklayer_hours * bricklayer_rate + electrician_hours * electrician_rate

/-- Proves that the total payment for the given work scenario is $1170 -/
theorem work_payment_proof :
  total_payment 90 67.5 12 16 = 1170 := by
  sorry

end NUMINAMATH_CALUDE_work_payment_proof_l2842_284272


namespace NUMINAMATH_CALUDE_fourth_place_points_value_l2842_284238

def first_place_points : ℕ := 11
def second_place_points : ℕ := 7
def third_place_points : ℕ := 5
def total_participations : ℕ := 7
def total_points_product : ℕ := 38500

def is_valid_fourth_place_points (fourth_place_points : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    a + b + c + d = total_participations ∧
    first_place_points ^ a * second_place_points ^ b * third_place_points ^ c * fourth_place_points ^ d = total_points_product

theorem fourth_place_points_value :
  ∃! (x : ℕ), is_valid_fourth_place_points x ∧ x = 4 := by sorry

end NUMINAMATH_CALUDE_fourth_place_points_value_l2842_284238


namespace NUMINAMATH_CALUDE_cost_of_20_pencils_12_notebooks_l2842_284212

/-- The cost of a pencil in dollars -/
def pencil_cost : ℚ := sorry

/-- The cost of a notebook in dollars -/
def notebook_cost : ℚ := sorry

/-- The first condition: 8 pencils and 10 notebooks cost $5.20 -/
axiom condition1 : 8 * pencil_cost + 10 * notebook_cost = 5.20

/-- The second condition: 6 pencils and 4 notebooks cost $2.24 -/
axiom condition2 : 6 * pencil_cost + 4 * notebook_cost = 2.24

/-- The theorem to prove -/
theorem cost_of_20_pencils_12_notebooks : 
  20 * pencil_cost + 12 * notebook_cost = 6.84 := by sorry

end NUMINAMATH_CALUDE_cost_of_20_pencils_12_notebooks_l2842_284212


namespace NUMINAMATH_CALUDE_power_multiplication_simplification_l2842_284285

theorem power_multiplication_simplification (x : ℝ) : (x^5 * x^3) * x^2 = x^10 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_simplification_l2842_284285


namespace NUMINAMATH_CALUDE_simplify_expression_l2842_284244

theorem simplify_expression : 
  (5^5 + 5^3 + 5) / (5^4 - 2*5^2 + 5) = 651 / 116 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2842_284244


namespace NUMINAMATH_CALUDE_basketball_tournament_l2842_284263

theorem basketball_tournament (n : ℕ) (h_pos : n > 0) : 
  let total_players := 5 * n
  let total_matches := (total_players * (total_players - 1)) / 2
  let women_wins := 3 * total_matches / 7
  let men_wins := 4 * total_matches / 7
  (women_wins + men_wins = total_matches) → 
  (n ≠ 2 ∧ n ≠ 3 ∧ n ≠ 4 ∧ n ≠ 5) :=
by sorry

end NUMINAMATH_CALUDE_basketball_tournament_l2842_284263


namespace NUMINAMATH_CALUDE_existence_of_solution_l2842_284246

theorem existence_of_solution : ∃ (x y : ℕ), x^99 = 2013 * y^100 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_solution_l2842_284246


namespace NUMINAMATH_CALUDE_certain_number_sum_l2842_284289

theorem certain_number_sum (x : ℤ) : x + (-27) = 30 → x = 57 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_sum_l2842_284289


namespace NUMINAMATH_CALUDE_grady_blue_cubes_l2842_284265

theorem grady_blue_cubes (grady_red : ℕ) (gage_initial_red gage_initial_blue : ℕ) (gage_total : ℕ) :
  grady_red = 20 →
  gage_initial_red = 10 →
  gage_initial_blue = 12 →
  gage_total = 35 →
  ∃ (grady_blue : ℕ),
    (2 * grady_red / 5 + grady_blue / 3 + gage_initial_red + gage_initial_blue = gage_total) ∧
    grady_blue = 15 :=
by sorry

end NUMINAMATH_CALUDE_grady_blue_cubes_l2842_284265


namespace NUMINAMATH_CALUDE_walking_distance_l2842_284217

/-- Proves that walking at 3 miles per hour for 1.5 hours results in a distance of 4.5 miles -/
theorem walking_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 3 → time = 1.5 → distance = speed * time → distance = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_walking_distance_l2842_284217


namespace NUMINAMATH_CALUDE_maxim_birth_probability_maxim_birth_probability_proof_l2842_284253

/-- The year Maxim starts first grade -/
def start_year : ℕ := 2014

/-- The month Maxim starts first grade (September = 9) -/
def start_month : ℕ := 9

/-- The day Maxim starts first grade -/
def start_day : ℕ := 1

/-- Maxim's age when he starts first grade -/
def start_age : ℕ := 6

/-- The year we're interested in for Maxim's birth -/
def birth_year_of_interest : ℕ := 2008

/-- Function to determine if a year is a leap year -/
def is_leap_year (year : ℕ) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

/-- Function to get the number of days in a month -/
def days_in_month (year : ℕ) (month : ℕ) : ℕ :=
  if month == 2 then
    if is_leap_year year then 29 else 28
  else if month ∈ [4, 6, 9, 11] then 30
  else 31

/-- The probability that Maxim was born in 2008 -/
theorem maxim_birth_probability : ℚ :=
  244 / 365

/-- Proof of the probability calculation -/
theorem maxim_birth_probability_proof :
  maxim_birth_probability = 244 / 365 := by
  sorry

end NUMINAMATH_CALUDE_maxim_birth_probability_maxim_birth_probability_proof_l2842_284253


namespace NUMINAMATH_CALUDE_widgets_per_shipping_box_l2842_284247

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.width * d.length * d.height

/-- Represents the packing scenario at the Widget Factory -/
structure WidgetPacking where
  cartonDimensions : BoxDimensions
  shippingBoxDimensions : BoxDimensions
  widgetsPerCarton : ℕ

/-- Theorem stating the number of widgets that can be shipped in each shipping box -/
theorem widgets_per_shipping_box (p : WidgetPacking) 
  (h1 : p.cartonDimensions = BoxDimensions.mk 4 4 5)
  (h2 : p.shippingBoxDimensions = BoxDimensions.mk 20 20 20)
  (h3 : p.widgetsPerCarton = 3) : 
  (boxVolume p.shippingBoxDimensions / boxVolume p.cartonDimensions) * p.widgetsPerCarton = 300 := by
  sorry


end NUMINAMATH_CALUDE_widgets_per_shipping_box_l2842_284247


namespace NUMINAMATH_CALUDE_vector_expression_l2842_284209

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (4, 2)

theorem vector_expression : c = 3 • a - b := by sorry

end NUMINAMATH_CALUDE_vector_expression_l2842_284209


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2842_284288

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum :
  let a : ℚ := 1/5
  let r : ℚ := 2/5
  let n : ℕ := 8
  geometric_sum a r n = 390369/1171875 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2842_284288


namespace NUMINAMATH_CALUDE_cube_surface_area_l2842_284240

/-- The surface area of a cube with edge length 11 cm is 726 cm². -/
theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 11) :
  6 * edge_length^2 = 726 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l2842_284240


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2842_284278

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_def : ∀ n : ℕ, S n = (n : ℝ) * a 1 + (n : ℝ) * (n - 1) / 2 * (a 2 - a 1)

theorem arithmetic_sequence_properties (seq : ArithmeticSequence)
    (h : seq.a 1 + 5 * seq.a 3 = seq.S 8) :
    seq.a 10 = 0 ∧ seq.S 7 = seq.S 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2842_284278


namespace NUMINAMATH_CALUDE_zero_unique_additive_multiplicative_property_l2842_284280

theorem zero_unique_additive_multiplicative_property :
  ∀ x : ℤ, (∀ z : ℤ, z + x = z) ∧ (∀ z : ℤ, z * x = 0) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_unique_additive_multiplicative_property_l2842_284280


namespace NUMINAMATH_CALUDE_propositions_truth_l2842_284252

theorem propositions_truth :
  (¬ ∀ x : ℝ, x^4 > x^2) ∧
  (∃ α : ℝ, Real.sin (3 * α) = 3 * Real.sin α) ∧
  (¬ ∃ a : ℝ, ∀ x : ℝ, x^2 + 2*x + a < 0) :=
by sorry

end NUMINAMATH_CALUDE_propositions_truth_l2842_284252


namespace NUMINAMATH_CALUDE_equation_solution_l2842_284225

theorem equation_solution (x y : ℝ) :
  x * y^3 - y^2 = y * x^3 - x^2 → y = -x ∨ y = x ∨ y = 1 / x :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2842_284225


namespace NUMINAMATH_CALUDE_solution_to_equation_l2842_284215

theorem solution_to_equation : ∃ y : ℝ, (7 - y = 10) ∧ (y = -3) := by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2842_284215


namespace NUMINAMATH_CALUDE_only_cylinder_has_quadrilateral_cross_section_l2842_284268

-- Define the types of solids
inductive Solid
  | Cone
  | Cylinder
  | Sphere

-- Define a function that determines if a solid can have a quadrilateral cross-section
def canHaveQuadrilateralCrossSection (s : Solid) : Prop :=
  match s with
  | Solid.Cylinder => True
  | _ => False

-- Theorem statement
theorem only_cylinder_has_quadrilateral_cross_section :
  ∀ s : Solid, canHaveQuadrilateralCrossSection s ↔ s = Solid.Cylinder :=
by
  sorry


end NUMINAMATH_CALUDE_only_cylinder_has_quadrilateral_cross_section_l2842_284268


namespace NUMINAMATH_CALUDE_d_bounds_l2842_284295

-- Define the circle
def Circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

-- Define points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the distance function
def d (P : ℝ × ℝ) : ℝ :=
  let (px, py) := P
  (px - A.1)^2 + (py - A.2)^2 + (px - B.1)^2 + (py - B.2)^2

-- Theorem statement
theorem d_bounds :
  ∀ P : ℝ × ℝ, Circle P.1 P.2 → 
  66 - 16 * Real.sqrt 2 ≤ d P ∧ d P ≤ 66 + 16 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_d_bounds_l2842_284295


namespace NUMINAMATH_CALUDE_line_symmetry_l2842_284276

-- Define the lines
def l1 (x y : ℝ) : Prop := x - 2*y - 3 = 0
def l2 (x y : ℝ) : Prop := 2*x - y - 3 = 0
def symmetry_line (x y : ℝ) : Prop := x + y = 0

-- Define the symmetry relation
def symmetric_points (x1 y1 x2 y2 : ℝ) : Prop :=
  symmetry_line ((x1 + x2)/2) ((y1 + y2)/2) ∧ x1 + y2 = 0 ∧ y1 + x2 = 0

-- Theorem statement
theorem line_symmetry :
  (∀ x y : ℝ, l1 x y ↔ l2 (-y) (-x)) →
  (∀ x1 y1 x2 y2 : ℝ, l1 x1 y1 ∧ l2 x2 y2 → symmetric_points x1 y1 x2 y2) →
  ∀ x y : ℝ, l2 x y ↔ 2*x - y - 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l2842_284276


namespace NUMINAMATH_CALUDE_arccos_less_than_arctan_in_interval_l2842_284202

theorem arccos_less_than_arctan_in_interval :
  ∀ x : ℝ, 0.5 < x ∧ x ≤ 1 → Real.arccos x < Real.arctan x := by
  sorry

end NUMINAMATH_CALUDE_arccos_less_than_arctan_in_interval_l2842_284202


namespace NUMINAMATH_CALUDE_complementary_angle_of_60_degrees_l2842_284203

/-- Given that complementary angles sum to 180°, prove that the complementary angle of 60° is 120°. -/
theorem complementary_angle_of_60_degrees :
  (∀ x y : ℝ, x + y = 180 → (x = 60 → y = 120)) :=
by sorry

end NUMINAMATH_CALUDE_complementary_angle_of_60_degrees_l2842_284203


namespace NUMINAMATH_CALUDE_school_greening_area_equation_l2842_284227

/-- Represents the growth of a greening area over time -/
def greeningAreaGrowth (initialArea finalArea : ℝ) (years : ℕ) (growthRate : ℝ) : Prop :=
  initialArea * (1 + growthRate) ^ years = finalArea

/-- The equation for the school's greening area growth -/
theorem school_greening_area_equation :
  greeningAreaGrowth 1000 1440 2 x ↔ 1000 * (1 + x)^2 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_school_greening_area_equation_l2842_284227


namespace NUMINAMATH_CALUDE_no_1999_primes_in_ap_l2842_284286

theorem no_1999_primes_in_ap (a d : ℕ) (h : a > 0 ∧ d > 0) :
  (∀ k : ℕ, k < 1999 → a + k * d < 12345 ∧ Nat.Prime (a + k * d)) →
  False :=
sorry

end NUMINAMATH_CALUDE_no_1999_primes_in_ap_l2842_284286


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2842_284239

def p (x : ℝ) : ℝ := 4 * x^3 - 8 * x^2 + 8 * x - 16

theorem polynomial_divisibility :
  (∃ q : ℝ → ℝ, ∀ x, p x = (x - 2) * q x) ∧
  (∃ r : ℝ → ℝ, ∀ x, p x = (x^2 + 1) * r x) :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2842_284239


namespace NUMINAMATH_CALUDE_swimmer_distance_l2842_284279

/-- Proves that the distance swam against the current is 6 km given the specified conditions -/
theorem swimmer_distance (swimmer_speed : ℝ) (current_speed : ℝ) (time : ℝ) 
  (h1 : swimmer_speed = 4)
  (h2 : current_speed = 1)
  (h3 : time = 2) :
  (swimmer_speed - current_speed) * time = 6 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_distance_l2842_284279


namespace NUMINAMATH_CALUDE_sum_of_bn_l2842_284294

theorem sum_of_bn (m : ℕ) (a : ℕ → ℝ) (b : ℕ → ℝ) :
  (∀ n ∈ Finset.range (2 * m + 1), (a n) * (a (n + 1)) = b n) →
  (∀ n ∈ Finset.range (2 * m), (a n) + (a (n + 1)) = -4 * n) →
  a 1 = 0 →
  (Finset.range (2 * m)).sum b = (8 * m / 3) * (4 * m^2 + 3 * m - 1) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_bn_l2842_284294


namespace NUMINAMATH_CALUDE_video_game_expenditure_l2842_284274

theorem video_game_expenditure (total : ℚ) (books_frac snacks_frac toys_frac : ℚ) 
  (h_total : total = 60)
  (h_books : books_frac = 1/4)
  (h_snacks : snacks_frac = 1/6)
  (h_toys : toys_frac = 2/5)
  : total - (books_frac * total + snacks_frac * total + toys_frac * total) = 11 := by
  sorry

end NUMINAMATH_CALUDE_video_game_expenditure_l2842_284274


namespace NUMINAMATH_CALUDE_sequence_bound_l2842_284205

theorem sequence_bound (a : ℕ → ℝ) 
  (h_pos : ∀ n, n ≥ 1 → a n > 0)
  (h_ineq : ∀ n, n ≥ 1 → (a (n + 1))^2 + (a n) * (a (n + 2)) ≤ a n + a (n + 2)) :
  a 2023 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_bound_l2842_284205


namespace NUMINAMATH_CALUDE_parabola_symmetric_point_l2842_284248

/-- Given a parabola y = x^2 + 4x - m where (1, 2) is a point on the parabola,
    and a point B symmetric to (1, 2) with respect to the axis of symmetry,
    prove that the coordinates of B are (-5, 2). -/
theorem parabola_symmetric_point (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + 4*x - m
  let A : ℝ × ℝ := (1, 2)
  let axis_of_symmetry : ℝ := -2
  let B : ℝ × ℝ := (-5, 2)
  (f A.1 = A.2) →  -- A is on the parabola
  (A.1 - axis_of_symmetry = axis_of_symmetry - B.1) →  -- A and B are symmetric
  (A.2 = B.2) →  -- y-coordinates of A and B are equal
  B = (-5, 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_symmetric_point_l2842_284248


namespace NUMINAMATH_CALUDE_range_of_a_l2842_284219

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x, x^2 + (a-1)*x + a^2 > 0

-- Define proposition q
def q (a : ℝ) : Prop := ∀ x y, x < y → (2*a^2 - a)^x < (2*a^2 - a)^y

-- Theorem statement
theorem range_of_a :
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) →
  (∀ a : ℝ, (1/3 < a ∧ a ≤ 1) ∨ (-1 ≤ a ∧ a < -1/2) ↔ (p a ∨ q a) ∧ ¬(p a ∧ q a)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2842_284219


namespace NUMINAMATH_CALUDE_largest_four_digit_base4_is_255_l2842_284258

/-- Converts a base-4 digit to its base-10 equivalent -/
def base4DigitToBase10 (digit : Nat) : Nat :=
  if digit < 4 then digit else 0

/-- Calculates the base-10 value of a four-digit base-4 number -/
def fourDigitBase4ToBase10 (d1 d2 d3 d4 : Nat) : Nat :=
  (base4DigitToBase10 d1) * (4^3) +
  (base4DigitToBase10 d2) * (4^2) +
  (base4DigitToBase10 d3) * (4^1) +
  (base4DigitToBase10 d4) * (4^0)

/-- The largest four-digit base-4 number, when converted to base-10, equals 255 -/
theorem largest_four_digit_base4_is_255 :
  fourDigitBase4ToBase10 3 3 3 3 = 255 := by
  sorry

#eval fourDigitBase4ToBase10 3 3 3 3

end NUMINAMATH_CALUDE_largest_four_digit_base4_is_255_l2842_284258


namespace NUMINAMATH_CALUDE_ricks_road_trip_l2842_284275

/-- Rick's road trip problem -/
theorem ricks_road_trip (D : ℝ) : 
  D > 0 ∧ 
  40 = D / 2 → 
  D + 2 * D + 40 + 2 * (D + 2 * D + 40) = 840 := by
  sorry

end NUMINAMATH_CALUDE_ricks_road_trip_l2842_284275


namespace NUMINAMATH_CALUDE_sum_of_digits_up_to_2023_l2842_284256

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits of all numbers from 1 to n -/
def sumOfDigitsUpTo (n : ℕ) : ℕ := 
  (List.range n).map (λ i => sumOfDigits (i + 1)) |>.sum

/-- The sum of digits of all numbers from 1 to 2023 is 27314 -/
theorem sum_of_digits_up_to_2023 : sumOfDigitsUpTo 2023 = 27314 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_up_to_2023_l2842_284256


namespace NUMINAMATH_CALUDE_same_color_probability_l2842_284254

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans a person has -/
def JellyBeans.total (jb : JellyBeans) : ℕ := jb.green + jb.red + jb.yellow

/-- Abe's jelly bean distribution -/
def abe : JellyBeans := { green := 2, red := 2, yellow := 0 }

/-- Bob's jelly bean distribution -/
def bob : JellyBeans := { green := 2, red := 3, yellow := 1 }

/-- Calculates the probability of picking the same color jelly bean -/
def probSameColor (jb1 jb2 : JellyBeans) : ℚ :=
  (jb1.green * jb2.green + jb1.red * jb2.red) / (jb1.total * jb2.total)

theorem same_color_probability :
  probSameColor abe bob = 5/12 := by sorry

end NUMINAMATH_CALUDE_same_color_probability_l2842_284254


namespace NUMINAMATH_CALUDE_unique_number_with_property_l2842_284290

/-- A four-digit natural number -/
def FourDigitNumber (x y z w : ℕ) : ℕ := 1000 * x + 100 * y + 10 * z + w

/-- The property that the sum of the number and its digits equals 2003 -/
def HasProperty (x y z w : ℕ) : Prop :=
  FourDigitNumber x y z w + x + y + z + w = 2003

/-- The theorem stating that 1978 is the only four-digit number satisfying the property -/
theorem unique_number_with_property :
  (∃! n : ℕ, ∃ x y z w : ℕ, 
    x ≠ 0 ∧ 
    n = FourDigitNumber x y z w ∧ 
    HasProperty x y z w) ∧
  (∃ x y z w : ℕ, 
    x ≠ 0 ∧ 
    1978 = FourDigitNumber x y z w ∧ 
    HasProperty x y z w) :=
sorry

end NUMINAMATH_CALUDE_unique_number_with_property_l2842_284290


namespace NUMINAMATH_CALUDE_expand_product_l2842_284226

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2842_284226


namespace NUMINAMATH_CALUDE_total_bones_in_pile_l2842_284223

def number_of_dogs : ℕ := 5

def bones_first_dog : ℕ := 3

def bones_second_dog (first : ℕ) : ℕ := first - 1

def bones_third_dog (second : ℕ) : ℕ := 2 * second

def bones_fourth_dog : ℕ := 1

def bones_fifth_dog (fourth : ℕ) : ℕ := 2 * fourth

theorem total_bones_in_pile :
  bones_first_dog +
  bones_second_dog bones_first_dog +
  bones_third_dog (bones_second_dog bones_first_dog) +
  bones_fourth_dog +
  bones_fifth_dog bones_fourth_dog = 12 :=
by sorry

end NUMINAMATH_CALUDE_total_bones_in_pile_l2842_284223


namespace NUMINAMATH_CALUDE_marble_probability_l2842_284237

theorem marble_probability (total : ℕ) (p_white p_green p_black : ℚ) :
  total = 120 ∧
  p_white = 1/4 ∧
  p_green = 1/6 ∧
  p_black = 1/8 →
  1 - (p_white + p_green + p_black) = 11/24 :=
by sorry

end NUMINAMATH_CALUDE_marble_probability_l2842_284237


namespace NUMINAMATH_CALUDE_math_competition_score_ratio_l2842_284208

theorem math_competition_score_ratio :
  let sammy_score : ℕ := 20
  let gab_score : ℕ := 2 * sammy_score
  let opponent_score : ℕ := 85
  let total_score : ℕ := opponent_score + 55
  let cher_score : ℕ := total_score - (sammy_score + gab_score)
  (cher_score : ℚ) / (gab_score : ℚ) = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_math_competition_score_ratio_l2842_284208


namespace NUMINAMATH_CALUDE_tree_height_difference_l2842_284273

def birch_height : ℚ := 25/2
def maple_height : ℚ := 55/3
def pine_height : ℚ := 63/4

def tallest_height : ℚ := max (max birch_height maple_height) pine_height
def shortest_height : ℚ := min (min birch_height maple_height) pine_height

theorem tree_height_difference :
  tallest_height - shortest_height = 35/6 :=
sorry

end NUMINAMATH_CALUDE_tree_height_difference_l2842_284273


namespace NUMINAMATH_CALUDE_simplify_expression_l2842_284261

theorem simplify_expression (x y : ℝ) (h : y = Real.sqrt (x - 2) + Real.sqrt (2 - x) + 2) :
  |y - Real.sqrt 3| - (x - 2 + Real.sqrt 2)^2 = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2842_284261


namespace NUMINAMATH_CALUDE_contribution_rate_of_random_error_l2842_284234

theorem contribution_rate_of_random_error 
  (sum_squared_residuals : ℝ) 
  (total_sum_squares : ℝ) 
  (h1 : sum_squared_residuals = 325) 
  (h2 : total_sum_squares = 923) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ 
  abs (sum_squared_residuals / total_sum_squares - 0.352) < ε :=
sorry

end NUMINAMATH_CALUDE_contribution_rate_of_random_error_l2842_284234


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l2842_284270

/-- Given three lines in a plane, if they intersect at the same point, 
    we can determine the value of a parameter in one of the lines. -/
theorem intersection_of_three_lines (a : ℝ) : 
  (∃ (x y : ℝ), (a*x + 2*y + 6 = 0) ∧ (x + y - 4 = 0) ∧ (2*x - y + 1 = 0)) →
  a = -12 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l2842_284270


namespace NUMINAMATH_CALUDE_power_of_three_equation_l2842_284271

theorem power_of_three_equation (k : ℤ) : 
  3^1999 - 3^1998 - 3^1997 + 3^1996 = k * 3^1996 → k = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_equation_l2842_284271


namespace NUMINAMATH_CALUDE_total_flooring_cost_l2842_284291

/-- Represents the dimensions of a room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a room given its dimensions -/
def roomArea (d : RoomDimensions) : ℝ := d.length * d.width

/-- Calculates the cost of flooring for a room given its area and slab rate -/
def roomCost (area : ℝ) (slabRate : ℝ) : ℝ := area * slabRate

/-- Theorem: The total cost of flooring for the house is Rs. 81,390 -/
theorem total_flooring_cost : 
  let room1 : RoomDimensions := ⟨5.5, 3.75⟩
  let room2 : RoomDimensions := ⟨6, 4.2⟩
  let room3 : RoomDimensions := ⟨4.8, 3.25⟩
  let slabRate1 : ℝ := 1200
  let slabRate2 : ℝ := 1350
  let slabRate3 : ℝ := 1450
  let totalCost : ℝ := 
    roomCost (roomArea room1) slabRate1 + 
    roomCost (roomArea room2) slabRate2 + 
    roomCost (roomArea room3) slabRate3
  totalCost = 81390 := by
  sorry

end NUMINAMATH_CALUDE_total_flooring_cost_l2842_284291


namespace NUMINAMATH_CALUDE_acid_concentration_solution_l2842_284228

/-- Represents the acid concentration problem with three flasks of acid and one of water -/
def AcidConcentrationProblem (acid1 acid2 acid3 : ℝ) (concentration1 concentration2 : ℝ) : Prop :=
  let water1 := acid1 / concentration1 - acid1
  let water2 := acid2 / concentration2 - acid2
  let total_water := water1 + water2
  let concentration3 := acid3 / (acid3 + total_water)
  (acid1 = 10) ∧ 
  (acid2 = 20) ∧ 
  (acid3 = 30) ∧ 
  (concentration1 = 0.05) ∧ 
  (concentration2 = 70/300) ∧ 
  (concentration3 = 0.105)

/-- Theorem stating the solution to the acid concentration problem -/
theorem acid_concentration_solution : 
  ∃ (acid1 acid2 acid3 concentration1 concentration2 : ℝ),
  AcidConcentrationProblem acid1 acid2 acid3 concentration1 concentration2 :=
by
  sorry

end NUMINAMATH_CALUDE_acid_concentration_solution_l2842_284228


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l2842_284260

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_equation_solution :
  ∀ N : ℕ, N > 0 → (factorial 5 * factorial 9 = 12 * factorial N) → N = 10 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l2842_284260


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l2842_284207

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line
def line (x y k : ℝ) : Prop := y = k*x - 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1)

-- Define the intersection points
def intersection_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line x₁ y₁ k ∧ line x₂ y₂ k ∧
    x₁ > 0 ∧ y₁ > 0 ∧ x₂ > 0 ∧ y₂ > 0 ∧
    x₁ ≠ x₂

-- Define the distance ratio condition
def distance_ratio (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ((x₁ - 0)^2 + (y₁ - 1)^2).sqrt = 3 * ((x₂ - 0)^2 + (y₂ - 1)^2).sqrt

-- The theorem statement
theorem parabola_intersection_theorem (k : ℝ) :
  intersection_points k →
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
    line x₁ y₁ k ∧ line x₂ y₂ k ∧
    distance_ratio x₁ y₁ x₂ y₂) →
  k = (2 * Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l2842_284207


namespace NUMINAMATH_CALUDE_range_x_when_a_is_one_range_a_for_not_p_sufficient_not_necessary_for_not_q_l2842_284287

/-- Proposition p: x^2 - 4ax + 3a^2 < 0 -/
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

/-- Proposition q: x^2 - x - 6 ≤ 0 and x^2 + 2x - 8 > 0 -/
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

theorem range_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x < 3) :=
sorry

theorem range_a_for_not_p_sufficient_not_necessary_for_not_q :
  ∀ a : ℝ, (∀ x : ℝ, (¬p x a → (x^2 - x - 6 > 0 ∨ x^2 + 2*x - 8 ≤ 0)) ∧
    ∃ x : ℝ, (x^2 - x - 6 > 0 ∨ x^2 + 2*x - 8 ≤ 0) ∧ p x a) ↔ (1 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_range_x_when_a_is_one_range_a_for_not_p_sufficient_not_necessary_for_not_q_l2842_284287


namespace NUMINAMATH_CALUDE_dessert_preference_l2842_284206

structure Classroom where
  total : ℕ
  apple : ℕ
  chocolate : ℕ
  pumpkin : ℕ
  none : ℕ

def likes_apple_and_chocolate_not_pumpkin (c : Classroom) : ℕ :=
  c.apple + c.chocolate - (c.total - c.none) - 2

theorem dessert_preference (c : Classroom) 
  (h_total : c.total = 50)
  (h_apple : c.apple = 25)
  (h_chocolate : c.chocolate = 20)
  (h_pumpkin : c.pumpkin = 10)
  (h_none : c.none = 16) :
  likes_apple_and_chocolate_not_pumpkin c = 9 := by
  sorry

end NUMINAMATH_CALUDE_dessert_preference_l2842_284206


namespace NUMINAMATH_CALUDE_smallest_m_inequality_l2842_284242

theorem smallest_m_inequality (a b c : ℤ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  ∃ (m : ℝ), (∀ (x y z : ℤ), x > 0 → y > 0 → z > 0 → x + y + z = 1 → 
    m * (x^3 + y^3 + z^3 : ℝ) ≥ 6 * (x^2 + y^2 + z^2 : ℝ) + 1) ∧ 
  m = 27 ∧
  ∀ (n : ℝ), (∀ (x y z : ℤ), x > 0 → y > 0 → z > 0 → x + y + z = 1 → 
    n * (x^3 + y^3 + z^3 : ℝ) ≥ 6 * (x^2 + y^2 + z^2 : ℝ) + 1) → n ≥ 27 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_inequality_l2842_284242


namespace NUMINAMATH_CALUDE_coffee_cost_calculation_l2842_284229

/-- Calculates the weekly coffee cost for a household -/
def weekly_coffee_cost (people : ℕ) (cups_per_day : ℕ) (oz_per_cup : ℚ) (cost_per_oz : ℚ) : ℚ :=
  people * cups_per_day * oz_per_cup * cost_per_oz * 7

theorem coffee_cost_calculation :
  let people : ℕ := 4
  let cups_per_day : ℕ := 2
  let oz_per_cup : ℚ := 1/2
  let cost_per_oz : ℚ := 5/4
  weekly_coffee_cost people cups_per_day oz_per_cup cost_per_oz = 35 := by
  sorry

#eval weekly_coffee_cost 4 2 (1/2) (5/4)

end NUMINAMATH_CALUDE_coffee_cost_calculation_l2842_284229


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l2842_284241

theorem cubic_sum_theorem (a b : ℝ) (h : a^3 + b^3 + 3*a*b = 1) : 
  a + b = 1 ∨ a + b = -2 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l2842_284241


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_4_power_minus_2_power_29_l2842_284298

theorem greatest_prime_factor_of_4_power_minus_2_power_29 (n : ℕ) : 
  (∃ (p : ℕ), Nat.Prime p ∧ p ∣ (4^n - 2^29) ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ (4^n - 2^29) → q ≤ p) ∧
  (∀ (q : ℕ), Nat.Prime q → q ∣ (4^n - 2^29) → q ≤ 31) ∧
  (31 ∣ (4^n - 2^29)) →
  n = 17 :=
by sorry


end NUMINAMATH_CALUDE_greatest_prime_factor_of_4_power_minus_2_power_29_l2842_284298


namespace NUMINAMATH_CALUDE_pole_length_l2842_284249

theorem pole_length (pole_length : ℝ) (gate_height : ℝ) (gate_width : ℝ) : 
  gate_width = 3 →
  pole_length = gate_height + 1 →
  pole_length^2 = gate_height^2 + gate_width^2 →
  pole_length = 5 := by
sorry

end NUMINAMATH_CALUDE_pole_length_l2842_284249


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l2842_284250

theorem smallest_part_of_proportional_division (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 90 ∧ a = 3 ∧ b = 5 ∧ c = 7 →
  ∃ x : ℚ, x > 0 ∧ total = a * x + b * x + c * x ∧ min (a * x) (min (b * x) (c * x)) = 18 :=
by sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l2842_284250


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2842_284235

theorem max_value_of_expression (w x y z : ℝ) : 
  w ≥ 0 → x ≥ 0 → y ≥ 0 → z ≥ 0 → w + x + y + z = 200 → 
  w * z + x * y + z * x ≤ 7500 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2842_284235


namespace NUMINAMATH_CALUDE_pencil_discount_l2842_284284

theorem pencil_discount (original_cost final_price : ℝ) 
  (h1 : original_cost = 4)
  (h2 : final_price = 3.37) : 
  original_cost - final_price = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_pencil_discount_l2842_284284


namespace NUMINAMATH_CALUDE_perpendicular_diagonals_imply_cyclic_projections_l2842_284262

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the concept of perpendicular diagonals
def has_perpendicular_diagonals (q : Quadrilateral) : Prop :=
  let AC := (q.C.1 - q.A.1, q.C.2 - q.A.2)
  let BD := (q.D.1 - q.B.1, q.D.2 - q.B.2)
  AC.1 * BD.1 + AC.2 * BD.2 = 0

-- Define the projection of a point onto a line segment
def project_point (P : ℝ × ℝ) (A B : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- Define the intersection point of diagonals
def diagonal_intersection (q : Quadrilateral) : ℝ × ℝ :=
  sorry

-- Define a cyclic quadrilateral
def is_cyclic (A B C D : ℝ × ℝ) : Prop :=
  sorry

-- Main theorem
theorem perpendicular_diagonals_imply_cyclic_projections (q : Quadrilateral) :
  has_perpendicular_diagonals q →
  let I := diagonal_intersection q
  let A1 := project_point I q.A q.B
  let B1 := project_point I q.B q.C
  let C1 := project_point I q.C q.D
  let D1 := project_point I q.D q.A
  is_cyclic A1 B1 C1 D1 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_diagonals_imply_cyclic_projections_l2842_284262


namespace NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_l2842_284267

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := -x^2 + 4*x + 5 < 0
def inequality2 (x : ℝ) : Prop := 2*x^2 - 5*x + 2 ≤ 0

-- Define the solution sets
def solution_set1 : Set ℝ := {x | x < -1 ∨ x > 5}
def solution_set2 : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 2}

-- Theorem statements
theorem inequality1_solution : 
  ∀ x : ℝ, inequality1 x ↔ x ∈ solution_set1 := by sorry

theorem inequality2_solution : 
  ∀ x : ℝ, inequality2 x ↔ x ∈ solution_set2 := by sorry

end NUMINAMATH_CALUDE_inequality1_solution_inequality2_solution_l2842_284267


namespace NUMINAMATH_CALUDE_expression_simplification_l2842_284245

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (x + 1) / (x^2 + 2*x + 1) / (1 - 2 / (x + 1)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2842_284245


namespace NUMINAMATH_CALUDE_fraction_problem_l2842_284259

theorem fraction_problem (f n : ℚ) (h1 : f * n - 5 = 5) (h2 : n = 50) : f = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2842_284259
