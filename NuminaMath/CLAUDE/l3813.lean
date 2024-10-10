import Mathlib

namespace choose_books_different_languages_l3813_381352

theorem choose_books_different_languages (chinese english japanese : ℕ) :
  chinese = 5 → english = 4 → japanese = 3 →
  chinese + english + japanese = 12 :=
by sorry

end choose_books_different_languages_l3813_381352


namespace larger_number_proof_l3813_381381

theorem larger_number_proof (a b : ℝ) : 
  a > 0 → b > 0 → a > b → a + b = 9 * (a - b) → a + b = 36 → a = 20 := by
  sorry

end larger_number_proof_l3813_381381


namespace bottles_remaining_l3813_381382

/-- Calculates the number of bottles remaining in storage given the initial quantities and percentages sold. -/
theorem bottles_remaining (small_initial : ℕ) (big_initial : ℕ) (small_percent_sold : ℚ) (big_percent_sold : ℚ) :
  small_initial = 6000 →
  big_initial = 15000 →
  small_percent_sold = 11 / 100 →
  big_percent_sold = 12 / 100 →
  (small_initial - small_initial * small_percent_sold) + (big_initial - big_initial * big_percent_sold) = 18540 := by
sorry

end bottles_remaining_l3813_381382


namespace jack_apples_to_father_l3813_381357

/-- The number of apples Jack bought -/
def total_apples : ℕ := 55

/-- The number of Jack's friends -/
def num_friends : ℕ := 4

/-- The number of apples each person (Jack and his friends) gets -/
def apples_per_person : ℕ := 9

/-- The number of apples Jack wants to give to his father -/
def apples_to_father : ℕ := total_apples - (num_friends + 1) * apples_per_person

theorem jack_apples_to_father :
  apples_to_father = 10 := by sorry

end jack_apples_to_father_l3813_381357


namespace triangle_inradius_l3813_381312

/-- Given a triangle with perimeter 39 cm and area 29.25 cm², its inradius is 1.5 cm -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
  (h1 : p = 39) 
  (h2 : A = 29.25) 
  (h3 : A = r * p / 2) : 
  r = 1.5 := by
sorry

end triangle_inradius_l3813_381312


namespace art_piece_value_increase_l3813_381318

def original_price : ℝ := 4000
def future_price : ℝ := 3 * original_price

theorem art_piece_value_increase : future_price - original_price = 8000 := by
  sorry

end art_piece_value_increase_l3813_381318


namespace complex_modulus_range_l3813_381305

theorem complex_modulus_range (a : ℝ) : 
  (∀ θ : ℝ, Complex.abs ((a + Real.cos θ) + (2 * a - Real.sin θ) * Complex.I) ≤ 2) ↔ 
  a ∈ Set.Icc (-1/2) (1/2) := by
sorry

end complex_modulus_range_l3813_381305


namespace estimate_two_sqrt_five_l3813_381390

theorem estimate_two_sqrt_five : 4 < 2 * Real.sqrt 5 ∧ 2 * Real.sqrt 5 < 5 := by
  sorry

end estimate_two_sqrt_five_l3813_381390


namespace keith_cards_l3813_381313

theorem keith_cards (x : ℕ) : 
  (x + 8) / 2 = 46 → x = 84 := by
  sorry

end keith_cards_l3813_381313


namespace tan_and_cot_inequalities_l3813_381395

open Real

theorem tan_and_cot_inequalities (x₁ x₂ : ℝ) 
  (h1 : 0 < x₁) (h2 : x₁ < π/2) (h3 : 0 < x₂) (h4 : x₂ < π/2) (h5 : x₁ ≠ x₂) :
  (1/2) * (tan x₁ + tan x₂) > tan ((x₁ + x₂)/2) ∧
  (1/2) * (1/tan x₁ + 1/tan x₂) > 1/tan ((x₁ + x₂)/2) := by
  sorry

end tan_and_cot_inequalities_l3813_381395


namespace barium_oxide_required_l3813_381353

/-- Represents a chemical substance with its number of moles -/
structure Substance where
  name : String
  moles : ℚ

/-- Represents a chemical reaction with reactants and products -/
structure Reaction where
  reactants : List Substance
  products : List Substance

def barium_oxide_water_reaction : Reaction :=
  { reactants := [
      { name := "BaO", moles := 1 },
      { name := "H2O", moles := 1 }
    ],
    products := [
      { name := "Ba(OH)2", moles := 1 }
    ]
  }

theorem barium_oxide_required (water_moles : ℚ) (barium_hydroxide_moles : ℚ) :
  water_moles = barium_hydroxide_moles →
  (∃ (bao : Substance),
    bao.name = "BaO" ∧
    bao.moles = water_moles ∧
    bao.moles = barium_hydroxide_moles ∧
    (∃ (h2o : Substance) (baoh2 : Substance),
      h2o.name = "H2O" ∧
      h2o.moles = water_moles ∧
      baoh2.name = "Ba(OH)2" ∧
      baoh2.moles = barium_hydroxide_moles ∧
      barium_oxide_water_reaction.reactants = [bao, h2o] ∧
      barium_oxide_water_reaction.products = [baoh2])) :=
by
  sorry

end barium_oxide_required_l3813_381353


namespace red_candies_count_l3813_381368

theorem red_candies_count (total : ℕ) (blue : ℕ) (h1 : total = 3409) (h2 : blue = 3264) :
  total - blue = 145 := by
  sorry

end red_candies_count_l3813_381368


namespace fine_on_fifth_day_l3813_381331

/-- Calculates the fine for a given day based on the previous day's fine -/
def nextDayFine (prevFine : ℚ) : ℚ :=
  min (prevFine + 0.3) (prevFine * 2)

/-- Calculates the total fine for a given number of days -/
def totalFine : ℕ → ℚ
  | 0 => 0
  | 1 => 0.05
  | n + 1 => nextDayFine (totalFine n)

theorem fine_on_fifth_day :
  totalFine 5 = 0.7 := by
  sorry

end fine_on_fifth_day_l3813_381331


namespace side_to_hotdog_ratio_l3813_381379

def food_weights (chicken hamburger hotdog side : ℝ) : Prop :=
  chicken = 16 ∧
  hamburger = chicken / 2 ∧
  hotdog = hamburger + 2 ∧
  chicken + hamburger + hotdog + side = 39

theorem side_to_hotdog_ratio (chicken hamburger hotdog side : ℝ) :
  food_weights chicken hamburger hotdog side →
  side / hotdog = 1 / 2 := by
  sorry

end side_to_hotdog_ratio_l3813_381379


namespace fraction_addition_l3813_381384

theorem fraction_addition (x y : ℚ) (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 := by
  sorry

end fraction_addition_l3813_381384


namespace monic_quadratic_with_complex_root_l3813_381304

/-- A monic quadratic polynomial with real coefficients -/
def MonicQuadratic (a b : ℝ) : ℂ → ℂ := fun x ↦ x^2 + a*x + b

/-- The given complex number that is a root of the polynomial -/
def givenRoot : ℂ := 2 - 3*Complex.I

theorem monic_quadratic_with_complex_root :
  ∃! (a b : ℝ), (MonicQuadratic a b givenRoot = 0) ∧ (a = -4 ∧ b = 13) := by
  sorry

end monic_quadratic_with_complex_root_l3813_381304


namespace roger_birthday_money_l3813_381339

/-- Calculates the amount of birthday money Roger received -/
def birthday_money (initial_amount spent_amount final_amount : ℤ) : ℤ :=
  final_amount - initial_amount + spent_amount

/-- Proves that Roger received 28 dollars for his birthday -/
theorem roger_birthday_money :
  birthday_money 16 25 19 = 28 := by
  sorry

end roger_birthday_money_l3813_381339


namespace tom_age_l3813_381335

theorem tom_age (carla dave emily tom : ℕ) : 
  tom = 2 * carla - 1 →
  dave = carla + 3 →
  emily = carla / 2 →
  carla + dave + emily + tom = 48 →
  tom = 19 := by
  sorry

end tom_age_l3813_381335


namespace lindsey_squat_weight_l3813_381397

/-- The weight Lindsey will squat given exercise bands and a dumbbell -/
theorem lindsey_squat_weight 
  (num_bands : ℕ) 
  (resistance_per_band : ℕ) 
  (dumbbell_weight : ℕ) 
  (h1 : num_bands = 2)
  (h2 : resistance_per_band = 5)
  (h3 : dumbbell_weight = 10) :
  num_bands * resistance_per_band + dumbbell_weight = 20 := by
  sorry

end lindsey_squat_weight_l3813_381397


namespace cooler_capacity_sum_l3813_381334

theorem cooler_capacity_sum (c1 c2 c3 : ℝ) : 
  c1 = 100 →
  c2 = c1 + c1 * 0.5 →
  c3 = c2 / 2 →
  c1 + c2 + c3 = 325 := by
sorry

end cooler_capacity_sum_l3813_381334


namespace count_integers_in_list_integers_in_list_D_l3813_381333

def consecutive_integers (start : Int) (count : Nat) : List Int :=
  List.range count |>.map (fun i => start + i)

theorem count_integers_in_list (start : Int) (positive_range : Nat) : 
  let list := consecutive_integers start (positive_range + start.natAbs + 1)
  list.length = positive_range + start.natAbs + 1 :=
by sorry

-- The main theorem
theorem integers_in_list_D : 
  let start := -4
  let positive_range := 6
  let list_D := consecutive_integers start (positive_range + start.natAbs + 1)
  list_D.length = 12 :=
by sorry

end count_integers_in_list_integers_in_list_D_l3813_381333


namespace inequality_solution_l3813_381314

theorem inequality_solution :
  {x : ℝ | 0 ≤ x^2 - x - 2 ∧ x^2 - x - 2 ≤ 4} =
  {x : ℝ | (-2 ≤ x ∧ x ≤ -1) ∨ (2 ≤ x ∧ x ≤ 3)} := by
  sorry

end inequality_solution_l3813_381314


namespace no_solution_exists_l3813_381355

theorem no_solution_exists : ¬∃ x : ℝ, x^2 * 1 * 3 - x * 1 * 3^2 = 6 := by sorry

end no_solution_exists_l3813_381355


namespace green_shells_count_l3813_381394

/-- Proves that the number of green shells is 49 given the total number of shells,
    the number of red shells, and the number of shells that are not red or green. -/
theorem green_shells_count
  (total_shells : ℕ)
  (red_shells : ℕ)
  (not_red_or_green_shells : ℕ)
  (h1 : total_shells = 291)
  (h2 : red_shells = 76)
  (h3 : not_red_or_green_shells = 166) :
  total_shells - not_red_or_green_shells - red_shells = 49 :=
by sorry

end green_shells_count_l3813_381394


namespace tenth_term_of_sequence_l3813_381344

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n ≥ 2, a n - a (n - 1) = 2) ∧ (a 1 = 1)

theorem tenth_term_of_sequence (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 10 = 19 := by
  sorry

end tenth_term_of_sequence_l3813_381344


namespace staircase_extension_l3813_381346

def toothpicks_for_step (n : ℕ) : ℕ := 12 + 2 * (n - 5)

theorem staircase_extension : 
  (toothpicks_for_step 5) + (toothpicks_for_step 6) = 26 :=
by sorry

end staircase_extension_l3813_381346


namespace absolute_value_inequality_solution_range_l3813_381315

theorem absolute_value_inequality_solution_range (m : ℝ) :
  (∃ x : ℝ, |x + 2| - |x + 3| > m) → m < -1 := by
  sorry

end absolute_value_inequality_solution_range_l3813_381315


namespace no_extrema_on_open_interval_l3813_381309

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem no_extrema_on_open_interval : 
  ¬ (∃ (x : ℝ), x ∈ Set.Ioo (-1) 1 ∧ (∀ (y : ℝ), y ∈ Set.Ioo (-1) 1 → f y ≤ f x)) ∧
  ¬ (∃ (x : ℝ), x ∈ Set.Ioo (-1) 1 ∧ (∀ (y : ℝ), y ∈ Set.Ioo (-1) 1 → f y ≥ f x)) :=
by sorry

end no_extrema_on_open_interval_l3813_381309


namespace negation_of_proposition_l3813_381302

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 0 → x^2 + x + 1 > 0)) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + x₀ + 1 ≤ 0) :=
by sorry

end negation_of_proposition_l3813_381302


namespace cube_root_equation_solution_l3813_381321

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 + x / 3) ^ (1/3 : ℝ) = 2 :=
by
  sorry

end cube_root_equation_solution_l3813_381321


namespace correct_financial_equation_l3813_381356

/-- Represents Howard's financial transactions -/
def howards_finances (W D X Y : ℝ) : Prop :=
  let initial_money : ℝ := 26
  let final_money : ℝ := 52
  let window_washing_income : ℝ := W
  let dog_walking_income : ℝ := D
  let window_supplies_expense : ℝ := X
  let dog_treats_expense : ℝ := Y
  initial_money + window_washing_income + dog_walking_income - window_supplies_expense - dog_treats_expense = final_money

theorem correct_financial_equation (W D X Y : ℝ) :
  howards_finances W D X Y ↔ 26 + W + D - X - Y = 52 := by sorry

end correct_financial_equation_l3813_381356


namespace ellipse_semi_minor_axis_l3813_381387

/-- Given an ellipse with specified center, focus, and semi-major axis endpoint,
    prove that its semi-minor axis has length √8. -/
theorem ellipse_semi_minor_axis 
  (center : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (semi_major_endpoint : ℝ × ℝ)
  (h_center : center = (1, -2))
  (h_focus : focus = (1, -3))
  (h_semi_major : semi_major_endpoint = (1, 1)) :
  let c := Real.sqrt ((center.1 - focus.1)^2 + (center.2 - focus.2)^2)
  let a := Real.sqrt ((center.1 - semi_major_endpoint.1)^2 + (center.2 - semi_major_endpoint.2)^2)
  let b := Real.sqrt (a^2 - c^2)
  b = Real.sqrt 8 := by sorry

end ellipse_semi_minor_axis_l3813_381387


namespace polynomial_division_l3813_381301

theorem polynomial_division (x : ℝ) (h : x ≠ 0) :
  (6 * x^4 - 8 * x^3) / (-2 * x^2) = -3 * x^2 + 4 * x := by
  sorry

end polynomial_division_l3813_381301


namespace geometric_sequence_sum_l3813_381308

def geometric_sequence (a : ℕ → ℤ) (r : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℤ) (r : ℤ) :
  geometric_sequence a r → a 1 = 1 → r = -2 →
  a 1 + |a 2| + |a 3| + a 4 = 15 := by
  sorry

end geometric_sequence_sum_l3813_381308


namespace imaginary_part_of_reciprocal_l3813_381370

theorem imaginary_part_of_reciprocal (z : ℂ) : z = 1 - 3*I → (1/z).im = 3/10 := by
  sorry

end imaginary_part_of_reciprocal_l3813_381370


namespace no_solution_exists_l3813_381365

def matrix (y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3*y, 4],
    ![2*y, y]]

theorem no_solution_exists (y : ℝ) (h : y + 1 = 0) :
  ¬ ∃ y, (3 * y^2 - 8 * y = 5 ∧ y + 1 = 0) := by
  sorry

end no_solution_exists_l3813_381365


namespace apple_cost_calculation_l3813_381322

/-- Given that 3 dozen apples cost $23.40, prove that 5 dozen apples at the same rate cost $39.00 -/
theorem apple_cost_calculation (cost_three_dozen : ℝ) (h1 : cost_three_dozen = 23.40) :
  let cost_per_dozen : ℝ := cost_three_dozen / 3
  let cost_five_dozen : ℝ := 5 * cost_per_dozen
  cost_five_dozen = 39.00 := by
sorry

end apple_cost_calculation_l3813_381322


namespace group_size_problem_l3813_381386

theorem group_size_problem (n : ℕ) (h : ℝ) : 
  (n : ℝ) * ((n : ℝ) * h) = 362525 → n = 5 := by
  sorry

end group_size_problem_l3813_381386


namespace city_area_most_reliable_xiao_liang_most_reliable_l3813_381363

/-- Represents a survey method for assessing elderly health conditions -/
inductive SurveyMethod
  | Hospital
  | SquareDancing
  | CityArea

/-- Represents the reliability of a survey method -/
def reliability (method : SurveyMethod) : ℕ :=
  match method with
  | .Hospital => 1
  | .SquareDancing => 2
  | .CityArea => 3

/-- Theorem stating that the CityArea survey method is the most reliable -/
theorem city_area_most_reliable :
  ∀ (method : SurveyMethod), method ≠ SurveyMethod.CityArea →
    reliability method < reliability SurveyMethod.CityArea :=
by sorry

/-- Corollary: Xiao Liang's survey (CityArea) is the most reliable -/
theorem xiao_liang_most_reliable :
  reliability SurveyMethod.CityArea = max (reliability SurveyMethod.Hospital)
    (max (reliability SurveyMethod.SquareDancing) (reliability SurveyMethod.CityArea)) :=
by sorry

end city_area_most_reliable_xiao_liang_most_reliable_l3813_381363


namespace gym_cost_theorem_l3813_381300

/-- Calculates the total cost for gym memberships and personal training for one year -/
def total_gym_cost (cheap_monthly : ℝ) (cheap_signup : ℝ) (cheap_maintenance : ℝ)
                   (expensive_monthly_factor : ℝ) (expensive_signup_months : ℝ) (expensive_maintenance : ℝ)
                   (signup_discount : ℝ) (cheap_pt_base : ℝ) (cheap_pt_discount : ℝ)
                   (expensive_pt_base : ℝ) (expensive_pt_discount : ℝ) : ℝ :=
  let cheap_total := cheap_monthly * 12 + cheap_signup * (1 - signup_discount) + cheap_maintenance +
                     (cheap_pt_base * 10 + cheap_pt_base * (1 - cheap_pt_discount) * 10)
  let expensive_monthly := cheap_monthly * expensive_monthly_factor
  let expensive_total := expensive_monthly * 12 + (expensive_monthly * expensive_signup_months) * (1 - signup_discount) +
                         expensive_maintenance + (expensive_pt_base * 5 + expensive_pt_base * (1 - expensive_pt_discount) * 10)
  cheap_total + expensive_total

/-- The theorem states that the total gym cost for the given parameters is $1780.50 -/
theorem gym_cost_theorem :
  total_gym_cost 10 50 30 3 4 60 0.1 25 0.2 45 0.15 = 1780.50 := by
  sorry

end gym_cost_theorem_l3813_381300


namespace g_2000_divisors_l3813_381316

/-- g(n) is the smallest power of 5 such that 1/g(n) has exactly n digits after the decimal point -/
def g (n : ℕ) : ℕ := 5^n

/-- The number of positive integer divisors of x -/
def num_divisors (x : ℕ) : ℕ := sorry

theorem g_2000_divisors : num_divisors (g 2000) = 2001 := by sorry

end g_2000_divisors_l3813_381316


namespace circular_mat_radius_increase_l3813_381361

theorem circular_mat_radius_increase (initial_circumference final_circumference : ℝ) 
  (h1 : initial_circumference = 40)
  (h2 : final_circumference = 50) : 
  (final_circumference / (2 * Real.pi)) - (initial_circumference / (2 * Real.pi)) = 5 / Real.pi :=
by sorry

end circular_mat_radius_increase_l3813_381361


namespace quadratic_polynomial_unique_l3813_381388

theorem quadratic_polynomial_unique (q : ℝ → ℝ) : 
  (∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c) →
  q (-4) = 17 →
  q 1 = 2 →
  q 3 = 10 →
  ∀ x, q x = x^2 + 1 :=
by
  sorry

end quadratic_polynomial_unique_l3813_381388


namespace distance_before_collision_l3813_381392

/-- The distance between two boats one minute before collision -/
theorem distance_before_collision (v1 v2 d : ℝ) (hv1 : v1 = 5) (hv2 : v2 = 21) (hd : d = 20) :
  let total_speed := v1 + v2
  let time_to_collision := d / total_speed
  let distance_per_minute := total_speed / 60
  distance_per_minute = 0.4333 := by sorry

end distance_before_collision_l3813_381392


namespace chimney_bricks_l3813_381354

/-- Represents the time (in hours) it takes Brenda to build the chimney alone -/
def brenda_time : ℝ := 8

/-- Represents the time (in hours) it takes Bob to build the chimney alone -/
def bob_time : ℝ := 12

/-- Represents the decrease in productivity (in bricks per hour) when working together -/
def productivity_decrease : ℝ := 15

/-- Represents the time (in hours) it takes Brenda and Bob to build the chimney together -/
def joint_time : ℝ := 6

/-- Theorem stating that the number of bricks in the chimney is 360 -/
theorem chimney_bricks : ℝ := by
  sorry

end chimney_bricks_l3813_381354


namespace mika_stickers_l3813_381320

/-- The number of stickers Mika has left after a series of transactions -/
def stickers_left (initial : Float) (bought : Float) (birthday : Float) (from_friend : Float)
  (to_sister : Float) (used : Float) (sold : Float) : Float :=
  initial + bought + birthday + from_friend - to_sister - used - sold

/-- Theorem stating that Mika has 6 stickers left after the given transactions -/
theorem mika_stickers :
  stickers_left 20.5 26.25 19.75 7.5 6.3 58.5 3.2 = 6 := by
  sorry

end mika_stickers_l3813_381320


namespace group_size_calculation_l3813_381348

theorem group_size_calculation (average_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  average_increase = 3 →
  old_weight = 70 →
  new_weight = 94 →
  (new_weight - old_weight) / average_increase = 8 := by
sorry

end group_size_calculation_l3813_381348


namespace enlarged_parallelepiped_volume_equals_l3813_381347

/-- The volume of the set of points that are inside or within one unit of a rectangular parallelepiped with dimensions 4 by 5 by 6 units -/
def enlarged_parallelepiped_volume : ℝ := sorry

/-- The dimensions of the original parallelepiped -/
def original_dimensions : Fin 3 → ℕ
| 0 => 4
| 1 => 5
| 2 => 6
| _ => 0

theorem enlarged_parallelepiped_volume_equals : 
  enlarged_parallelepiped_volume = (1884 + 139 * Real.pi) / 3 := by sorry

end enlarged_parallelepiped_volume_equals_l3813_381347


namespace number_comparisons_l3813_381378

theorem number_comparisons :
  (0.5 < 0.8) ∧ (0.5 < 0.7) ∧ (Real.log 125 < Real.log 1215) := by
  sorry

end number_comparisons_l3813_381378


namespace range_of_a_when_not_p_range_of_m_when_p_necessary_not_sufficient_l3813_381398

-- Define the propositions
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - a*x + a + 3 = 0
def q (m a : ℝ) : Prop := m - 1 ≤ a ∧ a ≤ m + 1

-- Theorem 1: When ¬p is true, a ∈ (-2, 6)
theorem range_of_a_when_not_p :
  ∀ a : ℝ, ¬(p a) → -2 < a ∧ a < 6 :=
sorry

-- Theorem 2: When p is necessary but not sufficient for q, m ∈ (-∞, -3] ∪ [7, +∞)
theorem range_of_m_when_p_necessary_not_sufficient :
  ∀ m : ℝ, (∀ a : ℝ, q m a → p a) ∧ (∃ a : ℝ, p a ∧ ¬(q m a)) →
  m ≤ -3 ∨ m ≥ 7 :=
sorry

end range_of_a_when_not_p_range_of_m_when_p_necessary_not_sufficient_l3813_381398


namespace triangle_properties_l3813_381399

open Real

theorem triangle_properties (A B C : ℝ) (R : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  R = 1 →
  (sin A - sin B + sin C) / sin C = sin B / (sin A + sin B - sin C) →
  ∃ (S : ℝ),
    A = π / 3 ∧
    S ≤ 3 * sqrt 3 / 4 ∧
    (∀ (S' : ℝ), S' ≤ S) :=
by sorry

end triangle_properties_l3813_381399


namespace percentage_equality_l3813_381329

theorem percentage_equality (x : ℝ) : (60 / 100 * 500 = 50 / 100 * x) → x = 600 :=
by sorry

end percentage_equality_l3813_381329


namespace percentage_problem_l3813_381391

theorem percentage_problem (x : ℝ) (hx : x > 0) : 
  x / 100 * 150 - 20 = 10 → x = 20 := by sorry

end percentage_problem_l3813_381391


namespace sum_of_squares_is_45_l3813_381327

/-- Represents the ages of Alice, Bob, and Charlie -/
structure Ages where
  alice : ℕ
  bob : ℕ
  charlie : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  (3 * ages.alice + 2 * ages.bob = 4 * ages.charlie) ∧
  (3 * ages.charlie^2 = 4 * ages.alice^2 + 2 * ages.bob^2) ∧
  (Nat.gcd ages.alice ages.bob = 1) ∧
  (Nat.gcd ages.alice ages.charlie = 1) ∧
  (Nat.gcd ages.bob ages.charlie = 1)

/-- The theorem to be proved -/
theorem sum_of_squares_is_45 (ages : Ages) :
  satisfies_conditions ages →
  ages.alice^2 + ages.bob^2 + ages.charlie^2 = 45 := by
  sorry

end sum_of_squares_is_45_l3813_381327


namespace platform_length_l3813_381389

/-- The length of a platform given train speed and crossing times -/
theorem platform_length
  (train_speed : ℝ)
  (platform_crossing_time : ℝ)
  (man_crossing_time : ℝ)
  (h1 : train_speed = 72)  -- km/h
  (h2 : platform_crossing_time = 30)  -- seconds
  (h3 : man_crossing_time = 15)  -- seconds
  : ∃ (platform_length : ℝ), platform_length = 300 :=
by
  sorry

end platform_length_l3813_381389


namespace mike_earnings_l3813_381377

def mower_blade_cost : ℕ := 10
def game_cost : ℕ := 8
def number_of_games : ℕ := 4

def total_money_earned : ℕ :=
  mower_blade_cost + number_of_games * game_cost

theorem mike_earnings : total_money_earned = 42 := by
  sorry

end mike_earnings_l3813_381377


namespace dot_product_of_vectors_l3813_381374

def vector_a : ℝ × ℝ := (-4, 7)
def vector_b : ℝ × ℝ := (5, 2)

theorem dot_product_of_vectors :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) = -6 := by sorry

end dot_product_of_vectors_l3813_381374


namespace digit_150_is_3_l3813_381393

/-- The decimal expansion of 5/37 has a repeating block of length 3 -/
def repeating_block_length : ℕ := 3

/-- The repeating block in the decimal expansion of 5/37 is [1, 3, 5] -/
def repeating_block : List ℕ := [1, 3, 5]

/-- The 150th digit after the decimal point in the decimal expansion of 5/37 -/
def digit_150 : ℕ := repeating_block[(150 - 1) % repeating_block_length]

theorem digit_150_is_3 : digit_150 = 3 := by
  sorry

end digit_150_is_3_l3813_381393


namespace diagonal_difference_l3813_381385

/-- The number of diagonals in a convex polygon with n sides -/
def f (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The difference between the number of diagonals in a convex polygon
    with n+1 sides and n sides is n-1, for n ≥ 4 -/
theorem diagonal_difference (n : ℕ) (h : n ≥ 4) : f (n + 1) - f n = n - 1 :=
  sorry

end diagonal_difference_l3813_381385


namespace square_side_length_range_l3813_381351

theorem square_side_length_range (a : ℝ) : 
  (a > 0) → (a^2 = 37) → (6 < a ∧ a < 7) := by
  sorry

end square_side_length_range_l3813_381351


namespace ernie_circles_l3813_381310

theorem ernie_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) 
  (ali_circles : ℕ) (h1 : total_boxes = 80) (h2 : ali_boxes_per_circle = 8) 
  (h3 : ernie_boxes_per_circle = 10) (h4 : ali_circles = 5) : 
  (total_boxes - ali_circles * ali_boxes_per_circle) / ernie_boxes_per_circle = 4 := by
  sorry

end ernie_circles_l3813_381310


namespace net_salary_proof_l3813_381360

/-- Represents a person's monthly financial situation -/
structure MonthlySalary where
  net : ℝ
  discretionary : ℝ
  remaining : ℝ

/-- Calculates the net monthly salary given the conditions -/
def calculate_net_salary (m : MonthlySalary) : Prop :=
  m.discretionary = m.net / 5 ∧
  m.remaining = m.discretionary * 0.1 ∧
  m.remaining = 105 ∧
  m.net = 5250

theorem net_salary_proof (m : MonthlySalary) :
  calculate_net_salary m → m.net = 5250 := by
  sorry

end net_salary_proof_l3813_381360


namespace negative_sixty_four_to_four_thirds_l3813_381358

theorem negative_sixty_four_to_four_thirds : (-64 : ℝ) ^ (4/3) = 256 := by
  sorry

end negative_sixty_four_to_four_thirds_l3813_381358


namespace count_proposition_permutations_l3813_381317

/-- The number of distinct permutations of letters in "PROPOSITION" -/
def proposition_permutations : ℕ :=
  Nat.factorial 10 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating the number of distinct permutations of "PROPOSITION" -/
theorem count_proposition_permutations :
  proposition_permutations = 453600 := by
  sorry

end count_proposition_permutations_l3813_381317


namespace fixed_costs_correct_l3813_381369

/-- Represents the fixed monthly costs for producing electronic components -/
def fixed_monthly_costs : ℝ := 16500

/-- Represents the production cost per component -/
def production_cost_per_unit : ℝ := 80

/-- Represents the shipping cost per component -/
def shipping_cost_per_unit : ℝ := 5

/-- Represents the number of components produced and sold monthly -/
def monthly_units : ℕ := 150

/-- Represents the lowest selling price per component -/
def lowest_selling_price : ℝ := 195

/-- Theorem stating that the fixed monthly costs are correct given the conditions -/
theorem fixed_costs_correct :
  fixed_monthly_costs =
    monthly_units * lowest_selling_price -
    monthly_units * (production_cost_per_unit + shipping_cost_per_unit) := by
  sorry

#check fixed_costs_correct

end fixed_costs_correct_l3813_381369


namespace larger_number_l3813_381359

theorem larger_number (a b : ℝ) (h1 : a - b = 6) (h2 : a + b = 40) : max a b = 23 := by
  sorry

end larger_number_l3813_381359


namespace equal_area_rectangles_l3813_381326

/-- Given two rectangles of equal area, where one rectangle has dimensions 6 inches by 50 inches,
    and the other has a width of 20 inches, prove that the length of the second rectangle is 15 inches. -/
theorem equal_area_rectangles (area : ℝ) (length_jordan width_jordan width_carol : ℝ) :
  area = length_jordan * width_jordan →
  length_jordan = 6 →
  width_jordan = 50 →
  width_carol = 20 →
  ∃ length_carol : ℝ, area = length_carol * width_carol ∧ length_carol = 15 := by
  sorry

end equal_area_rectangles_l3813_381326


namespace tan_sum_pi_twelfths_l3813_381366

theorem tan_sum_pi_twelfths : 
  Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 * Real.sqrt 2 - 4 := by
  sorry

end tan_sum_pi_twelfths_l3813_381366


namespace athlete_heartbeats_l3813_381336

/-- Calculates the total number of heartbeats during a race --/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

/-- Proves that the athlete's heart beats 28800 times during the race --/
theorem athlete_heartbeats :
  total_heartbeats 160 30 6 = 28800 := by
  sorry

end athlete_heartbeats_l3813_381336


namespace bowling_ball_weight_is_14_l3813_381303

/-- The weight of a bowling ball in pounds -/
def bowling_ball_weight : ℝ := sorry

/-- The weight of a canoe in pounds -/
def canoe_weight : ℝ := sorry

/-- Theorem stating that one bowling ball weighs 14 pounds -/
theorem bowling_ball_weight_is_14 : bowling_ball_weight = 14 := by
  have h1 : 8 * bowling_ball_weight = 4 * canoe_weight := sorry
  have h2 : 3 * canoe_weight = 84 := sorry
  sorry


end bowling_ball_weight_is_14_l3813_381303


namespace min_value_theorem_l3813_381367

theorem min_value_theorem (p q r s t u v w : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) 
  (ht : t > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h1 : p * q * r * s = 16)
  (h2 : t * u * v * w = 25)
  (h3 : p * t = q * u)
  (h4 : p * t = r * v)
  (h5 : p * t = s * w) :
  (∀ x : ℝ, (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 80) ∧
  (∃ x : ℝ, (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 = 80) :=
by sorry

end min_value_theorem_l3813_381367


namespace homework_group_existence_l3813_381324

theorem homework_group_existence :
  ∀ (S : Finset ℕ) (f : Finset ℕ → Finset ℕ → Prop),
    S.card = 21 →
    (∀ a b c : ℕ, a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c →
      (f {a, b, c} {0} ∨ f {a, b, c} {1}) ∧
      ¬(f {a, b, c} {0} ∧ f {a, b, c} {1})) →
    ∃ T : Finset ℕ, T ⊆ S ∧ T.card = 4 ∧
      (∀ a b c : ℕ, a ∈ T → b ∈ T → c ∈ T → a ≠ b → b ≠ c → a ≠ c →
        (f {a, b, c} {0} ∨ f {a, b, c} {1})) :=
by sorry


end homework_group_existence_l3813_381324


namespace alex_income_l3813_381325

/-- Represents the tax structure and Alex's tax payment --/
structure TaxSystem where
  q : ℝ  -- Base tax rate as a percentage
  income : ℝ  -- Alex's annual income
  total_tax : ℝ  -- Total tax paid by Alex

/-- The tax system satisfies the given conditions --/
def valid_tax_system (ts : TaxSystem) : Prop :=
  ts.total_tax = 
    (if ts.income ≤ 50000 then
      (ts.q / 100) * ts.income
    else
      (ts.q / 100) * 50000 + ((ts.q + 3) / 100) * (ts.income - 50000))
  ∧ ts.total_tax = ((ts.q + 0.5) / 100) * ts.income

/-- Theorem stating that Alex's income is $60000 --/
theorem alex_income (ts : TaxSystem) (h : valid_tax_system ts) : ts.income = 60000 := by
  sorry

end alex_income_l3813_381325


namespace arithmetic_sequence_first_term_l3813_381375

theorem arithmetic_sequence_first_term 
  (a : ℚ) -- First term of the sequence
  (d : ℚ) -- Common difference of the sequence
  (h1 : (30 : ℚ) / 2 * (a + (a + 29 * d)) = 600) -- Sum of first 30 terms
  (h2 : (30 : ℚ) / 2 * ((a + 30 * d) + (a + 59 * d)) = 2100) -- Sum of next 30 terms
  : a = 5 / 6 := by
  sorry

end arithmetic_sequence_first_term_l3813_381375


namespace roots_relation_l3813_381364

-- Define the polynomials
def f (x : ℝ) : ℝ := x^3 + 5*x^2 + 6*x - 8
def g (x u v w : ℝ) : ℝ := x^3 + u*x^2 + v*x + w

-- Define the theorem
theorem roots_relation (p q r u v w : ℝ) : 
  (f p = 0 ∧ f q = 0 ∧ f r = 0) → 
  (g (p+q) u v w = 0 ∧ g (q+r) u v w = 0 ∧ g (r+p) u v w = 0) →
  w = 8 := by
sorry

end roots_relation_l3813_381364


namespace cubic_difference_l3813_381306

theorem cubic_difference (x y : ℝ) (h1 : x + y = 14) (h2 : 3 * x + y = 20) :
  x^3 - y^3 = -1304 := by
sorry

end cubic_difference_l3813_381306


namespace total_situps_is_110_l3813_381396

/-- The number of situps Diana did -/
def diana_situps : ℕ := 40

/-- Diana's rate of situps per minute -/
def diana_rate : ℕ := 4

/-- The difference between Hani's and Diana's situp rates -/
def rate_difference : ℕ := 3

/-- Calculates the total number of situps done by Hani and Diana together -/
def total_situps : ℕ :=
  let diana_time := diana_situps / diana_rate
  let hani_rate := diana_rate + rate_difference
  let hani_situps := hani_rate * diana_time
  diana_situps + hani_situps

/-- Theorem stating that the total number of situps is 110 -/
theorem total_situps_is_110 : total_situps = 110 := by
  sorry

end total_situps_is_110_l3813_381396


namespace purely_imaginary_modulus_l3813_381330

theorem purely_imaginary_modulus (a : ℝ) :
  let z : ℂ := (a + 3 * Complex.I) / (1 + 2 * Complex.I)
  (∃ b : ℝ, z = b * Complex.I) → Complex.abs z = 3 := by
  sorry

end purely_imaginary_modulus_l3813_381330


namespace age_difference_proof_l3813_381340

def age_difference (a b c : ℕ) : ℕ := (a + b) - (b + c)

theorem age_difference_proof (a b c : ℕ) (h : c = a - 11) :
  age_difference a b c = 11 := by
  sorry

end age_difference_proof_l3813_381340


namespace deduction_from_second_number_l3813_381319

theorem deduction_from_second_number 
  (n : ℕ) 
  (avg_initial : ℚ)
  (avg_final : ℚ)
  (deduct_first : ℚ)
  (deduct_third : ℚ)
  (deduct_fourth_to_ninth : List ℚ)
  (h1 : n = 10)
  (h2 : avg_initial = 16)
  (h3 : avg_final = 11.5)
  (h4 : deduct_first = 9)
  (h5 : deduct_third = 7)
  (h6 : deduct_fourth_to_ninth = [6, 5, 4, 3, 2, 1]) :
  ∃ (deduct_second : ℚ), deduct_second = 8 ∧
    (n * avg_final = n * avg_initial - 
      (deduct_first + deduct_second + deduct_third + 
       deduct_fourth_to_ninth.sum)) :=
by sorry

end deduction_from_second_number_l3813_381319


namespace kenny_friday_jacks_l3813_381362

/-- The number of jumping jacks Kenny did last week -/
def last_week_total : ℕ := 324

/-- The number of jumping jacks Kenny did on Sunday -/
def sunday_jacks : ℕ := 34

/-- The number of jumping jacks Kenny did on Monday -/
def monday_jacks : ℕ := 20

/-- The number of jumping jacks Kenny did on Tuesday -/
def tuesday_jacks : ℕ := 0

/-- The number of jumping jacks Kenny did on Wednesday -/
def wednesday_jacks : ℕ := 123

/-- The number of jumping jacks Kenny did on Thursday -/
def thursday_jacks : ℕ := 64

/-- The number of jumping jacks Kenny did on some unspecified day -/
def some_day_jacks : ℕ := 61

/-- The number of jumping jacks Kenny did on Friday -/
def friday_jacks : ℕ := 23

/-- Theorem stating that Kenny did 23 jumping jacks on Friday -/
theorem kenny_friday_jacks : 
  friday_jacks = 23 ∧ 
  friday_jacks + sunday_jacks + monday_jacks + tuesday_jacks + wednesday_jacks + thursday_jacks + some_day_jacks > last_week_total :=
by sorry

end kenny_friday_jacks_l3813_381362


namespace church_capacity_l3813_381343

/-- Calculates the total number of people that can sit in a church when it's full -/
theorem church_capacity (rows : ℕ) (chairs_per_row : ℕ) (people_per_chair : ℕ) : 
  rows = 20 → chairs_per_row = 6 → people_per_chair = 5 → 
  rows * chairs_per_row * people_per_chair = 600 := by
  sorry

#check church_capacity

end church_capacity_l3813_381343


namespace rebecca_current_income_l3813_381332

/-- Rebecca's current yearly income --/
def rebecca_income : ℝ := sorry

/-- Jimmy's annual income --/
def jimmy_income : ℝ := 18000

/-- The increase in Rebecca's income --/
def income_increase : ℝ := 7000

/-- The percentage of Rebecca's new income in their combined income --/
def rebecca_percentage : ℝ := 0.55

theorem rebecca_current_income :
  rebecca_income = 15000 ∧
  (rebecca_income + income_increase) = 
    rebecca_percentage * (rebecca_income + income_increase + jimmy_income) :=
by sorry

end rebecca_current_income_l3813_381332


namespace profit_calculation_l3813_381350

-- Define the number of items bought and the price paid
def items_bought : ℕ := 60
def price_paid : ℕ := 46

-- Define the discount rate
def discount_rate : ℚ := 1 / 100

-- Define a function to calculate the profit percent
def profit_percent (items : ℕ) (price : ℕ) (discount : ℚ) : ℚ :=
  let cost_per_item : ℚ := price / items
  let selling_price : ℚ := 1 - discount
  let profit_per_item : ℚ := selling_price - cost_per_item
  (profit_per_item / cost_per_item) * 100

-- State the theorem
theorem profit_calculation :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  abs (profit_percent items_bought price_paid discount_rate - 2911/100) < ε :=
sorry

end profit_calculation_l3813_381350


namespace asha_win_probability_l3813_381338

theorem asha_win_probability (p_lose p_tie : ℚ) 
  (h_lose : p_lose = 3/7)
  (h_tie : p_tie = 1/5) :
  1 - p_lose - p_tie = 13/35 := by
  sorry

end asha_win_probability_l3813_381338


namespace squirrels_in_tree_l3813_381380

theorem squirrels_in_tree (nuts : ℕ) (squirrels : ℕ) 
  (h1 : nuts = 2)
  (h2 : squirrels - nuts = 2) :
  squirrels = 4 := by
  sorry

end squirrels_in_tree_l3813_381380


namespace average_xyz_l3813_381372

theorem average_xyz (x y z : ℝ) (h : (5 / 2) * (x + y + z) = 20) : 
  (x + y + z) / 3 = 8 / 3 := by
sorry

end average_xyz_l3813_381372


namespace integer_solutions_yk_eq_x2_plus_x_l3813_381342

theorem integer_solutions_yk_eq_x2_plus_x (k : ℕ) (hk : k > 1) :
  ∀ x y : ℤ, y^k = x^2 + x ↔ (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = 0) := by
  sorry

end integer_solutions_yk_eq_x2_plus_x_l3813_381342


namespace markup_calculation_l3813_381337

theorem markup_calculation (purchase_price overhead_percentage net_profit : ℝ) 
  (h1 : purchase_price = 48)
  (h2 : overhead_percentage = 0.35)
  (h3 : net_profit = 18) :
  purchase_price + purchase_price * overhead_percentage + net_profit - purchase_price = 34.80 := by
  sorry

end markup_calculation_l3813_381337


namespace right_focus_coordinates_l3813_381311

/-- The coordinates of the right focus of a hyperbola with equation x^2 - 2y^2 = 1 -/
theorem right_focus_coordinates :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 - 2*y^2 = 1}
  ∃ (f : ℝ × ℝ), f ∈ hyperbola ∧ f.1 > 0 ∧ f.2 = 0 ∧ 
    ∀ (p : ℝ × ℝ), p ∈ hyperbola ∧ p.1 > 0 ∧ p.2 = 0 → p = f ∧
    f = (Real.sqrt (3/2), 0) :=
by sorry

end right_focus_coordinates_l3813_381311


namespace functional_equation_solution_l3813_381307

/-- The functional equation solution for f(x+y) f(x-y) = (f(x))^2 -/
theorem functional_equation_solution (f : ℝ → ℝ) (hf : Continuous f) 
  (h : ∀ x y : ℝ, f (x + y) * f (x - y) = (f x)^2) :
  ∃ a c : ℝ, ∀ x : ℝ, f x = a * (c^x) := by
sorry

end functional_equation_solution_l3813_381307


namespace clothing_distribution_l3813_381373

/-- Given a total of 39 pieces of clothing, with 19 pieces in the first load
    and the rest split into 5 equal loads, prove that each small load
    contains 4 pieces of clothing. -/
theorem clothing_distribution (total : Nat) (first_load : Nat) (num_small_loads : Nat)
    (h1 : total = 39)
    (h2 : first_load = 19)
    (h3 : num_small_loads = 5) :
    (total - first_load) / num_small_loads = 4 := by
  sorry

end clothing_distribution_l3813_381373


namespace optimal_container_dimensions_l3813_381349

/-- Represents the dimensions and volume of a rectangular container --/
structure Container where
  shorter_side : Real
  longer_side : Real
  height : Real
  volume : Real

/-- Calculates the volume of a container given its dimensions --/
def calculate_volume (c : Container) : Real :=
  c.shorter_side * c.longer_side * c.height

/-- Defines the constraints for the container based on the problem --/
def container_constraints (c : Container) : Prop :=
  c.longer_side = c.shorter_side + 0.5 ∧
  c.height = 3.2 - 2 * c.shorter_side ∧
  c.volume = calculate_volume c ∧
  0 < c.shorter_side ∧ c.shorter_side < 1.6

/-- Theorem stating the optimal dimensions and maximum volume of the container --/
theorem optimal_container_dimensions :
  ∃ (c : Container), container_constraints c ∧
    c.shorter_side = 1 ∧
    c.height = 1.2 ∧
    c.volume = 1.8 ∧
    ∀ (c' : Container), container_constraints c' → c'.volume ≤ c.volume :=
  sorry

end optimal_container_dimensions_l3813_381349


namespace M_intersect_N_equals_N_l3813_381371

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = x^2 - 2}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 - 2}

-- Statement to prove
theorem M_intersect_N_equals_N : M ∩ N = N := by sorry

end M_intersect_N_equals_N_l3813_381371


namespace square_root_of_1024_l3813_381328

theorem square_root_of_1024 (y : ℝ) (h1 : y > 0) (h2 : y^2 = 1024) : y = 32 := by
  sorry

end square_root_of_1024_l3813_381328


namespace sameTerminalSideAs315_eq_l3813_381345

/-- The set of angles with the same terminal side as 315° -/
def sameTerminalSideAs315 : Set ℝ :=
  {α | ∃ k : ℤ, α = 2 * k * Real.pi - Real.pi / 4}

/-- Theorem stating that the set of angles with the same terminal side as 315° 
    is equal to {α | α = 2kπ - π/4, k ∈ ℤ} -/
theorem sameTerminalSideAs315_eq : 
  sameTerminalSideAs315 = {α | ∃ k : ℤ, α = 2 * k * Real.pi - Real.pi / 4} := by
  sorry


end sameTerminalSideAs315_eq_l3813_381345


namespace number_sequence_count_l3813_381323

/-- The total number of numbers in the sequence -/
def n : ℕ := 8

/-- The average of all numbers -/
def total_average : ℚ := 25

/-- The average of the first two numbers -/
def first_two_average : ℚ := 20

/-- The average of the next three numbers -/
def next_three_average : ℚ := 26

/-- The sixth number in the sequence -/
def sixth_number : ℚ := 14

/-- The last (eighth) number in the sequence -/
def last_number : ℚ := 30

theorem number_sequence_count :
  (2 * first_two_average + 3 * next_three_average + sixth_number + 
   (sixth_number + 4) + (sixth_number + 6) + last_number) / n = total_average := by
  sorry

#check number_sequence_count

end number_sequence_count_l3813_381323


namespace raisin_cost_fraction_nut_to_dried_fruit_ratio_dried_fruit_percentage_l3813_381341

/-- Represents the trail mix problem with raisins, nuts, and dried fruit. -/
structure TrailMix where
  x : ℝ
  raisin_cost : ℝ
  raisin_weight : ℝ := 3 * x
  nut_weight : ℝ := 4 * x
  dried_fruit_weight : ℝ := 5 * x
  nut_cost : ℝ := 3 * raisin_cost
  dried_fruit_cost : ℝ := 1.5 * raisin_cost

/-- The total cost of raisins is 1/7.5 of the total cost of the mixture. -/
theorem raisin_cost_fraction (mix : TrailMix) :
  (mix.raisin_weight * mix.raisin_cost) / 
  (mix.raisin_weight * mix.raisin_cost + mix.nut_weight * mix.nut_cost + mix.dried_fruit_weight * mix.dried_fruit_cost) = 1 / 7.5 := by
  sorry

/-- The ratio of the cost of nuts to the cost of dried fruit is 2:1. -/
theorem nut_to_dried_fruit_ratio (mix : TrailMix) :
  mix.nut_cost / mix.dried_fruit_cost = 2 := by
  sorry

/-- The total cost of dried fruit is 50% of the total cost of raisins and nuts combined. -/
theorem dried_fruit_percentage (mix : TrailMix) :
  (mix.dried_fruit_weight * mix.dried_fruit_cost) / 
  (mix.raisin_weight * mix.raisin_cost + mix.nut_weight * mix.nut_cost) = 1 / 2 := by
  sorry

end raisin_cost_fraction_nut_to_dried_fruit_ratio_dried_fruit_percentage_l3813_381341


namespace tan_double_angle_special_case_l3813_381376

theorem tan_double_angle_special_case (α : Real) 
  (h : (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3) : 
  Real.tan (2 * α) = -8/15 := by
  sorry

end tan_double_angle_special_case_l3813_381376


namespace max_m_inequality_l3813_381383

theorem max_m_inequality (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (∃ m : ℝ, ∀ x y : ℝ, x > 1 → y > 1 → x^2 / (y - 1) + y^2 / (x - 1) ≥ 3 * m - 1) ∧
  (∀ m : ℝ, (∀ x y : ℝ, x > 1 → y > 1 → x^2 / (y - 1) + y^2 / (x - 1) ≥ 3 * m - 1) → m ≤ 3) :=
by sorry

end max_m_inequality_l3813_381383
