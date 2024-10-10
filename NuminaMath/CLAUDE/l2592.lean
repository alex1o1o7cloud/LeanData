import Mathlib

namespace brownies_needed_l2592_259238

/-- Represents the amount of frosting used for different baked goods -/
structure FrostingUsage where
  layerCake : ℝ
  singleCake : ℝ
  panBrownies : ℝ
  dozenCupcakes : ℝ

/-- Represents the quantities of baked goods Paul needs to prepare -/
structure BakedGoods where
  layerCakes : ℕ
  singleCakes : ℕ
  dozenCupcakes : ℕ

def totalFrostingNeeded : ℝ := 21

theorem brownies_needed (usage : FrostingUsage) (goods : BakedGoods) 
  (h1 : usage.layerCake = 1)
  (h2 : usage.singleCake = 0.5)
  (h3 : usage.panBrownies = 0.5)
  (h4 : usage.dozenCupcakes = 0.5)
  (h5 : goods.layerCakes = 3)
  (h6 : goods.singleCakes = 12)
  (h7 : goods.dozenCupcakes = 6) :
  (totalFrostingNeeded - 
   (goods.layerCakes * usage.layerCake + 
    goods.singleCakes * usage.singleCake + 
    goods.dozenCupcakes * usage.dozenCupcakes)) / usage.panBrownies = 18 := by
  sorry

end brownies_needed_l2592_259238


namespace polynomial_evaluation_l2592_259231

theorem polynomial_evaluation :
  let y : ℤ := -2
  y^3 - y^2 + y - 1 = -7 := by sorry

end polynomial_evaluation_l2592_259231


namespace solution_sum_l2592_259222

theorem solution_sum (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 20) : x + y = 2 := by
  sorry

end solution_sum_l2592_259222


namespace f_derivative_at_zero_l2592_259201

def f (x : ℝ) : ℝ := x * (1 + x)

theorem f_derivative_at_zero : 
  (deriv f) 0 = 1 := by sorry

end f_derivative_at_zero_l2592_259201


namespace reflection_of_A_across_y_axis_l2592_259234

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- The original point A -/
def A : ℝ × ℝ := (-2, 5)

theorem reflection_of_A_across_y_axis :
  reflect_y A = (2, 5) := by sorry

end reflection_of_A_across_y_axis_l2592_259234


namespace g_minimum_value_l2592_259228

open Real

noncomputable def g (x : ℝ) : ℝ :=
  x + (2*x)/(x^2 + 1) + (x*(x + 3))/(x^2 + 3) + (3*(x + 1))/(x*(x^2 + 3))

theorem g_minimum_value (x : ℝ) (hx : x > 0) : g x ≥ 7 := by
  sorry

end g_minimum_value_l2592_259228


namespace units_digit_150_factorial_is_zero_l2592_259268

-- Define the factorial function
def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_150_factorial_is_zero :
  unitsDigit (factorial 150) = 0 :=
by sorry

end units_digit_150_factorial_is_zero_l2592_259268


namespace initial_cows_l2592_259288

theorem initial_cows (cows dogs : ℕ) : 
  cows = 2 * dogs →
  (3 / 4 : ℚ) * cows + (1 / 4 : ℚ) * dogs = 161 →
  cows = 184 := by
  sorry

end initial_cows_l2592_259288


namespace base7_25_to_binary_l2592_259233

def base7ToDecimal (n : ℕ) : ℕ :=
  (n / 10) * 7 + (n % 10)

def decimalToBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec toBinary (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else toBinary (m / 2) ((m % 2) :: acc)
  toBinary n []

theorem base7_25_to_binary :
  decimalToBinary (base7ToDecimal 25) = [1, 0, 0, 1, 1] := by
  sorry

end base7_25_to_binary_l2592_259233


namespace sum_of_ages_in_two_years_l2592_259252

def Matt_age (Fem_age : ℕ) : ℕ := 4 * Fem_age

def current_Fem_age : ℕ := 11

def future_age (current_age : ℕ) : ℕ := current_age + 2

theorem sum_of_ages_in_two_years :
  future_age (Matt_age current_Fem_age) + future_age current_Fem_age = 59 := by
  sorry

end sum_of_ages_in_two_years_l2592_259252


namespace absolute_value_of_expression_l2592_259210

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem absolute_value_of_expression : 
  Complex.abs (2 + i^2 + 2*i^3) = Real.sqrt 5 := by sorry

end absolute_value_of_expression_l2592_259210


namespace circle_intersection_range_l2592_259247

/-- The range of m for which two circles intersect -/
theorem circle_intersection_range :
  let circle1 : ℝ → ℝ → ℝ → Prop := λ x y m ↦ x^2 + y^2 = m
  let circle2 : ℝ → ℝ → Prop := λ x y ↦ x^2 + y^2 - 6*x + 8*y - 24 = 0
  ∀ m : ℝ, (∃ x y : ℝ, circle1 x y m ∧ circle2 x y) ↔ 4 < m ∧ m < 144 := by
  sorry

end circle_intersection_range_l2592_259247


namespace companion_numbers_example_companion_numbers_expression_l2592_259275

/-- Two numbers are companion numbers if their sum equals their product. -/
def CompanionNumbers (a b : ℝ) : Prop := a + b = a * b

theorem companion_numbers_example : CompanionNumbers (-1) (1/2) := by sorry

theorem companion_numbers_expression (m n : ℝ) (h : CompanionNumbers m n) :
  -2 * m * n + 1/2 * (3 * m + 2 * (1/2 * n - m) + 3 * m * n - 6) = -3 := by sorry

end companion_numbers_example_companion_numbers_expression_l2592_259275


namespace cyclists_distance_l2592_259282

/-- Calculates the distance between two cyclists traveling in opposite directions -/
def distance_between_cyclists (speed1 speed2 time : ℝ) : ℝ :=
  (speed1 * time) + (speed2 * time)

/-- Theorem stating the distance between two cyclists after 2 hours -/
theorem cyclists_distance :
  let speed1 : ℝ := 10  -- Speed of first cyclist in km/h
  let speed2 : ℝ := 15  -- Speed of second cyclist in km/h
  let time : ℝ := 2     -- Time in hours
  distance_between_cyclists speed1 speed2 time = 50 := by
  sorry

#check cyclists_distance

end cyclists_distance_l2592_259282


namespace integer_sequence_count_l2592_259293

def sequence_term (n : ℕ) : ℚ :=
  16200 / (5 ^ n)

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem integer_sequence_count : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∀ (k : ℕ), k < n → is_integer (sequence_term k)) ∧ 
    (∀ (k : ℕ), k ≥ n → ¬ is_integer (sequence_term k))) ∧
  (∃! (n : ℕ), n > 0 ∧ 
    (∀ (k : ℕ), k < n → is_integer (sequence_term k)) ∧ 
    (∀ (k : ℕ), k ≥ n → ¬ is_integer (sequence_term k))) ∧
  (∃ (n : ℕ), n = 3 ∧ 
    (∀ (k : ℕ), k < n → is_integer (sequence_term k)) ∧ 
    (∀ (k : ℕ), k ≥ n → ¬ is_integer (sequence_term k))) :=
by sorry

end integer_sequence_count_l2592_259293


namespace smallest_n_for_integer_T_l2592_259250

def K : ℚ := (1:ℚ)/1 + (1:ℚ)/3 + (1:ℚ)/5 + (1:ℚ)/7 + (1:ℚ)/9

def T (n : ℕ) : ℚ := n * (5^(n-1) : ℚ) * K

def is_integer (q : ℚ) : Prop := ∃ (z : ℤ), q = z

theorem smallest_n_for_integer_T :
  ∀ n : ℕ, (n > 0 ∧ is_integer (T n)) → n ≥ 63 :=
sorry

end smallest_n_for_integer_T_l2592_259250


namespace m_fourth_plus_n_fourth_l2592_259278

theorem m_fourth_plus_n_fourth (m n : ℝ) 
  (h1 : m - n = -5)
  (h2 : m^2 + n^2 = 13) :
  m^4 + n^4 = 97 := by
sorry

end m_fourth_plus_n_fourth_l2592_259278


namespace min_n_for_constant_term_l2592_259285

theorem min_n_for_constant_term (n : ℕ) : 
  (∃ k : ℕ, 2 * n = 5 * k) ∧ (∀ m : ℕ, m < n → ¬∃ k : ℕ, 2 * m = 5 * k) ↔ n = 5 := by
  sorry

end min_n_for_constant_term_l2592_259285


namespace total_hours_is_fifty_l2592_259218

/-- Calculates the total hours needed to make dresses from two types of fabric -/
def total_hours_for_dresses (fabric_a_total : ℕ) (fabric_a_per_dress : ℕ) (fabric_a_hours : ℕ)
                            (fabric_b_total : ℕ) (fabric_b_per_dress : ℕ) (fabric_b_hours : ℕ) : ℕ :=
  let dresses_a := fabric_a_total / fabric_a_per_dress
  let dresses_b := fabric_b_total / fabric_b_per_dress
  dresses_a * fabric_a_hours + dresses_b * fabric_b_hours

/-- Theorem stating that the total hours needed to make dresses from the given fabrics is 50 -/
theorem total_hours_is_fifty :
  total_hours_for_dresses 40 4 3 28 5 4 = 50 := by
  sorry

end total_hours_is_fifty_l2592_259218


namespace train_speed_l2592_259211

/-- Proves that a train with given length and time to cross a pole has a specific speed in km/h -/
theorem train_speed (length : Real) (time : Real) (speed_kmh : Real) : 
  length = 140 ∧ time = 7 → speed_kmh = 72 := by
  sorry

end train_speed_l2592_259211


namespace find_z_l2592_259267

theorem find_z (m : ℕ) (z : ℝ) 
  (h1 : ((1 ^ m) / (5 ^ m)) * ((1 ^ 16) / (z ^ 16)) = 1 / (2 * (10 ^ 31)))
  (h2 : m = 31) : z = 4 := by
  sorry

end find_z_l2592_259267


namespace correct_systematic_sample_l2592_259237

def systematicSample (n : ℕ) (k : ℕ) (start : ℕ) : List ℕ :=
  List.range k |>.map (fun i => start + i * (n / k))

theorem correct_systematic_sample :
  systematicSample 20 4 5 = [5, 10, 15, 20] := by
  sorry

end correct_systematic_sample_l2592_259237


namespace sequence_sum_l2592_259219

theorem sequence_sum (a : ℕ → ℕ) : 
  (a 1 = 1) → 
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2^n) → 
  a 10 = 1023 := by
sorry

end sequence_sum_l2592_259219


namespace negation_of_universal_proposition_l2592_259276

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x + 1 ≤ 0) :=
by sorry

end negation_of_universal_proposition_l2592_259276


namespace inverse_g_at_505_l2592_259261

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x^3 + 5

-- State the theorem
theorem inverse_g_at_505 : g⁻¹ 505 = 5 := by sorry

end inverse_g_at_505_l2592_259261


namespace fraction_equality_l2592_259298

theorem fraction_equality (x : ℝ) : (2 + x) / (4 + x) = (3 + x) / (7 + x) ↔ x = -1 := by
  sorry

end fraction_equality_l2592_259298


namespace no_real_roots_iff_k_less_than_negative_one_l2592_259227

theorem no_real_roots_iff_k_less_than_negative_one (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) ↔ k < -1 :=
by sorry

end no_real_roots_iff_k_less_than_negative_one_l2592_259227


namespace smallest_positive_solution_congruence_l2592_259213

theorem smallest_positive_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (4 * x) % 27 = 13 % 27 ∧
  ∀ (y : ℕ), y > 0 ∧ (4 * y) % 27 = 13 % 27 → x ≤ y :=
by sorry

end smallest_positive_solution_congruence_l2592_259213


namespace j_value_at_one_l2592_259279

theorem j_value_at_one (p q r : ℝ) : 
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (x^3 + p*x^2 + 2*x + 20 = 0) ∧
    (y^3 + p*y^2 + 2*y + 20 = 0) ∧
    (z^3 + p*z^2 + 2*z + 20 = 0)) →
  (∀ x : ℝ, x^3 + p*x^2 + 2*x + 20 = 0 → x^4 + 2*x^3 + q*x^2 + 150*x + r = 0) →
  1^4 + 2*1^3 + q*1^2 + 150*1 + r = -13755 :=
by sorry

end j_value_at_one_l2592_259279


namespace cone_height_l2592_259286

/-- Given a cone with slant height 2√2 cm and lateral surface area 4 cm², its height is 2 cm. -/
theorem cone_height (s : ℝ) (A : ℝ) (h : ℝ) :
  s = 2 * Real.sqrt 2 →
  A = 4 →
  A = π * s * (Real.sqrt (s^2 - h^2)) →
  h = 2 :=
by sorry

end cone_height_l2592_259286


namespace largest_angle_measure_l2592_259204

/-- A triangle PQR is obtuse and isosceles with angle P measuring 30 degrees. -/
structure ObtusePQR where
  /-- Triangle PQR is obtuse -/
  obtuse : Bool
  /-- Triangle PQR is isosceles -/
  isosceles : Bool
  /-- Angle P measures 30 degrees -/
  angle_p : ℝ
  /-- Angle P is 30 degrees -/
  h_angle_p : angle_p = 30

/-- The measure of the largest interior angle in triangle PQR is 120 degrees -/
theorem largest_angle_measure (t : ObtusePQR) : ℝ := by
  sorry

#check largest_angle_measure

end largest_angle_measure_l2592_259204


namespace greatest_consecutive_nonneg_integers_sum_120_l2592_259203

theorem greatest_consecutive_nonneg_integers_sum_120 :
  ∀ n : ℕ, (∃ a : ℕ, (n : ℕ) * (2 * a + n - 1) = 240) →
  n ≤ 16 :=
by sorry

end greatest_consecutive_nonneg_integers_sum_120_l2592_259203


namespace sum_of_roots_l2592_259209

theorem sum_of_roots (N : ℝ) : N * (N - 6) = -7 → ∃ N₁ N₂ : ℝ, N₁ * (N₁ - 6) = -7 ∧ N₂ * (N₂ - 6) = -7 ∧ N₁ + N₂ = 6 := by
  sorry

end sum_of_roots_l2592_259209


namespace birthday_money_calculation_l2592_259243

def money_spent : ℕ := 34
def money_left : ℕ := 33

theorem birthday_money_calculation :
  money_spent + money_left = 67 := by sorry

end birthday_money_calculation_l2592_259243


namespace angle_C_in_triangle_ABC_l2592_259206

theorem angle_C_in_triangle_ABC (A B C : ℝ) (h1 : 2 * Real.sin A + 3 * Real.cos B = 4) 
  (h2 : 3 * Real.sin B + 2 * Real.cos A = Real.sqrt 3) 
  (h3 : A + B + C = Real.pi) : C = Real.pi / 6 := by
  sorry

end angle_C_in_triangle_ABC_l2592_259206


namespace disjunction_true_l2592_259296

theorem disjunction_true : 
  (∃ α : ℝ, Real.cos (π - α) = Real.cos α) ∨ 
  (∀ x : ℝ, x^2 + 1 > 0) := by
sorry

end disjunction_true_l2592_259296


namespace initial_money_calculation_l2592_259297

/-- The amount of money Monica and Sheila's mother gave them initially -/
def initial_money : ℝ := 50

/-- The cost of toilet paper -/
def toilet_paper_cost : ℝ := 12

/-- The cost of groceries -/
def groceries_cost : ℝ := 2 * toilet_paper_cost

/-- The amount of money left after buying toilet paper and groceries -/
def money_left : ℝ := initial_money - (toilet_paper_cost + groceries_cost)

/-- The cost of one pair of boots -/
def boot_cost : ℝ := 3 * money_left

/-- The additional money needed to buy two pairs of boots -/
def additional_money : ℝ := 2 * 35

theorem initial_money_calculation :
  initial_money = toilet_paper_cost + groceries_cost + money_left ∧
  2 * boot_cost = 2 * 3 * money_left ∧
  2 * boot_cost = 2 * 3 * money_left ∧
  2 * boot_cost - money_left = additional_money := by
  sorry

end initial_money_calculation_l2592_259297


namespace digit_456_is_8_l2592_259225

/-- The decimal representation of 17/59 has a repeating cycle of 29 digits -/
def decimal_cycle : List Nat := [2, 8, 8, 1, 3, 5, 5, 9, 3, 2, 2, 0, 3, 3, 8, 9, 8, 3, 0, 5, 0, 8, 4, 7, 4, 5, 7, 6, 2, 7, 1, 1]

/-- The length of the repeating cycle in the decimal representation of 17/59 -/
def cycle_length : Nat := 29

/-- The 456th digit after the decimal point in the representation of 17/59 -/
def digit_456 : Nat := decimal_cycle[(456 % cycle_length) - 1]

theorem digit_456_is_8 : digit_456 = 8 := by sorry

end digit_456_is_8_l2592_259225


namespace fabric_needed_for_coats_l2592_259208

/-- 
Given that:
- 16 meters of fabric can make 4 men's coats and 2 children's coats
- 18 meters of fabric can make 2 men's coats and 6 children's coats

Prove that the fabric needed for one men's coat is 3 meters and for one children's coat is 2 meters.
-/
theorem fabric_needed_for_coats : 
  ∀ (m c : ℝ), 
  (4 * m + 2 * c = 16) → 
  (2 * m + 6 * c = 18) → 
  (m = 3 ∧ c = 2) :=
by sorry

end fabric_needed_for_coats_l2592_259208


namespace age_difference_l2592_259283

/-- Given information about Jacob and Michael's ages, prove their current age difference -/
theorem age_difference (jacob_current : ℕ) (michael_current : ℕ) : 
  (jacob_current + 4 = 13) → 
  (michael_current + 3 = 2 * (jacob_current + 3)) →
  (michael_current - jacob_current = 12) := by
sorry

end age_difference_l2592_259283


namespace pie_slices_left_l2592_259257

theorem pie_slices_left (total_slices : ℕ) (half_given : ℚ) (quarter_given : ℚ) : 
  total_slices = 8 → half_given = 1/2 → quarter_given = 1/4 → 
  total_slices - (half_given * total_slices + quarter_given * total_slices) = 2 := by
sorry

end pie_slices_left_l2592_259257


namespace one_sport_count_l2592_259284

/-- The number of members who play only one sport (badminton, tennis, or basketball) -/
def members_one_sport (total members badminton tennis basketball badminton_tennis badminton_basketball tennis_basketball all_three none : ℕ) : ℕ :=
  let badminton_only := badminton - badminton_tennis - badminton_basketball + all_three
  let tennis_only := tennis - badminton_tennis - tennis_basketball + all_three
  let basketball_only := basketball - badminton_basketball - tennis_basketball + all_three
  badminton_only + tennis_only + basketball_only

theorem one_sport_count :
  members_one_sport 150 65 80 60 20 15 25 10 12 = 115 := by
  sorry

end one_sport_count_l2592_259284


namespace building_height_l2592_259266

/-- Given a flagpole and a building casting shadows under similar conditions,
    calculate the height of the building. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagpole : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 50)
  : (flagpole_height / flagpole_shadow) * building_shadow = 20 := by
  sorry

end building_height_l2592_259266


namespace quadratic_root_difference_l2592_259271

theorem quadratic_root_difference (a b c d e : ℝ) :
  (2 * a^2 - 5 * a + 2 = 3 * a + 24) →
  ∃ x y : ℝ, (x ≠ y) ∧ 
             (2 * x^2 - 5 * x + 2 = 3 * x + 24) ∧ 
             (2 * y^2 - 5 * y + 2 = 3 * y + 24) ∧ 
             (abs (x - y) = 2 * Real.sqrt 15) :=
by sorry

end quadratic_root_difference_l2592_259271


namespace club_women_count_l2592_259215

/-- Proves the number of women in a club given certain conditions -/
theorem club_women_count (total : ℕ) (attendees : ℕ) (men : ℕ) (women : ℕ) :
  total = 30 →
  attendees = 18 →
  men + women = total →
  men + (women / 3) = attendees →
  women = 18 := by
  sorry

end club_women_count_l2592_259215


namespace fraction_equality_l2592_259224

theorem fraction_equality (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 1 / 3) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = 1 := by
  sorry

end fraction_equality_l2592_259224


namespace triangular_pyramid_theorem_l2592_259223

/-- A triangular pyramid with face areas S₁, S₂, S₃, S₄, distances H₁, H₂, H₃, H₄ 
    from any internal point to the faces, volume V, and constant k. -/
structure TriangularPyramid where
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ
  S₄ : ℝ
  H₁ : ℝ
  H₂ : ℝ
  H₃ : ℝ
  H₄ : ℝ
  V : ℝ
  k : ℝ
  h_positive : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0 ∧ H₁ > 0 ∧ H₂ > 0 ∧ H₃ > 0 ∧ H₄ > 0 ∧ V > 0 ∧ k > 0
  h_ratio : S₁ / 1 = S₂ / 2 ∧ S₂ / 2 = S₃ / 3 ∧ S₃ / 3 = S₄ / 4 ∧ S₄ / 4 = k

/-- The theorem to be proved -/
theorem triangular_pyramid_theorem (p : TriangularPyramid) : 
  1 * p.H₁ + 2 * p.H₂ + 3 * p.H₃ + 4 * p.H₄ = 3 * p.V / p.k := by
  sorry

end triangular_pyramid_theorem_l2592_259223


namespace square_area_equal_perimeter_l2592_259294

/-- Given an equilateral triangle with side length 30 cm and a square with the same perimeter,
    the area of the square is 506.25 cm^2. -/
theorem square_area_equal_perimeter (triangle_side : ℝ) (square_side : ℝ) : 
  triangle_side = 30 →
  3 * triangle_side = 4 * square_side →
  square_side^2 = 506.25 := by
  sorry

end square_area_equal_perimeter_l2592_259294


namespace discount_calculation_l2592_259217

/-- Given a 25% discount on a purchase where the final price paid is $120, prove that the discount amount is $40. -/
theorem discount_calculation (original_price : ℝ) : 
  (original_price * 0.75 = 120) → (original_price - 120 = 40) := by
  sorry

end discount_calculation_l2592_259217


namespace intersection_of_A_and_B_l2592_259264

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x < 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 1} := by
  sorry

end intersection_of_A_and_B_l2592_259264


namespace ratio_problem_l2592_259291

theorem ratio_problem (a b c d : ℚ) 
  (h1 : b / a = 3)
  (h2 : c / b = 4)
  (h3 : d = 2 * b - a) :
  (a + b + d) / (b + c + d) = 9 / 20 := by
  sorry

end ratio_problem_l2592_259291


namespace initial_data_points_l2592_259274

theorem initial_data_points (x : ℝ) : 
  (1.20 * x - 0.25 * (1.20 * x) = 180) → x = 200 := by
  sorry

end initial_data_points_l2592_259274


namespace probability_at_least_two_liking_chi_square_association_l2592_259295

-- Define the total number of students and their preferences
def total_students : ℕ := 200
def students_liking : ℕ := 140
def students_disliking : ℕ := 60

-- Define the gender-based preferences
def male_liking : ℕ := 60
def male_disliking : ℕ := 40
def female_liking : ℕ := 80
def female_disliking : ℕ := 20

-- Define the significance level
def alpha : ℝ := 0.005

-- Define the critical value for α = 0.005
def critical_value : ℝ := 7.879

-- Theorem 1: Probability of selecting at least 2 students who like employment
theorem probability_at_least_two_liking :
  (Nat.choose 3 2 * (students_liking / total_students)^2 * (students_disliking / total_students) +
   (students_liking / total_students)^3) = 98 / 125 := by sorry

-- Theorem 2: Chi-square test for association between intention and gender
theorem chi_square_association :
  let n := total_students
  let a := male_liking
  let b := male_disliking
  let c := female_liking
  let d := female_disliking
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d)) > critical_value := by sorry

end probability_at_least_two_liking_chi_square_association_l2592_259295


namespace total_course_hours_l2592_259292

/-- Represents the total hours spent on a course over the duration of 24 weeks --/
structure CourseHours where
  weekly : ℕ
  additional : ℕ

/-- Calculates the total hours for a course over 24 weeks --/
def totalHours (c : CourseHours) : ℕ := c.weekly * 24 + c.additional

/-- Data analytics course structure --/
def dataAnalyticsCourse : CourseHours :=
  { weekly := 14,  -- 10 hours class + 4 hours homework
    additional := 90 }  -- 48 hours lab sessions + 42 hours projects

/-- Programming course structure --/
def programmingCourse : CourseHours :=
  { weekly := 18,  -- 4 hours class + 8 hours lab + 6 hours assignments
    additional := 0 }

/-- Statistics course structure --/
def statisticsCourse : CourseHours :=
  { weekly := 11,  -- 6 hours class + 2 hours lab + 3 hours group projects
    additional := 45 }  -- 5 hours/week for 9 weeks for exam study

/-- The main theorem stating the total hours spent on all courses --/
theorem total_course_hours :
  totalHours dataAnalyticsCourse +
  totalHours programmingCourse +
  totalHours statisticsCourse = 1167 := by
  sorry

end total_course_hours_l2592_259292


namespace duke_of_york_men_percentage_l2592_259269

/-- The percentage of men remaining after two consecutive losses -/
theorem duke_of_york_men_percentage : 
  let initial_men : ℕ := 10000
  let first_loss_rate : ℚ := 1/10
  let second_loss_rate : ℚ := 3/20
  let remaining_men : ℚ := initial_men * (1 - first_loss_rate) * (1 - second_loss_rate)
  let percentage_remaining : ℚ := remaining_men / initial_men * 100
  percentage_remaining = 76.5 := by
  sorry

end duke_of_york_men_percentage_l2592_259269


namespace fraction_problem_l2592_259221

theorem fraction_problem (a b : ℤ) (ha : a > 0) (hb : b > 0) :
  (a : ℚ) / (b + 6) = 1 / 6 ∧ (a + 4 : ℚ) / b = 1 / 4 →
  (a : ℚ) / b = 11 / 60 := by
  sorry

end fraction_problem_l2592_259221


namespace shaded_area_theorem_l2592_259248

theorem shaded_area_theorem (r R : ℝ) (h1 : R > 0) (h2 : r > 0) : 
  (π * R^2 = 100 * π) → (r = R / 2) → 
  (π * R^2 / 2 + π * r^2 / 4 = 31.25 * π) := by
  sorry

end shaded_area_theorem_l2592_259248


namespace union_of_M_and_N_l2592_259241

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1, 2} := by
  sorry

end union_of_M_and_N_l2592_259241


namespace william_arrival_time_l2592_259254

/-- Represents time as hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hValid : minutes < 60

/-- Adds hours and minutes to a given time -/
def addTime (t : Time) (h : ℕ) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + h * 60 + m
  let newHours := totalMinutes / 60
  let newMinutes := totalMinutes % 60
  ⟨newHours % 24, newMinutes, by sorry⟩

/-- Calculates arrival time given departure time, total time on road, and total stop time -/
def calculateArrivalTime (departureTime : Time) (totalTimeOnRoad : ℕ) (totalStopTime : ℕ) : Time :=
  let actualDrivingTime := totalTimeOnRoad - (totalStopTime / 60)
  addTime departureTime actualDrivingTime 0

theorem william_arrival_time :
  let departureTime : Time := ⟨7, 0, by sorry⟩
  let totalTimeOnRoad : ℕ := 12
  let stopTimes : List ℕ := [25, 10, 25]
  let totalStopTime : ℕ := stopTimes.sum
  let arrivalTime := calculateArrivalTime departureTime totalTimeOnRoad totalStopTime
  arrivalTime = ⟨18, 0, by sorry⟩ := by sorry

end william_arrival_time_l2592_259254


namespace rent_calculation_l2592_259256

def problem (salary : ℕ) (milk groceries education petrol misc rent : ℕ) : Prop :=
  let savings := salary / 10
  let other_expenses := milk + groceries + education + petrol + misc
  salary = savings + rent + other_expenses ∧
  milk = 1500 ∧
  groceries = 4500 ∧
  education = 2500 ∧
  petrol = 2000 ∧
  misc = 2500 ∧
  savings = 2000

theorem rent_calculation :
  ∀ salary milk groceries education petrol misc rent,
    problem salary milk groceries education petrol misc rent →
    rent = 5000 := by
  sorry

end rent_calculation_l2592_259256


namespace task_completion_ways_l2592_259246

theorem task_completion_ways (m₁ m₂ : ℕ) : ∃ N : ℕ, N = m₁ + m₂ := by
  sorry

end task_completion_ways_l2592_259246


namespace positive_A_value_l2592_259239

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 7 = 200) : A = Real.sqrt 151 := by
  sorry

end positive_A_value_l2592_259239


namespace box_height_is_nine_l2592_259255

/-- A rectangular box containing spheres -/
structure SphereBox where
  height : ℝ
  large_sphere_radius : ℝ
  small_sphere_radius : ℝ
  large_sphere_count : ℕ
  small_sphere_count : ℕ

/-- The specific box described in the problem -/
def problem_box : SphereBox :=
  { height := 9,
    large_sphere_radius := 3,
    small_sphere_radius := 1.5,
    large_sphere_count := 1,
    small_sphere_count := 8 }

/-- Theorem stating that the height of the box must be 9 -/
theorem box_height_is_nine (box : SphereBox) :
  box.height = 9 ∧
  box.large_sphere_radius = 3 ∧
  box.small_sphere_radius = 1.5 ∧
  box.large_sphere_count = 1 ∧
  box.small_sphere_count = 8 →
  box = problem_box :=
sorry

end box_height_is_nine_l2592_259255


namespace pizza_dough_liquids_l2592_259287

/-- Pizza dough recipe calculation -/
theorem pizza_dough_liquids (milk_ratio : ℚ) (flour_ratio : ℚ) (flour_amount : ℚ) :
  milk_ratio = 75 →
  flour_ratio = 375 →
  flour_amount = 1125 →
  let portions := flour_amount / flour_ratio
  let milk_amount := portions * milk_ratio
  let water_amount := milk_amount / 2
  milk_amount + water_amount = 337.5 := by
  sorry

#check pizza_dough_liquids

end pizza_dough_liquids_l2592_259287


namespace tire_cost_l2592_259220

theorem tire_cost (total_cost : ℝ) (num_tires : ℕ) (cost_per_tire : ℝ) :
  total_cost = 4 →
  num_tires = 8 →
  cost_per_tire = total_cost / num_tires →
  cost_per_tire = 0.50 := by
sorry

end tire_cost_l2592_259220


namespace max_x_value_l2592_259253

theorem max_x_value (x y z : ℝ) 
  (sum_eq : x + y + z = 5) 
  (prod_sum_eq : x*y + x*z + y*z = 8) : 
  x ≤ 7/3 :=
sorry

end max_x_value_l2592_259253


namespace train_speed_problem_l2592_259229

theorem train_speed_problem (length1 length2 speed1 time : ℝ) 
  (h1 : length1 = 145)
  (h2 : length2 = 165)
  (h3 : speed1 = 60)
  (h4 : time = 8)
  (h5 : speed1 > 0) :
  let total_length := length1 + length2
  let relative_speed := total_length / time
  let speed2 := relative_speed - speed1
  speed2 = 79.5 := by
sorry

end train_speed_problem_l2592_259229


namespace first_bear_price_correct_l2592_259216

/-- The price of the first bear in a sequence of bear prices -/
def first_bear_price : ℚ := 57 / 2

/-- The number of bears purchased -/
def num_bears : ℕ := 101

/-- The discount applied to each bear after the first -/
def discount : ℚ := 1 / 2

/-- The total cost of all bears -/
def total_cost : ℚ := 354

/-- Theorem stating that the first bear price is correct given the conditions -/
theorem first_bear_price_correct :
  (num_bears : ℚ) / 2 * (2 * first_bear_price - (num_bears - 1) * discount) = total_cost :=
by sorry

end first_bear_price_correct_l2592_259216


namespace hexomino_min_containing_rectangle_area_l2592_259259

/-- A hexomino is a polyomino of 6 connected unit squares. -/
def Hexomino : Type := Unit  -- Placeholder definition

/-- The minimum area of a rectangle that contains a given hexomino. -/
def minContainingRectangleArea (h : Hexomino) : ℝ := sorry

/-- Theorem: The minimum area of any rectangle containing a hexomino is 21/2. -/
theorem hexomino_min_containing_rectangle_area (h : Hexomino) :
  minContainingRectangleArea h = 21 / 2 := by sorry

end hexomino_min_containing_rectangle_area_l2592_259259


namespace quadrilateral_side_length_l2592_259258

/-- Given a 9x16 rectangle that is cut into two congruent quadrilaterals
    which can be repositioned to form a square, the side length z of
    one quadrilateral is 12. -/
theorem quadrilateral_side_length (z : ℝ) : z = 12 :=
  let rectangle_area : ℝ := 9 * 16
  let square_side : ℝ := Real.sqrt rectangle_area
  sorry


end quadrilateral_side_length_l2592_259258


namespace inverse_one_implies_one_l2592_259242

theorem inverse_one_implies_one (a : ℝ) (h : a ≠ 0) : a⁻¹ = (-1)^0 → a = 1 := by
  sorry

end inverse_one_implies_one_l2592_259242


namespace inscribed_circle_l2592_259249

-- Define the triangle vertices
def A : ℝ × ℝ := (-2, 1)
def B : ℝ × ℝ := (2, 5)
def C : ℝ × ℝ := (5, 2)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 2

-- State the theorem
theorem inscribed_circle :
  ∃ (x y : ℝ), circle_equation x y ∧
  (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    ((x - A.1)^2 + (y - A.2)^2 = (t * (B.1 - A.1))^2 + (t * (B.2 - A.2))^2) ∧
    ((x - B.1)^2 + (y - B.2)^2 = (t * (C.1 - B.1))^2 + (t * (C.2 - B.2))^2) ∧
    ((x - C.1)^2 + (y - C.2)^2 = (t * (A.1 - C.1))^2 + (t * (A.2 - C.2))^2)) :=
sorry

end inscribed_circle_l2592_259249


namespace no_natural_divisible_by_49_l2592_259260

theorem no_natural_divisible_by_49 : ∀ n : ℕ, ¬(49 ∣ (n^2 + 5*n + 1)) := by sorry

end no_natural_divisible_by_49_l2592_259260


namespace parcel_weight_sum_l2592_259205

/-- Given three parcels with weights x, y, and z, prove that their total weight is 195 pounds
    if the sum of each pair of parcels weighs 112, 146, and 132 pounds respectively. -/
theorem parcel_weight_sum (x y z : ℝ) 
  (pair_xy : x + y = 112)
  (pair_yz : y + z = 146)
  (pair_zx : z + x = 132) :
  x + y + z = 195 := by
  sorry

end parcel_weight_sum_l2592_259205


namespace greatest_multiple_3_4_under_500_l2592_259251

theorem greatest_multiple_3_4_under_500 : ∃ n : ℕ, n = 492 ∧ 
  (∀ m : ℕ, m < 500 ∧ 3 ∣ m ∧ 4 ∣ m → m ≤ n) := by
  sorry

end greatest_multiple_3_4_under_500_l2592_259251


namespace min_value_expression_l2592_259277

theorem min_value_expression (n : ℕ+) : 
  (n : ℝ) / 3 + 27 / (n : ℝ) ≥ 6 ∧ 
  ((n : ℝ) / 3 + 27 / (n : ℝ) = 6 ↔ n = 9) :=
by sorry

end min_value_expression_l2592_259277


namespace function_inequality_l2592_259235

/-- For any differentiable function f on ℝ, if (x + 1)f'(x) ≥ 0 for all x in ℝ, 
    then f(0) + f(-2) ≥ 2f(-1) -/
theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h : ∀ x, (x + 1) * deriv f x ≥ 0) : 
  f 0 + f (-2) ≥ 2 * f (-1) := by
  sorry

end function_inequality_l2592_259235


namespace units_digit_sum_factorials_49_l2592_259230

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum_factorials_49 :
  units_digit (sum_factorials 49) = 3 := by sorry

end units_digit_sum_factorials_49_l2592_259230


namespace tees_per_member_l2592_259270

/-- The number of people in Bill's golfing group -/
def group_size : ℕ := 4

/-- The number of tees in a generic package -/
def generic_package_size : ℕ := 12

/-- The number of tees in an aero flight package -/
def aero_package_size : ℕ := 2

/-- The maximum number of generic packages Bill can buy -/
def max_generic_packages : ℕ := 2

/-- The number of aero flight packages Bill must purchase -/
def aero_packages : ℕ := 28

/-- Theorem stating that the number of golf tees per member is 20 -/
theorem tees_per_member :
  (max_generic_packages * generic_package_size + aero_packages * aero_package_size) / group_size = 20 := by
  sorry

end tees_per_member_l2592_259270


namespace quadratic_inequality_equivalence_l2592_259232

theorem quadratic_inequality_equivalence :
  ∀ x : ℝ, x * (2 * x + 3) < 2 ↔ x ∈ Set.Ioo (-2 : ℝ) (1/2 : ℝ) := by sorry

end quadratic_inequality_equivalence_l2592_259232


namespace parallel_transitivity_l2592_259226

-- Define a type for lines
def Line : Type := ℝ → ℝ → Prop

-- Define the parallel relation between lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem parallel_transitivity (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by sorry

end parallel_transitivity_l2592_259226


namespace exist_six_games_twelve_players_l2592_259244

structure Tournament where
  players : Finset ℕ
  games : Finset (ℕ × ℕ)
  player_in_game : ∀ p ∈ players, ∃ g ∈ games, p ∈ g.1 :: g.2 :: []

theorem exist_six_games_twelve_players (t : Tournament) 
  (h1 : t.players.card = 20)
  (h2 : t.games.card = 14) :
  ∃ (subset_games : Finset (ℕ × ℕ)) (subset_players : Finset ℕ),
    subset_games ⊆ t.games ∧
    subset_games.card = 6 ∧
    subset_players ⊆ t.players ∧
    subset_players.card = 12 ∧
    ∀ g ∈ subset_games, g.1 ∈ subset_players ∧ g.2 ∈ subset_players :=
sorry

end exist_six_games_twelve_players_l2592_259244


namespace solution_set_characterization_l2592_259299

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem solution_set_characterization (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_nonneg : ∀ x ≥ 0, f x = x^3 - 8) :
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} := by
  sorry

end solution_set_characterization_l2592_259299


namespace count_valid_words_l2592_259290

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 25

/-- The maximum word length -/
def max_word_length : ℕ := 5

/-- The number of total possible words without restrictions -/
def total_words : ℕ := 
  (alphabet_size ^ 1) + (alphabet_size ^ 2) + (alphabet_size ^ 3) + 
  (alphabet_size ^ 4) + (alphabet_size ^ 5)

/-- The number of words with fewer than two 'A's -/
def words_with_less_than_two_a : ℕ := 
  ((alphabet_size - 1) ^ 2) + (2 * (alphabet_size - 1)) + 
  ((alphabet_size - 1) ^ 3) + (3 * (alphabet_size - 1) ^ 2) + 
  ((alphabet_size - 1) ^ 4) + (4 * (alphabet_size - 1) ^ 3) + 
  ((alphabet_size - 1) ^ 5) + (5 * (alphabet_size - 1) ^ 4)

/-- The number of valid words in the language -/
def valid_words : ℕ := total_words - words_with_less_than_two_a

theorem count_valid_words : 
  valid_words = (25^1 + 25^2 + 25^3 + 25^4 + 25^5) - 
                (24^2 + 2 * 24 + 24^3 + 3 * 24^2 + 24^4 + 4 * 24^3 + 24^5 + 5 * 24^4) := by
  sorry

end count_valid_words_l2592_259290


namespace rectangle_enclosure_l2592_259202

def rectangle_largest_side (length width : ℝ) : Prop :=
  let perimeter := 2 * (length + width)
  let area := length * width
  perimeter = 240 ∧ 
  area = 12 * perimeter ∧ 
  length ≥ width ∧
  length = 72

theorem rectangle_enclosure :
  ∃ (length width : ℝ), rectangle_largest_side length width :=
sorry

end rectangle_enclosure_l2592_259202


namespace path_count_equals_combination_l2592_259262

/-- The width of the grid -/
def grid_width : ℕ := 6

/-- The height of the grid -/
def grid_height : ℕ := 5

/-- The total number of steps required to reach from A to B -/
def total_steps : ℕ := grid_width + grid_height - 2

/-- The number of vertical steps required -/
def vertical_steps : ℕ := grid_height - 1

theorem path_count_equals_combination : 
  (Nat.choose total_steps vertical_steps) = 126 := by sorry

end path_count_equals_combination_l2592_259262


namespace root_product_equals_two_l2592_259272

theorem root_product_equals_two : 
  ∃ (x₁ x₂ x₃ : ℝ), 
    (Real.sqrt 4050 * x₁^3 - 8101 * x₁^2 + 4 = 0) ∧
    (Real.sqrt 4050 * x₂^3 - 8101 * x₂^2 + 4 = 0) ∧
    (Real.sqrt 4050 * x₃^3 - 8101 * x₃^2 + 4 = 0) ∧
    (x₁ < x₂) ∧ (x₂ < x₃) ∧
    (x₂ * (x₁ + x₃) = 2) := by
  sorry

end root_product_equals_two_l2592_259272


namespace cube_shortest_distances_l2592_259207

/-- A cube with edge length 1 -/
structure Cube where
  edge_length : ℝ
  edge_length_pos : edge_length = 1

/-- A point on the surface of the cube -/
structure CubePoint (c : Cube) where
  x : ℝ
  y : ℝ
  z : ℝ
  on_surface : (x = 0 ∨ x = c.edge_length) ∨ 
               (y = 0 ∨ y = c.edge_length) ∨ 
               (z = 0 ∨ z = c.edge_length)

/-- The shortest distance between two points on the cube's surface -/
def shortest_distance (c : Cube) (p1 p2 : CubePoint c) : ℝ :=
  sorry

/-- Two adjacent vertices of the cube -/
def adjacent_vertices (c : Cube) : CubePoint c × CubePoint c :=
  ⟨⟨0, 0, 0, by simp⟩, ⟨c.edge_length, 0, 0, by simp⟩⟩

/-- Two points on adjacent edges, each 1 unit from their common vertex -/
def adjacent_edge_points (c : Cube) : CubePoint c × CubePoint c :=
  ⟨⟨c.edge_length, 0, c.edge_length, by simp⟩, ⟨c.edge_length, c.edge_length, 0, by simp⟩⟩

/-- Two non-adjacent vertices of the cube -/
def non_adjacent_vertices (c : Cube) : CubePoint c × CubePoint c :=
  ⟨⟨0, 0, 0, by simp⟩, ⟨c.edge_length, c.edge_length, 0, by simp⟩⟩

theorem cube_shortest_distances (c : Cube) :
  let (v1, v2) := adjacent_vertices c
  let (p1, p2) := adjacent_edge_points c
  let (u1, u2) := non_adjacent_vertices c
  shortest_distance c v1 v2 = 1 ∧
  shortest_distance c p1 p2 = 2 ∧
  shortest_distance c u1 u2 = Real.sqrt 5 := by
  sorry

end cube_shortest_distances_l2592_259207


namespace sum_of_symmetric_roots_l2592_259245

/-- A function f: ℝ → ℝ that satisfies f(1-x) = f(1+x) for all real x -/
def SymmetricAboutOne (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (1 - x) = f (1 + x)

/-- The theorem stating that if f is symmetric about 1 and has exactly 2009 real roots,
    then the sum of these roots is 2009 -/
theorem sum_of_symmetric_roots
  (f : ℝ → ℝ)
  (h_sym : SymmetricAboutOne f)
  (h_roots : ∃! (s : Finset ℝ), s.card = 2009 ∧ ∀ x ∈ s, f x = 0) :
  ∃ (s : Finset ℝ), s.card = 2009 ∧ (∀ x ∈ s, f x = 0) ∧ (s.sum id = 2009) :=
sorry

end sum_of_symmetric_roots_l2592_259245


namespace inequality_system_solution_l2592_259273

theorem inequality_system_solution : 
  {x : ℝ | (8 * x - 3 ≤ 13) ∧ ((x - 1) / 3 - 2 < x - 1)} = {x : ℝ | -2 < x ∧ x ≤ 2} := by
  sorry

end inequality_system_solution_l2592_259273


namespace remaining_lives_total_l2592_259200

def game_scenario (initial_players : ℕ) (first_quitters : ℕ) (second_quitters : ℕ) (lives_per_player : ℕ) : ℕ :=
  (initial_players - first_quitters - second_quitters) * lives_per_player

theorem remaining_lives_total :
  game_scenario 15 5 4 7 = 42 := by
  sorry

end remaining_lives_total_l2592_259200


namespace equation_solutions_and_first_m_first_m_above_1959_l2592_259281

theorem equation_solutions_and_first_m (m n : ℕ+) :
  (8 * m - 7 = n^2) ↔ 
  (∃ s : ℕ, m = 1 + s * (s + 1) / 2 ∧ n = 2 * s + 1) :=
sorry

theorem first_m_above_1959 :
  (∃ m₀ : ℕ+, m₀ > 1959 ∧ 
   (∀ m : ℕ+, m > 1959 ∧ (∃ n : ℕ+, 8 * m - 7 = n^2) → m ≥ m₀) ∧
   m₀ = 2017) :=
sorry

end equation_solutions_and_first_m_first_m_above_1959_l2592_259281


namespace widest_strip_width_l2592_259236

theorem widest_strip_width (bolt_width_1 bolt_width_2 : ℕ) 
  (h1 : bolt_width_1 = 45) 
  (h2 : bolt_width_2 = 60) : 
  Nat.gcd bolt_width_1 bolt_width_2 = 15 := by
  sorry

end widest_strip_width_l2592_259236


namespace smallest_cookie_boxes_l2592_259265

theorem smallest_cookie_boxes : ∃ (n : ℕ), n > 0 ∧ (15 * n - 1) % 11 = 0 ∧ ∀ (m : ℕ), m > 0 → (15 * m - 1) % 11 = 0 → n ≤ m := by
  sorry

end smallest_cookie_boxes_l2592_259265


namespace water_remaining_l2592_259263

theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 5/4 → remaining = initial - used → remaining = 7/4 := by
  sorry

end water_remaining_l2592_259263


namespace solve_system_l2592_259240

theorem solve_system (x y : ℝ) 
  (eq1 : 3 * x - 2 * y = 8) 
  (eq2 : 2 * x + 3 * y = 1) : 
  x = 2 := by
sorry

end solve_system_l2592_259240


namespace square_area_percentage_l2592_259280

/-- Given a rectangle enclosing a square, this theorem proves the percentage
    of the rectangle's area occupied by the square. -/
theorem square_area_percentage (s : ℝ) (h1 : s > 0) : 
  let w := 3 * s  -- width of rectangle
  let l := 3 * w / 2  -- length of rectangle
  let square_area := s^2
  let rectangle_area := l * w
  (square_area / rectangle_area) * 100 = 200 / 27 := by sorry

end square_area_percentage_l2592_259280


namespace unequal_gender_probability_l2592_259212

theorem unequal_gender_probability : 
  let n : ℕ := 12  -- Total number of grandchildren
  let p : ℚ := 1/2 -- Probability of each child being male (or female)
  -- Probability of unequal number of grandsons and granddaughters
  (1 : ℚ) - (n.choose (n/2) : ℚ) / 2^n = 793/1024 :=
by sorry

end unequal_gender_probability_l2592_259212


namespace actual_distance_traveled_prove_actual_distance_l2592_259289

/-- The actual distance traveled by a person, given two different walking speeds and a distance difference. -/
theorem actual_distance_traveled (speed1 speed2 distance_diff : ℝ) (h1 : speed1 > 0) (h2 : speed2 > 0) 
  (h3 : speed2 > speed1) (h4 : distance_diff > 0) : ℝ :=
  let time := distance_diff / (speed2 - speed1)
  let actual_distance := speed1 * time
  actual_distance

/-- Proves that the actual distance traveled is 20 km under the given conditions. -/
theorem prove_actual_distance : 
  actual_distance_traveled 10 20 20 (by norm_num) (by norm_num) (by norm_num) (by norm_num) = 20 := by
  sorry

end actual_distance_traveled_prove_actual_distance_l2592_259289


namespace pool_perimeter_is_20_l2592_259214

/-- Represents the dimensions and constraints of a rectangular pool in a garden --/
structure PoolInGarden where
  garden_length : ℝ
  garden_width : ℝ
  pool_area : ℝ
  walkway_width : ℝ
  pool_length : ℝ := garden_length - 2 * walkway_width
  pool_width : ℝ := garden_width - 2 * walkway_width

/-- Calculates the perimeter of the pool --/
def pool_perimeter (p : PoolInGarden) : ℝ :=
  2 * (p.pool_length + p.pool_width)

/-- Theorem: The perimeter of the pool is 20 meters --/
theorem pool_perimeter_is_20 (p : PoolInGarden) 
    (h1 : p.garden_length = 8)
    (h2 : p.garden_width = 6)
    (h3 : p.pool_area = 24)
    (h4 : p.pool_length * p.pool_width = p.pool_area) : 
  pool_perimeter p = 20 := by
  sorry

#check pool_perimeter_is_20

end pool_perimeter_is_20_l2592_259214
