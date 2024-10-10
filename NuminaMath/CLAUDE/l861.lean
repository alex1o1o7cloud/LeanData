import Mathlib

namespace complex_fraction_equality_l861_86165

theorem complex_fraction_equality : (Complex.I : ℂ) ^ 2 = -1 → (2 + 2 * Complex.I) / (1 - Complex.I) = 2 * Complex.I := by
  sorry

end complex_fraction_equality_l861_86165


namespace xyz_expression_bounds_l861_86126

theorem xyz_expression_bounds (x y z : ℝ) 
  (non_neg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (sum_one : x + y + z = 1) : 
  0 ≤ x*y + y*z + z*x - 3*x*y*z ∧ x*y + y*z + z*x - 3*x*y*z ≤ 1/4 := by
sorry

end xyz_expression_bounds_l861_86126


namespace total_carrots_is_40_l861_86157

/-- The number of carrots grown by Joan -/
def joan_carrots : ℕ := 29

/-- The number of carrots grown by Jessica -/
def jessica_carrots : ℕ := 11

/-- The total number of carrots grown by Joan and Jessica -/
def total_carrots : ℕ := joan_carrots + jessica_carrots

/-- Theorem stating that the total number of carrots grown is 40 -/
theorem total_carrots_is_40 : total_carrots = 40 := by
  sorry

end total_carrots_is_40_l861_86157


namespace perfect_square_trinomial_l861_86196

theorem perfect_square_trinomial (m : ℝ) :
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + 2*(m-1)*x + 16 = (a*x + b)^2) →
  m = 5 ∨ m = -3 := by
sorry

end perfect_square_trinomial_l861_86196


namespace translated_line_point_l861_86164

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically by a given amount -/
def translateLine (l : Line) (amount : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + amount }

/-- Checks if a point lies on a line -/
def pointOnLine (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem translated_line_point (m : ℝ) : 
  let original_line : Line := { slope := 1, intercept := 0 }
  let translated_line := translateLine original_line 3
  pointOnLine translated_line 2 m → m = 5 := by
  sorry

end translated_line_point_l861_86164


namespace no_prime_roots_for_quadratic_l861_86153

theorem no_prime_roots_for_quadratic : ¬∃ (k : ℤ), ∃ (p q : ℕ), 
  Prime p ∧ Prime q ∧ p ≠ q ∧ 
  (p : ℤ) + q = 57 ∧ (p : ℤ) * q = k ∧
  ∀ (x : ℤ), x^2 - 57*x + k = 0 ↔ x = p ∨ x = q := by
  sorry

end no_prime_roots_for_quadratic_l861_86153


namespace wedding_guests_l861_86133

theorem wedding_guests (total : ℕ) 
  (h1 : (83 : ℚ) / 100 * total + (9 : ℚ) / 100 * total + 16 = total) : 
  total = 200 := by
  sorry

end wedding_guests_l861_86133


namespace f_sum_value_l861_86177

noncomputable def f (x : ℝ) : ℝ := (1 / x) * Real.cos x

theorem f_sum_value : f Real.pi + (deriv f) (Real.pi / 2) = -3 / Real.pi := by
  sorry

end f_sum_value_l861_86177


namespace f_derivative_at_zero_l861_86170

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1)^4

-- State the theorem
theorem f_derivative_at_zero : 
  (deriv f) 0 = 4 := by sorry

end f_derivative_at_zero_l861_86170


namespace min_value_of_expression_l861_86169

theorem min_value_of_expression (x y : ℝ) (h : x^2 + y^2 - 4*x + 6*y + 12 = 0) :
  ∃ (min : ℝ), min = 5 - Real.sqrt 5 ∧ ∀ (x y : ℝ), x^2 + y^2 - 4*x + 6*y + 12 = 0 → |2*x - y - 2| ≥ min :=
by sorry

end min_value_of_expression_l861_86169


namespace horseshoe_division_l861_86108

/-- Represents a paper horseshoe with holes -/
structure Horseshoe where
  holes : ℕ

/-- Represents a cut on the horseshoe -/
inductive Cut
| straight : Cut

/-- Represents the state of the horseshoe after cuts -/
structure HorseshoeState where
  pieces : ℕ
  holesPerPiece : ℕ

/-- Function to apply a cut to the horseshoe -/
def applyCut (h : Horseshoe) (c : Cut) (s : HorseshoeState) : HorseshoeState :=
  sorry

/-- Function to rearrange pieces -/
def rearrange (s : HorseshoeState) : HorseshoeState :=
  sorry

/-- Theorem stating that a horseshoe can be divided into n parts with n holes using two straight cuts -/
theorem horseshoe_division (h : Horseshoe) :
  ∃ (c1 c2 : Cut), ∃ (s1 s2 s3 : HorseshoeState),
    s1 = applyCut h c1 {pieces := 1, holesPerPiece := h.holes} ∧
    s2 = rearrange s1 ∧
    s3 = applyCut h c2 s2 ∧
    s3.pieces = h.holes ∧
    s3.holesPerPiece = 1 :=
  sorry

end horseshoe_division_l861_86108


namespace reciprocal_sum_one_l861_86159

theorem reciprocal_sum_one (x y z : ℕ+) (h_sum : (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = 1) 
  (h_order : x ≤ y ∧ y ≤ z) : 
  (x = 2 ∧ y = 4 ∧ z = 4) ∨ (x = 2 ∧ y = 3 ∧ z = 6) ∨ (x = 3 ∧ y = 3 ∧ z = 3) := by
  sorry

#check reciprocal_sum_one

end reciprocal_sum_one_l861_86159


namespace cheetah_speed_calculation_l861_86107

/-- The top speed of a cheetah in miles per hour -/
def cheetah_speed : ℝ := 60

/-- The top speed of a gazelle in miles per hour -/
def gazelle_speed : ℝ := 40

/-- Conversion factor from miles per hour to feet per second -/
def mph_to_fps : ℝ := 1.5

/-- Time taken for the cheetah to catch up to the gazelle in seconds -/
def catch_up_time : ℝ := 7

/-- Initial distance between the cheetah and the gazelle in feet -/
def initial_distance : ℝ := 210

theorem cheetah_speed_calculation :
  cheetah_speed * mph_to_fps - gazelle_speed * mph_to_fps = initial_distance / catch_up_time :=
by sorry

end cheetah_speed_calculation_l861_86107


namespace contrapositive_real_roots_l861_86152

theorem contrapositive_real_roots (m : ℝ) :
  (¬(∃ x : ℝ, x^2 = m) → m < 0) ↔
  (m ≥ 0 → ∃ x : ℝ, x^2 = m) :=
by sorry

end contrapositive_real_roots_l861_86152


namespace recurrence_closed_form_l861_86180

def recurrence_sequence (a : ℕ → ℝ) : Prop :=
  (a 0 = 3) ∧ (a 1 = 5) ∧ (∀ n : ℕ, n ≥ 2 → a n = 4 * a (n - 1) - 3 * a (n - 2))

theorem recurrence_closed_form (a : ℕ → ℝ) (h : recurrence_sequence a) :
  ∀ n : ℕ, a n = 3^n + 2 :=
by
  sorry

end recurrence_closed_form_l861_86180


namespace parabola_circle_area_ratio_l861_86158

/-- The ratio of areas S1 to S2 for a parabola and tangent circle -/
theorem parabola_circle_area_ratio 
  (d : ℝ) 
  (hd : d > 0) : 
  let K : ℝ → ℝ := fun x ↦ (1/d) * x^2
  let P : ℝ × ℝ := (d, d)
  let Q : ℝ × ℝ := (0, d)
  let S1 : ℝ := ∫ x in (0)..(d), (d - K x)
  let S2 : ℝ := ∫ x in (0)..(d), (d - K x)
  S1 / S2 = 1 := by
  sorry

end parabola_circle_area_ratio_l861_86158


namespace regression_slope_l861_86193

-- Define the linear function
def f (x : ℝ) : ℝ := 2 - 3 * x

-- Theorem statement
theorem regression_slope (x : ℝ) :
  f (x + 1) = f x - 3 := by
  sorry

end regression_slope_l861_86193


namespace distance_between_runners_l861_86102

/-- The distance between two runners at the end of a 1 km race -/
theorem distance_between_runners (H J : ℝ) (t : ℝ) 
  (h_distance : 1000 = H * t) 
  (j_distance : 152 = J * t) : 
  1000 - 152 = 848 := by sorry

end distance_between_runners_l861_86102


namespace roots_modulus_one_preserved_l861_86128

theorem roots_modulus_one_preserved (a b c : ℂ) :
  (∃ α β γ : ℂ, (α^3 + a*α^2 + b*α + c = 0) ∧ 
                (β^3 + a*β^2 + b*β + c = 0) ∧ 
                (γ^3 + a*γ^2 + b*γ + c = 0) ∧
                (Complex.abs α = 1) ∧ (Complex.abs β = 1) ∧ (Complex.abs γ = 1)) →
  (∃ x y z : ℂ, (x^3 + Complex.abs a*x^2 + Complex.abs b*x + Complex.abs c = 0) ∧ 
                (y^3 + Complex.abs a*y^2 + Complex.abs b*y + Complex.abs c = 0) ∧ 
                (z^3 + Complex.abs a*z^2 + Complex.abs b*z + Complex.abs c = 0) ∧
                (Complex.abs x = 1) ∧ (Complex.abs y = 1) ∧ (Complex.abs z = 1)) :=
by sorry

end roots_modulus_one_preserved_l861_86128


namespace stratified_sampling_total_size_l861_86167

theorem stratified_sampling_total_size 
  (district1_ratio : ℚ) 
  (district2_ratio : ℚ) 
  (district3_ratio : ℚ) 
  (largest_district_sample : ℕ) : 
  district1_ratio + district2_ratio + district3_ratio = 1 →
  district3_ratio > district1_ratio →
  district3_ratio > district2_ratio →
  district3_ratio = 1/2 →
  largest_district_sample = 60 →
  2 * largest_district_sample = 120 :=
by
  sorry

#check stratified_sampling_total_size

end stratified_sampling_total_size_l861_86167


namespace probability_club_heart_king_l861_86131

theorem probability_club_heart_king (total_cards : ℕ) (clubs : ℕ) (hearts : ℕ) (kings : ℕ) :
  total_cards = 52 →
  clubs = 13 →
  hearts = 13 →
  kings = 4 →
  (clubs / total_cards) * (hearts / (total_cards - 1)) * (kings / (total_cards - 2)) = 13 / 2550 := by
  sorry

end probability_club_heart_king_l861_86131


namespace cube_root_of_product_with_nested_roots_l861_86120

theorem cube_root_of_product_with_nested_roots (N : ℝ) (h : N > 1) :
  (N * (N * N^(1/3))^(1/2))^(1/3) = N^(5/9) := by
  sorry

end cube_root_of_product_with_nested_roots_l861_86120


namespace greatest_four_digit_divisible_by_3_and_4_l861_86105

theorem greatest_four_digit_divisible_by_3_and_4 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 3 ∣ n ∧ 4 ∣ n → n ≤ 9996 :=
by sorry

end greatest_four_digit_divisible_by_3_and_4_l861_86105


namespace largest_expression_l861_86137

theorem largest_expression : 
  let e1 := 992 * 999 + 999
  let e2 := 993 * 998 + 998
  let e3 := 994 * 997 + 997
  let e4 := 995 * 996 + 996
  (e4 > e1) ∧ (e4 > e2) ∧ (e4 > e3) := by
sorry

end largest_expression_l861_86137


namespace product_of_fractions_l861_86162

theorem product_of_fractions : 
  (4 : ℚ) / 5 * 9 / 6 * 12 / 4 * 20 / 15 * 14 / 21 * 35 / 28 * 48 / 32 * 24 / 16 = 54 := by
  sorry

end product_of_fractions_l861_86162


namespace arithmetic_sequence_condition_l861_86175

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_condition (a : ℕ → ℝ) (m p q : ℕ) 
  (h_arithmetic : is_arithmetic_sequence a) (h_positive : m > 0 ∧ p > 0 ∧ q > 0) :
  (p + q = 2 * m → a p + a q = 2 * a m) ∧
  ∃ b : ℕ → ℝ, is_arithmetic_sequence b ∧ ∃ m' p' q' : ℕ, 
    m' > 0 ∧ p' > 0 ∧ q' > 0 ∧ b p' + b q' = 2 * b m' ∧ p' + q' ≠ 2 * m' :=
by sorry

end arithmetic_sequence_condition_l861_86175


namespace circle_equation_l861_86106

theorem circle_equation (x y : ℝ) :
  (x + 2)^2 + (y - 2)^2 = 25 ↔ 
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (-2, 2) ∧ 
    radius = 5 ∧ 
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_equation_l861_86106


namespace sweets_ratio_l861_86183

/-- Proves that the ratio of sweets received by the youngest child to the eldest child is 1:2 --/
theorem sweets_ratio (total : ℕ) (eldest : ℕ) (second : ℕ) : 
  total = 27 →
  eldest = 8 →
  second = 6 →
  (total - (total / 3) - eldest - second) * 2 = eldest := by
  sorry

end sweets_ratio_l861_86183


namespace distinct_colorings_count_l861_86101

/-- Represents the symmetries of a regular octagon -/
inductive OctagonSymmetry
| Identity
| Reflection (n : Fin 8)
| Rotation (n : Fin 4)

/-- Represents a coloring of 8 disks -/
def Coloring := Fin 8 → Fin 3

/-- The number of disks -/
def n : ℕ := 8

/-- The number of colors -/
def k : ℕ := 3

/-- The number of each color -/
def colorCounts : Fin 3 → ℕ
| 0 => 4  -- blue
| 1 => 3  -- red
| 2 => 1  -- green
| _ => 0  -- unreachable

/-- The set of all possible colorings -/
def allColorings : Finset Coloring := sorry

/-- Whether a coloring is fixed by a given symmetry -/
def isFixed (c : Coloring) (s : OctagonSymmetry) : Prop := sorry

/-- The number of colorings fixed by each symmetry -/
def fixedColorings (s : OctagonSymmetry) : ℕ := sorry

/-- The set of all symmetries -/
def symmetries : Finset OctagonSymmetry := sorry

/-- The main theorem: the number of distinct colorings is 21 -/
theorem distinct_colorings_count :
  (Finset.sum symmetries fixedColorings) / Finset.card symmetries = 21 := sorry

end distinct_colorings_count_l861_86101


namespace number_percentage_equality_l861_86194

theorem number_percentage_equality (x : ℝ) : 
  (25 / 100) * x = (20 / 100) * 30 → x = 24 := by
sorry

end number_percentage_equality_l861_86194


namespace roots_sum_logarithmic_equation_l861_86109

theorem roots_sum_logarithmic_equation (m : ℝ) :
  ∃ x₁ x₂ : ℝ, (Real.log (|x₁ - 2|) = m ∧ Real.log (|x₂ - 2|) = m) → x₁ + x₂ = 4 := by
  sorry

end roots_sum_logarithmic_equation_l861_86109


namespace sector_area_l861_86160

/-- Given a circular sector with circumference 6cm and central angle 1 radian, 
    prove that its area is 2cm². -/
theorem sector_area (circumference : ℝ) (central_angle : ℝ) (area : ℝ) : 
  circumference = 6 → central_angle = 1 → area = 2 := by sorry

end sector_area_l861_86160


namespace perpendicular_lines_angle_relation_l861_86130

-- Define a dihedral angle
structure DihedralAngle where
  plane_angle : ℝ
  -- Add other necessary properties

-- Define a point inside a dihedral angle
structure PointInDihedralAngle where
  dihedral : DihedralAngle
  -- Add other necessary properties

-- Define the angle formed by perpendicular lines
def perpendicularLinesAngle (p : PointInDihedralAngle) : ℝ := sorry

-- Define the relationship between angles
def isEqualOrComplementary (a b : ℝ) : Prop :=
  a = b ∨ a + b = Real.pi / 2

-- Theorem statement
theorem perpendicular_lines_angle_relation (p : PointInDihedralAngle) :
  isEqualOrComplementary (perpendicularLinesAngle p) p.dihedral.plane_angle := by
  sorry

end perpendicular_lines_angle_relation_l861_86130


namespace equation_solution_l861_86144

theorem equation_solution (a : ℝ) : 
  (∀ x, 2*(x+1) = 3*(x-1) ↔ x = a+2) →
  (∃! x, 2*(2*(x+3) - 3*(x-a)) = 3*a ∧ x = 10) := by
sorry

end equation_solution_l861_86144


namespace profit_share_difference_example_l861_86171

/-- Given a total profit and a ratio of profit division between two parties,
    calculate the difference between their profit shares. -/
def profit_share_difference (total_profit : ℚ) (ratio_x : ℚ) (ratio_y : ℚ) : ℚ :=
  let total_ratio := ratio_x + ratio_y
  let share_x := (ratio_x / total_ratio) * total_profit
  let share_y := (ratio_y / total_ratio) * total_profit
  share_x - share_y

/-- Theorem stating that for a total profit of 800 and a profit division ratio of 1/2 : 1/3,
    the difference between the profit shares is 160. -/
theorem profit_share_difference_example :
  profit_share_difference 800 (1/2) (1/3) = 160 := by
  sorry


end profit_share_difference_example_l861_86171


namespace sum_of_penultimate_terms_l861_86110

def arithmetic_sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ d : ℕ, ∀ i : ℕ, i < n → a (i + 1) = a i + d

theorem sum_of_penultimate_terms (a : ℕ → ℕ) :
  arithmetic_sequence a 7 →
  a 0 = 3 →
  a 6 = 33 →
  a 4 + a 5 = 48 := by
sorry

end sum_of_penultimate_terms_l861_86110


namespace function_properties_l861_86150

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (ω * x + φ)

theorem function_properties 
  (ω φ : ℝ) 
  (hω : ω > 0) 
  (hφ : 0 < φ ∧ φ < Real.pi / 2) 
  (hperiod : ∀ x, f ω φ (x + Real.pi) = f ω φ x)
  (hsymmetry : ∀ x, f ω φ (-Real.pi/24 + x) = f ω φ (-Real.pi/24 - x))
  (A B C : ℝ)
  (ha : ∀ a b c : ℝ, a = 3 → b + c = 6 → a^2 = b^2 + c^2 - 2*b*c*Real.cos A)
  (hf : f ω φ (-A/2) = Real.sqrt 2) :
  ω = 2 ∧ φ = Real.pi/12 ∧ ∃ (b c : ℝ), b = 3 ∧ c = 3 := by
  sorry

end function_properties_l861_86150


namespace store_inventory_count_l861_86155

theorem store_inventory_count : 
  ∀ (original_price : ℝ) (discount_rate : ℝ) (sold_percentage : ℝ) 
    (debt : ℝ) (remaining : ℝ),
  original_price = 50 →
  discount_rate = 0.8 →
  sold_percentage = 0.9 →
  debt = 15000 →
  remaining = 3000 →
  (((1 - discount_rate) * original_price * sold_percentage) * 
    (debt + remaining) / ((1 - discount_rate) * original_price * sold_percentage)) = 2000 :=
by
  sorry

end store_inventory_count_l861_86155


namespace correct_operation_l861_86141

theorem correct_operation (a b : ℝ) : 3 * a^2 * b - 3 * b * a^2 = 0 := by
  sorry

end correct_operation_l861_86141


namespace sin_cos_inequality_l861_86168

theorem sin_cos_inequality (x : ℝ) : 
  0 ≤ x ∧ x ≤ 2 * π ∧ Real.sin (x - π / 6) > Real.cos x → 
  π / 3 < x ∧ x < 4 * π / 3 := by
sorry

end sin_cos_inequality_l861_86168


namespace app_cost_is_four_l861_86199

/-- The average cost of an app given the total budget, remaining amount, and number of apps. -/
def average_app_cost (total_budget : ℚ) (remaining : ℚ) (num_apps : ℕ) : ℚ :=
  (total_budget - remaining) / num_apps

/-- Theorem stating that the average cost of an app is $4 given the problem conditions. -/
theorem app_cost_is_four :
  let total_budget : ℚ := 66
  let remaining : ℚ := 6
  let num_apps : ℕ := 15
  average_app_cost total_budget remaining num_apps = 4 := by
  sorry

end app_cost_is_four_l861_86199


namespace smaller_root_of_equation_l861_86116

theorem smaller_root_of_equation (x : ℝ) : 
  (x - 5/8) * (x - 5/8) + (x - 5/8) * (x - 2/3) = 0 → 
  (∃ y : ℝ, (y - 5/8) * (y - 5/8) + (y - 5/8) * (y - 2/3) = 0 ∧ y ≤ x) → 
  x = 29/48 := by
sorry

end smaller_root_of_equation_l861_86116


namespace shaded_area_square_configuration_l861_86117

/-- The area of the shaded region in a geometric configuration where a 4-inch square adjoins a 12-inch square -/
theorem shaded_area_square_configuration : 
  -- Large square side length
  ∀ (large_side : ℝ) 
  -- Small square side length
  (small_side : ℝ),
  -- Conditions
  large_side = 12 →
  small_side = 4 →
  -- The shaded area is the difference between the small square's area and the area of a triangle
  let shaded_area := small_side^2 - (1/2 * (3/4 * small_side) * small_side)
  -- Theorem statement
  shaded_area = 10 := by
sorry

end shaded_area_square_configuration_l861_86117


namespace equality_of_ordered_triples_l861_86121

theorem equality_of_ordered_triples
  (a b c x y z : ℝ)
  (positive_a : 0 < a) (positive_b : 0 < b) (positive_c : 0 < c)
  (positive_x : 0 < x) (positive_y : 0 < y) (positive_z : 0 < z)
  (sum_equality : x + y + z = a + b + c)
  (product_equality : x * y * z = a * b * c)
  (ordering_xyz : a ≤ x ∧ x < y ∧ y < z ∧ z ≤ c)
  (ordering_abc : a < b ∧ b < c) :
  a = x ∧ b = y ∧ c = z := by
  sorry

end equality_of_ordered_triples_l861_86121


namespace lego_set_cost_l861_86172

/-- Represents the sale of toys with given conditions and calculates the cost of a Lego set --/
def toy_sale (total_after_tax : ℚ) (car_price : ℚ) (car_discount : ℚ) (num_cars : ℕ) 
             (num_action_figures : ℕ) (tax_rate : ℚ) : ℚ :=
  let discounted_car_price := car_price * (1 - car_discount)
  let action_figure_price := 2 * discounted_car_price
  let board_game_price := action_figure_price + discounted_car_price
  let known_items_total := num_cars * discounted_car_price + 
                           num_action_figures * action_figure_price + 
                           board_game_price
  let total_before_tax := total_after_tax / (1 + tax_rate)
  total_before_tax - known_items_total

/-- Theorem stating that the Lego set costs $85 before tax --/
theorem lego_set_cost : 
  toy_sale 136.5 5 0.1 3 2 0.05 = 85 := by
  sorry

end lego_set_cost_l861_86172


namespace double_elimination_64_teams_games_range_l861_86146

/-- Represents a double-elimination tournament --/
structure DoubleEliminationTournament where
  num_teams : ℕ
  no_ties : Bool

/-- The minimum number of games required to determine a champion in a double-elimination tournament --/
def min_games (t : DoubleEliminationTournament) : ℕ := sorry

/-- The maximum number of games required to determine a champion in a double-elimination tournament --/
def max_games (t : DoubleEliminationTournament) : ℕ := sorry

/-- Theorem stating the range of games required for a 64-team double-elimination tournament --/
theorem double_elimination_64_teams_games_range (t : DoubleEliminationTournament) 
  (h1 : t.num_teams = 64) (h2 : t.no_ties = true) : 
  min_games t = 96 ∧ max_games t = 97 := by sorry

end double_elimination_64_teams_games_range_l861_86146


namespace polynomial_factorization_l861_86132

theorem polynomial_factorization (a : ℝ) : a^3 + 2*a^2 + a = a*(a+1)^2 := by
  sorry

end polynomial_factorization_l861_86132


namespace camden_rico_dog_fraction_l861_86119

/-- Proves that the fraction of dogs Camden bought compared to Rico is 3/4 -/
theorem camden_rico_dog_fraction :
  let justin_dogs : ℕ := 14
  let rico_dogs : ℕ := justin_dogs + 10
  let camden_dog_legs : ℕ := 72
  let legs_per_dog : ℕ := 4
  let camden_dogs : ℕ := camden_dog_legs / legs_per_dog
  (camden_dogs : ℚ) / rico_dogs = 3 / 4 := by
  sorry

end camden_rico_dog_fraction_l861_86119


namespace remainder_sum_l861_86143

theorem remainder_sum (a b c : ℤ) 
  (ha : a % 80 = 75)
  (hb : b % 120 = 115)
  (hc : c % 160 = 155) : 
  (a + b + c) % 40 = 25 := by
sorry

end remainder_sum_l861_86143


namespace maze_paths_count_l861_86182

/-- Represents a junction in the maze --/
structure Junction where
  choices : Nat  -- Number of possible directions at this junction

/-- Represents the maze structure --/
structure Maze where
  entrance_choices : Nat  -- Number of choices at the entrance
  x_junctions : Nat       -- Number of x junctions
  dot_junctions : Nat     -- Number of dot junctions per x junction

/-- Calculates the number of paths through the maze --/
def count_paths (m : Maze) : Nat :=
  m.entrance_choices * m.x_junctions * (2 ^ m.dot_junctions)

/-- Theorem stating that the number of paths in the given maze is 16 --/
theorem maze_paths_count :
  ∃ (m : Maze), count_paths m = 16 :=
sorry

end maze_paths_count_l861_86182


namespace race_completion_time_l861_86127

theorem race_completion_time (walking_time jogging_time total_time : ℕ) : 
  walking_time = 9 →
  jogging_time * 3 = walking_time * 4 →
  total_time = walking_time + jogging_time →
  total_time = 21 := by
sorry

end race_completion_time_l861_86127


namespace trees_planted_by_fourth_grade_l861_86189

theorem trees_planted_by_fourth_grade :
  ∀ (fifth_grade third_grade fourth_grade : ℕ),
    fifth_grade = 114 →
    fifth_grade = 2 * third_grade →
    fourth_grade = third_grade + 32 →
    fourth_grade = 89 := by sorry

end trees_planted_by_fourth_grade_l861_86189


namespace min_max_inequality_l861_86198

theorem min_max_inequality (a b x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : 0 < a) (h₂ : a < b) 
  (h₃ : a ≤ x₁ ∧ x₁ ≤ b) (h₄ : a ≤ x₂ ∧ x₂ ≤ b) 
  (h₅ : a ≤ x₃ ∧ x₃ ≤ b) (h₆ : a ≤ x₄ ∧ x₄ ≤ b) :
  1 ≤ (x₁^2/x₂ + x₂^2/x₃ + x₃^2/x₄ + x₄^2/x₁) / (x₁ + x₂ + x₃ + x₄) ∧ 
  (x₁^2/x₂ + x₂^2/x₃ + x₃^2/x₄ + x₄^2/x₁) / (x₁ + x₂ + x₃ + x₄) ≤ a/b + b/a - 1 :=
by sorry

end min_max_inequality_l861_86198


namespace magic_8_ball_probability_l861_86140

theorem magic_8_ball_probability : 
  let n : ℕ := 7  -- total number of questions
  let k : ℕ := 4  -- number of positive answers we're interested in
  let p : ℚ := 1/3  -- probability of a positive answer for each question
  Nat.choose n k * p^k * (1-p)^(n-k) = 280/2187 := by sorry

end magic_8_ball_probability_l861_86140


namespace point_relation_l861_86197

-- Define the line equation
def line_equation (x y b : ℝ) : Prop := y = -Real.sqrt 2 * x + b

-- Define the theorem
theorem point_relation (m n b : ℝ) 
  (h1 : line_equation (-2) m b)
  (h2 : line_equation 3 n b) : 
  m > n := by sorry

end point_relation_l861_86197


namespace min_c_squared_l861_86176

theorem min_c_squared (a b c : ℝ) (A B C : ℝ) : 
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + 2*b = 4 →
  a * Real.sin A + 4*b * Real.sin B = 6*a * Real.sin B * Real.sin C →
  c^2 ≥ 5 - (4 * Real.sqrt 5) / 3 :=
by sorry

end min_c_squared_l861_86176


namespace charles_earnings_l861_86134

def housesitting_rate : ℕ := 15
def dog_walking_rate : ℕ := 22
def housesitting_hours : ℕ := 10
def dogs_walked : ℕ := 3
def hours_per_dog : ℕ := 1

def total_earnings : ℕ := housesitting_rate * housesitting_hours + dog_walking_rate * dogs_walked * hours_per_dog

theorem charles_earnings : total_earnings = 216 := by
  sorry

end charles_earnings_l861_86134


namespace x_minus_q_in_terms_of_q_l861_86118

theorem x_minus_q_in_terms_of_q (x q : ℝ) (h1 : |x - 3| = q) (h2 : x < 3) : x - q = 3 - 2*q := by
  sorry

end x_minus_q_in_terms_of_q_l861_86118


namespace seventh_number_is_177_l861_86179

def digit_sum (n : ℕ) : ℕ := sorry

def is_valid_number (n : ℕ) : Prop :=
  n > 0 ∧ digit_sum n = 15

def nth_valid_number (n : ℕ) : ℕ := sorry

theorem seventh_number_is_177 : nth_valid_number 7 = 177 := by sorry

end seventh_number_is_177_l861_86179


namespace a_is_perfect_square_l861_86136

/-- Sequence c_n defined recursively -/
def c : ℕ → ℤ
  | 0 => 1
  | 1 => 0
  | 2 => 2005
  | (n + 3) => -3 * c (n + 1) - 4 * c n + 2008

/-- Sequence a_n defined in terms of c_n -/
def a (n : ℕ) : ℤ :=
  5 * (c (n + 2) - c n) * (502 - c (n - 1) - c (n - 2)) + 4^n * 2004 * 501

/-- Theorem stating that a_n is a perfect square for n > 2 -/
theorem a_is_perfect_square (n : ℕ) (h : n > 2) : ∃ k : ℤ, a n = k^2 := by
  sorry

end a_is_perfect_square_l861_86136


namespace theater_seats_l861_86173

/-- The number of people watching the movie -/
def people_watching : ℕ := 532

/-- The number of empty seats -/
def empty_seats : ℕ := 218

/-- The total number of seats in the theater -/
def total_seats : ℕ := people_watching + empty_seats

theorem theater_seats : total_seats = 750 := by sorry

end theater_seats_l861_86173


namespace negative_x_times_three_minus_x_l861_86184

theorem negative_x_times_three_minus_x (x : ℝ) : -x * (3 - x) = -3*x + x^2 := by
  sorry

end negative_x_times_three_minus_x_l861_86184


namespace sum_of_coefficients_is_nine_l861_86129

/-- A quadratic function with roots satisfying specific conditions -/
structure QuadraticWithSpecialRoots where
  a : ℝ
  b : ℝ
  m : ℝ
  n : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_roots : m ≠ n ∧ m > 0 ∧ n > 0
  h_vieta : m + n = a ∧ m * n = b
  h_arithmetic : (m - n = n - (-2)) ∨ (n - m = m - (-2))
  h_geometric : (m / n = n / (-2)) ∨ (n / m = m / (-2))

/-- The sum of coefficients a and b equals 9 -/
theorem sum_of_coefficients_is_nine (q : QuadraticWithSpecialRoots) : q.a + q.b = 9 := by
  sorry

end sum_of_coefficients_is_nine_l861_86129


namespace cyclic_sum_inequality_l861_86178

theorem cyclic_sum_inequality (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 1) :
  Real.sqrt (1 - a*b) + Real.sqrt (1 - b*c) + Real.sqrt (1 - c*d) + 
  Real.sqrt (1 - d*a) + Real.sqrt (1 - a*c) + Real.sqrt (1 - b*d) ≤ 2 * Real.sqrt 3 := by
  sorry

end cyclic_sum_inequality_l861_86178


namespace donut_combinations_l861_86111

/-- The number of ways to distribute n identical objects into k distinct boxes -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The total number of donuts Josh needs to buy -/
def total_donuts : ℕ := 8

/-- The number of different types of donuts -/
def donut_types : ℕ := 5

/-- The number of donuts Josh must buy of the first type -/
def first_type_min : ℕ := 2

/-- The number of donuts Josh must buy of each other type -/
def other_types_min : ℕ := 1

/-- The number of remaining donuts to distribute after meeting minimum requirements -/
def remaining_donuts : ℕ := total_donuts - (first_type_min + (donut_types - 1) * other_types_min)

theorem donut_combinations : stars_and_bars remaining_donuts donut_types = 15 := by
  sorry

end donut_combinations_l861_86111


namespace total_stamps_l861_86112

theorem total_stamps (harry_stamps : ℕ) (sister_stamps : ℕ) : 
  harry_stamps = 180 → sister_stamps = 60 → harry_stamps + sister_stamps = 240 :=
by
  sorry

end total_stamps_l861_86112


namespace amy_pencil_count_l861_86104

/-- The number of pencils Amy has after buying and giving away some pencils -/
def final_pencil_count (initial : ℕ) (bought_monday : ℕ) (bought_tuesday : ℕ) (given_away : ℕ) : ℕ :=
  initial + bought_monday + bought_tuesday - given_away

/-- Theorem stating that Amy has 12 pencils at the end -/
theorem amy_pencil_count : final_pencil_count 3 7 4 2 = 12 := by
  sorry

end amy_pencil_count_l861_86104


namespace thomas_savings_l861_86161

/-- Thomas's savings scenario --/
theorem thomas_savings (
  weekly_allowance : ℝ)
  (weeks_per_year : ℕ)
  (hours_per_week : ℕ)
  (car_cost : ℝ)
  (weekly_spending : ℝ)
  (additional_savings_needed : ℝ)
  (hourly_wage : ℝ)
  (h1 : weekly_allowance = 50)
  (h2 : weeks_per_year = 52)
  (h3 : hours_per_week = 30)
  (h4 : car_cost = 15000)
  (h5 : weekly_spending = 35)
  (h6 : additional_savings_needed = 2000)
  : hourly_wage = 7.83 := by
  sorry

end thomas_savings_l861_86161


namespace monotonic_function_k_range_l861_86148

theorem monotonic_function_k_range (k : ℝ) :
  (∀ x ≥ 1, Monotone (fun x : ℝ ↦ 4 * x^2 - k * x - 8)) →
  k ≤ 8 := by
  sorry

end monotonic_function_k_range_l861_86148


namespace friend_team_assignment_count_l861_86156

theorem friend_team_assignment_count : 
  let n_friends : ℕ := 8
  let n_teams : ℕ := 4
  n_teams ^ n_friends = 65536 := by
  sorry

end friend_team_assignment_count_l861_86156


namespace yearly_fluid_intake_l861_86188

def weekday_soda : ℕ := 5 * 12
def weekday_water : ℕ := 64
def weekday_juice : ℕ := 3 * 8
def weekday_sports : ℕ := 2 * 16

def weekend_soda : ℕ := 5 * 12
def weekend_water : ℕ := 64
def weekend_juice : ℕ := 3 * 8
def weekend_sports : ℕ := 1 * 16
def weekend_smoothie : ℕ := 32

def weekdays : ℕ := 260
def weekend_days : ℕ := 104
def holidays : ℕ := 1

def weekday_total : ℕ := weekday_soda + weekday_water + weekday_juice + weekday_sports
def weekend_total : ℕ := weekend_soda + weekend_water + weekend_juice + weekend_sports + weekend_smoothie

theorem yearly_fluid_intake :
  weekday_total * weekdays + weekend_total * (weekend_days + holidays) = 67380 := by
  sorry

end yearly_fluid_intake_l861_86188


namespace arithmetic_sequence_sum_property_l861_86103

/-- An arithmetic sequence is a sequence where the difference between 
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_property 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h1 : a 1 + a 4 + a 7 = 45) 
  (h2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 33 := by
sorry

end arithmetic_sequence_sum_property_l861_86103


namespace percent_commutation_l861_86139

theorem percent_commutation (x : ℝ) (h : 0.3 * 0.4 * x = 36) :
  0.4 * 0.3 * x = 0.3 * 0.4 * x :=
by
  sorry

end percent_commutation_l861_86139


namespace total_feed_amount_l861_86125

/-- Represents the total amount of dog feed mixed -/
def total_feed (cheap_feed expensive_feed : ℝ) : ℝ := cheap_feed + expensive_feed

/-- Represents the total cost of the mixed feed -/
def total_cost (cheap_feed expensive_feed : ℝ) : ℝ :=
  0.18 * cheap_feed + 0.53 * expensive_feed

/-- The theorem stating the total amount of feed mixed -/
theorem total_feed_amount :
  ∃ (expensive_feed : ℝ),
    total_feed 17 expensive_feed = 35 ∧
    total_cost 17 expensive_feed = 0.36 * total_feed 17 expensive_feed :=
sorry

end total_feed_amount_l861_86125


namespace inequality_solution_set_l861_86181

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, |x - 2| + |x - 1| > a) → a < 1 := by
  sorry

end inequality_solution_set_l861_86181


namespace max_intersection_points_for_arrangement_l861_86174

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  convex : Bool

/-- Represents the arrangement of two convex polygons in a plane -/
structure PolygonArrangement where
  A₁ : ConvexPolygon
  A₂ : ConvexPolygon
  same_plane : Bool
  can_intersect : Bool
  no_full_overlap : Bool

/-- Calculates the maximum number of intersection points between two polygons -/
def max_intersection_points (arr : PolygonArrangement) : ℕ :=
  arr.A₁.sides * arr.A₂.sides

/-- Theorem stating the maximum number of intersection points for the given arrangement -/
theorem max_intersection_points_for_arrangement 
  (m : ℕ) 
  (arr : PolygonArrangement) 
  (h1 : arr.A₁.sides = m) 
  (h2 : arr.A₂.sides = m + 2) 
  (h3 : arr.same_plane) 
  (h4 : arr.can_intersect) 
  (h5 : arr.no_full_overlap) 
  (h6 : arr.A₁.convex) 
  (h7 : arr.A₂.convex) : 
  max_intersection_points arr = m^2 + 2*m := by
  sorry

end max_intersection_points_for_arrangement_l861_86174


namespace inequality_solution_set_l861_86192

def solution_set (x : ℝ) : Prop := -2 < x ∧ x < 3

theorem inequality_solution_set :
  ∀ x : ℝ, (x - 3) * (x + 2) < 0 ↔ solution_set x :=
by sorry

end inequality_solution_set_l861_86192


namespace hyperbola_eccentricity_range_l861_86114

theorem hyperbola_eccentricity_range (a b : ℝ) (M : ℝ × ℝ) (F P Q : ℝ × ℝ) (h1 : a > 0) (h2 : b > 0) :
  let (x, y) := M
  (x^2 / a^2 - y^2 / b^2 = 1) →  -- M is on the hyperbola
  (F.1 = a * (a^2 + b^2).sqrt / (a^2 + b^2).sqrt ∧ F.2 = 0) →  -- F is a focus on x-axis
  (∃ r : ℝ, (M.1 - F.1)^2 + M.2^2 = r^2 ∧ P.1 = 0 ∧ Q.1 = 0 ∧ (P.2 - M.2)^2 + M.1^2 = r^2 ∧ (Q.2 - M.2)^2 + M.1^2 = r^2) →  -- Circle condition
  (0 < Real.arccos ((P.2 - M.2) * (Q.2 - M.2) / (((P.2 - M.2)^2 + M.1^2) * ((Q.2 - M.2)^2 + M.1^2)).sqrt) ∧ 
   Real.arccos ((P.2 - M.2) * (Q.2 - M.2) / (((P.2 - M.2)^2 + M.1^2) * ((Q.2 - M.2)^2 + M.1^2)).sqrt) < π/2) →  -- Acute triangle condition
  let e := ((a^2 + b^2) / a^2).sqrt
  (Real.sqrt 5 + 1) / 2 < e ∧ e < (Real.sqrt 6 + Real.sqrt 2) / 2 := by
sorry

end hyperbola_eccentricity_range_l861_86114


namespace money_ratio_l861_86154

theorem money_ratio (j : ℝ) (k : ℝ) : 
  (j + (2 * j - 7) + 60 = 113) →  -- Sum of all money
  (60 = k * j) →                  -- Patricia's money is a multiple of Jethro's
  (60 : ℝ) / j = 3 :=             -- Ratio of Patricia's to Jethro's money
by
  sorry

end money_ratio_l861_86154


namespace consecutive_even_odd_squares_divisibility_l861_86100

theorem consecutive_even_odd_squares_divisibility :
  (∀ n : ℕ+, ∃ k : ℕ, (2*n+2)^2 - (2*n)^2 = 4*k) ∧
  (∀ m : ℕ+, ∃ k : ℕ, (2*m+1)^2 - (2*m-1)^2 = 8*k) :=
by sorry

end consecutive_even_odd_squares_divisibility_l861_86100


namespace insufficient_info_for_both_correct_evans_class_test_l861_86115

theorem insufficient_info_for_both_correct (total_students : ℕ) 
  (q1_correct : ℕ) (absent : ℕ) (q2_correct : ℕ) : Prop :=
  total_students = 40 ∧ 
  q1_correct = 30 ∧ 
  absent = 10 ∧
  q2_correct ≥ 0 ∧ q2_correct ≤ (total_students - absent) →
  ∃ (both_correct₁ both_correct₂ : ℕ), 
    both_correct₁ ≠ both_correct₂ ∧
    both_correct₁ ≥ 0 ∧ both_correct₁ ≤ q1_correct ∧
    both_correct₂ ≥ 0 ∧ both_correct₂ ≤ q1_correct ∧
    both_correct₁ ≤ q2_correct ∧ both_correct₂ ≤ q2_correct

theorem evans_class_test : insufficient_info_for_both_correct 40 30 10 q2_correct :=
sorry

end insufficient_info_for_both_correct_evans_class_test_l861_86115


namespace albert_earnings_increase_l861_86166

theorem albert_earnings_increase (E : ℝ) (P : ℝ) : 
  E * (1 + P / 100) = 598 →
  E * 1.35 = 621 →
  P = 30 := by
sorry

end albert_earnings_increase_l861_86166


namespace bakery_sugar_amount_l861_86149

/-- Given the ratios of ingredients in a bakery storage room, prove the amount of sugar. -/
theorem bakery_sugar_amount 
  (sugar flour baking_soda : ℚ) 
  (h1 : sugar / flour = 5 / 6)
  (h2 : flour / baking_soda = 10 / 1)
  (h3 : flour / (baking_soda + 60) = 8 / 1) :
  sugar = 2000 := by
  sorry

#check bakery_sugar_amount

end bakery_sugar_amount_l861_86149


namespace mercedes_jonathan_ratio_l861_86185

def jonathan_distance : ℝ := 7.5

def mercedes_davonte_total : ℝ := 32

theorem mercedes_jonathan_ratio : 
  ∃ (mercedes_distance : ℝ),
    mercedes_distance + (mercedes_distance + 2) = mercedes_davonte_total ∧
    mercedes_distance / jonathan_distance = 2 := by
  sorry

end mercedes_jonathan_ratio_l861_86185


namespace remainder_theorem_l861_86123

theorem remainder_theorem : ∃ q : ℕ, 2^300 + 300 = (2^150 + 2^75 + 1) * q + 1 := by
  sorry

end remainder_theorem_l861_86123


namespace sum_lowest_two_scores_l861_86145

/-- Represents a set of math test scores -/
structure MathTests where
  scores : Finset ℕ
  count : Nat
  average : ℕ
  median : ℕ
  mode : ℕ

/-- The sum of the lowest two scores in a set of math tests -/
def sumLowestTwo (tests : MathTests) : ℕ :=
  sorry

/-- Theorem: Given 5 math test scores with an average of 90, a median of 91, 
    and a mode of 93, the sum of the lowest two scores is 173 -/
theorem sum_lowest_two_scores (tests : MathTests) 
  (h_count : tests.count = 5)
  (h_avg : tests.average = 90)
  (h_median : tests.median = 91)
  (h_mode : tests.mode = 93) :
  sumLowestTwo tests = 173 := by
  sorry

end sum_lowest_two_scores_l861_86145


namespace ellipse_a_range_l861_86147

/-- Represents an ellipse with the given equation and foci on the x-axis -/
structure Ellipse (a : ℝ) :=
  (eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / (a + 6) = 1)
  (foci_on_x : True)  -- We don't need to formalize this condition for the proof

/-- The range of a for which the given equation represents an ellipse with foci on the x-axis -/
theorem ellipse_a_range (a : ℝ) (e : Ellipse a) : a > 3 ∨ (-6 < a ∧ a < -2) :=
sorry

end ellipse_a_range_l861_86147


namespace z_change_l861_86138

theorem z_change (w h z : ℝ) (z' : ℝ) : 
  let q := 5 * w / (4 * h * z^2)
  let q' := 5 * (4 * w) / (4 * (2 * h) * z'^2)
  q' / q = 2 / 9 →
  z' / z = 3 * Real.sqrt 2 / 2 := by
sorry

end z_change_l861_86138


namespace vector_problem_l861_86124

/-- Given points A, B, C in ℝ², and vectors a, b, c, prove the following statements. -/
theorem vector_problem (A B C M N : ℝ × ℝ) (a b c : ℝ × ℝ) :
  A = (-2, 4) →
  B = (3, -1) →
  C = (-3, -4) →
  a = B - A →
  b = C - B →
  c = A - C →
  M - C = 3 • c →
  N - C = -2 • b →
  (3 • a + b - 3 • c = (6, -42)) ∧
  (a = -b - c) ∧
  (M = (0, 20) ∧ N = (9, 2) ∧ N - M = (9, -18)) := by
sorry


end vector_problem_l861_86124


namespace number_problem_l861_86151

theorem number_problem : ∃ n : ℝ, n - (1002 / 20.04) = 2984 ∧ n = 3034 := by
  sorry

end number_problem_l861_86151


namespace system_solution_l861_86186

theorem system_solution (x y : ℝ) (eq1 : 2 * x + y = 7) (eq2 : x + 2 * y = 8) : x - y = -1 := by
  sorry

end system_solution_l861_86186


namespace negation_of_p_l861_86163

variable (I : Set ℝ)

def p : Prop := ∀ x ∈ I, x^3 - x^2 + 1 ≤ 0

theorem negation_of_p : 
  ¬p I ↔ ∃ x ∈ I, x^3 - x^2 + 1 > 0 := by sorry

end negation_of_p_l861_86163


namespace sugar_solution_mixing_l861_86135

/-- Calculates the percentage of sugar in the resulting solution after replacing
    a portion of an initial sugar solution with another sugar solution. -/
theorem sugar_solution_mixing (initial_sugar_percentage : ℝ)
                               (replacement_portion : ℝ)
                               (replacement_sugar_percentage : ℝ) :
  initial_sugar_percentage = 8 →
  replacement_portion = 1/4 →
  replacement_sugar_percentage = 40 →
  let remaining_portion := 1 - replacement_portion
  let initial_sugar := initial_sugar_percentage * remaining_portion
  let replacement_sugar := replacement_sugar_percentage * replacement_portion
  let final_sugar_percentage := initial_sugar + replacement_sugar
  final_sugar_percentage = 16 := by
sorry

end sugar_solution_mixing_l861_86135


namespace point_in_first_quadrant_l861_86191

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition from the problem
def condition (x y : ℝ) : Prop :=
  x / (1 + i) = 1 - y * i

-- State the theorem
theorem point_in_first_quadrant (x y : ℝ) (h : condition x y) :
  x > 0 ∧ y > 0 := by
  sorry

end point_in_first_quadrant_l861_86191


namespace remaining_painting_time_l861_86142

/-- Calculates the remaining painting time for a building -/
def remaining_time (total_rooms : ℕ) (hours_per_room : ℕ) (painted_rooms : ℕ) : ℕ :=
  (total_rooms - painted_rooms) * hours_per_room

/-- Theorem: The remaining time to finish all painting work is 155 hours -/
theorem remaining_painting_time : 
  let building1 := remaining_time 12 7 5
  let building2 := remaining_time 15 6 4
  let building3 := remaining_time 10 5 2
  building1 + building2 + building3 = 155 := by
  sorry

end remaining_painting_time_l861_86142


namespace cube_difference_l861_86122

theorem cube_difference (x y : ℚ) (h1 : x + y = 10) (h2 : 2 * x - y = 16) :
  x^3 - y^3 = 17512 / 27 := by
  sorry

end cube_difference_l861_86122


namespace hyperbola_circle_no_intersection_l861_86190

/-- The range of real values of a for which the asymptotes of the hyperbola x^2/4 - y^2 = 1
    have no common points with the circle x^2 + y^2 - 2ax + 1 = 0 -/
theorem hyperbola_circle_no_intersection (a : ℝ) : 
  (∀ x y : ℝ, x^2/4 - y^2 = 1 → x^2 + y^2 - 2*a*x + 1 ≠ 0) ↔ 
  (a ∈ Set.Ioo (-Real.sqrt 5 / 2) (-1) ∪ Set.Ioo 1 (Real.sqrt 5 / 2)) :=
sorry

end hyperbola_circle_no_intersection_l861_86190


namespace abs_eq_sqrt_square_l861_86113

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

end abs_eq_sqrt_square_l861_86113


namespace range_of_m_l861_86195

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x ≤ 5}

-- Define set B parameterized by m
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem statement
theorem range_of_m (m : ℝ) : B m ⊆ A → m ≤ 3 := by
  sorry

end range_of_m_l861_86195


namespace sum_of_specific_series_l861_86187

def geometric_series (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_specific_series :
  let a : ℚ := 1/2
  let r : ℚ := -1/4
  let n : ℕ := 6
  geometric_series a r n = 4095/10240 := by
sorry

end sum_of_specific_series_l861_86187
