import Mathlib

namespace min_triangles_to_cover_l687_68753

theorem min_triangles_to_cover (side_large : ℝ) (side_small : ℝ) : 
  side_large = 12 → side_small = 1 → 
  (side_large / side_small) ^ 2 = 144 := by
  sorry

end min_triangles_to_cover_l687_68753


namespace modular_inverse_11_mod_1000_l687_68796

theorem modular_inverse_11_mod_1000 : ∃ x : ℕ, x < 1000 ∧ (11 * x) % 1000 = 1 :=
  by
  use 91
  sorry

end modular_inverse_11_mod_1000_l687_68796


namespace vector_addition_theorem_l687_68774

/-- Given vectors a and b, prove that 2a + b equals the specified result -/
theorem vector_addition_theorem (a b : ℝ × ℝ × ℝ) :
  a = (1, 2, -3) →
  b = (5, -7, 8) →
  (2 : ℝ) • a + b = (7, -3, 2) := by sorry

end vector_addition_theorem_l687_68774


namespace final_mango_distribution_l687_68797

/-- Represents the state of mango distribution among friends in a circle. -/
structure MangoDistribution :=
  (friends : ℕ)
  (mangos : List ℕ)

/-- Defines the rules for sharing mangos. -/
def share (d : MangoDistribution) : MangoDistribution :=
  sorry

/-- Defines the rules for eating mangos. -/
def eat (d : MangoDistribution) : MangoDistribution :=
  sorry

/-- Checks if any further actions (sharing or eating) are possible. -/
def canContinue (d : MangoDistribution) : Bool :=
  sorry

/-- Applies sharing and eating rules until no further actions are possible. -/
def applyRulesUntilStable (d : MangoDistribution) : MangoDistribution :=
  sorry

/-- Counts the number of people with mangos in the final distribution. -/
def countPeopleWithMangos (d : MangoDistribution) : ℕ :=
  sorry

/-- Main theorem stating that exactly 8 people will have mangos at the end. -/
theorem final_mango_distribution
  (initial : MangoDistribution)
  (h1 : initial.friends = 100)
  (h2 : initial.mangos = [2019] ++ List.replicate 99 0) :
  countPeopleWithMangos (applyRulesUntilStable initial) = 8 :=
sorry

end final_mango_distribution_l687_68797


namespace hyperbola_iff_m_negative_l687_68755

/-- A conic section defined by the equation x^2 + my^2 = 1 -/
structure Conic (m : ℝ) where
  x : ℝ
  y : ℝ
  eq : x^2 + m*y^2 = 1

/-- Definition of a hyperbola -/
def IsHyperbola (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 ↔ x^2 + m*y^2 = 1

/-- Theorem: The equation x^2 + my^2 = 1 represents a hyperbola if and only if m < 0 -/
theorem hyperbola_iff_m_negative (m : ℝ) : IsHyperbola m ↔ m < 0 :=
sorry

end hyperbola_iff_m_negative_l687_68755


namespace sine_cosine_inequality_l687_68793

theorem sine_cosine_inequality (n : ℕ+) (x : ℝ) : 
  (Real.sin (2 * x))^(n : ℝ) + (Real.sin x^(n : ℝ) - Real.cos x^(n : ℝ))^2 ≤ 1 := by
  sorry

end sine_cosine_inequality_l687_68793


namespace ten_boys_handshakes_l687_68739

/-- The number of handshakes when n boys each shake hands once with every other boy -/
def handshakes (n : ℕ) : ℕ := n.choose 2

/-- Theorem: When 10 boys each shake hands once with every other boy, there are 45 handshakes -/
theorem ten_boys_handshakes : handshakes 10 = 45 := by
  sorry

end ten_boys_handshakes_l687_68739


namespace hyperbola_tangent_equation_l687_68716

-- Define the ellipse
def is_ellipse (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the hyperbola
def is_hyperbola (x y a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the tangent line for the ellipse
def is_ellipse_tangent (x y x₀ y₀ a b : ℝ) : Prop := (x₀ * x / a^2) + (y₀ * y / b^2) = 1

-- Define the tangent line for the hyperbola
def is_hyperbola_tangent (x y x₀ y₀ a b : ℝ) : Prop := (x₀ * x / a^2) - (y₀ * y / b^2) = 1

-- State the theorem
theorem hyperbola_tangent_equation (a b x₀ y₀ : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_point : is_hyperbola x₀ y₀ a b) :
  is_hyperbola_tangent x y x₀ y₀ a b :=
sorry

end hyperbola_tangent_equation_l687_68716


namespace least_subtrahend_for_divisibility_l687_68786

theorem least_subtrahend_for_divisibility (n m : ℕ) (hn : n = 13602) (hm : m = 87) :
  ∃ (k : ℕ), k = 30 ∧ 
  (∀ (x : ℕ), x < k → ¬(∃ (q : ℕ), n - x = m * q)) ∧
  (∃ (q : ℕ), n - k = m * q) :=
sorry

end least_subtrahend_for_divisibility_l687_68786


namespace arccos_one_eq_zero_l687_68779

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l687_68779


namespace smallest_divisible_by_10_11_12_l687_68772

theorem smallest_divisible_by_10_11_12 : ∃ (n : ℕ), n > 0 ∧ 
  10 ∣ n ∧ 11 ∣ n ∧ 12 ∣ n ∧ 
  ∀ (m : ℕ), m > 0 ∧ 10 ∣ m ∧ 11 ∣ m ∧ 12 ∣ m → n ≤ m :=
by
  use 660
  sorry

end smallest_divisible_by_10_11_12_l687_68772


namespace mindys_tax_rate_l687_68737

/-- Given Mork's tax rate, Mindy's income relative to Mork's, and their combined tax rate,
    calculate Mindy's tax rate. -/
theorem mindys_tax_rate
  (morks_tax_rate : ℝ)
  (mindys_income_ratio : ℝ)
  (combined_tax_rate : ℝ)
  (h1 : morks_tax_rate = 0.40)
  (h2 : mindys_income_ratio = 3)
  (h3 : combined_tax_rate = 0.325) :
  let mindys_tax_rate := (combined_tax_rate * (1 + mindys_income_ratio) - morks_tax_rate) / mindys_income_ratio
  mindys_tax_rate = 0.30 :=
by sorry

end mindys_tax_rate_l687_68737


namespace saltwater_volume_l687_68756

/-- Proves that the initial volume of a saltwater solution is 160 gallons --/
theorem saltwater_volume : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (0.20 * x / x = 1/5) ∧ 
  ((0.20 * x + 16) / (3/4 * x + 24) = 1/3) ∧ 
  (x = 160) := by
  sorry

end saltwater_volume_l687_68756


namespace regular_polygon_sides_l687_68764

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (180 * (n - 2) : ℝ) = 140 * n → n = 9 := by
  sorry

end regular_polygon_sides_l687_68764


namespace quadratic_root_implies_a_values_l687_68782

theorem quadratic_root_implies_a_values (a : ℝ) : 
  ((-2)^2 + (3/2) * a * (-2) - a^2 = 0) → (a = 1 ∨ a = -4) := by
  sorry

end quadratic_root_implies_a_values_l687_68782


namespace sum_diagonal_blocks_420_eq_2517_l687_68711

/-- Given a 420 × 420 square grid tiled with 1 × 2 blocks, this function calculates
    the sum of all possible values for the total number of blocks
    that the two diagonals pass through. -/
def sum_diagonal_blocks_420 : ℕ :=
  let grid_size : ℕ := 420
  let diagonal_squares : ℕ := 2 * grid_size
  let non_center_squares : ℕ := diagonal_squares - 4
  let non_center_blocks : ℕ := non_center_squares
  let min_center_blocks : ℕ := 2
  let max_center_blocks : ℕ := 4
  (non_center_blocks + min_center_blocks) +
  (non_center_blocks + min_center_blocks + 1) +
  (non_center_blocks + max_center_blocks)

theorem sum_diagonal_blocks_420_eq_2517 :
  sum_diagonal_blocks_420 = 2517 := by
  sorry

end sum_diagonal_blocks_420_eq_2517_l687_68711


namespace product_change_l687_68780

theorem product_change (a b : ℝ) (h : (a - 3) * (b + 3) - a * b = 900) : 
  a * b - (a + 3) * (b - 3) = 918 := by
sorry

end product_change_l687_68780


namespace fraction_value_at_three_l687_68726

theorem fraction_value_at_three :
  let x : ℝ := 3
  (x^8 + 16*x^4 + 64) / (x^4 + 8) = 89 := by
  sorry

end fraction_value_at_three_l687_68726


namespace triangle_area_product_l687_68761

theorem triangle_area_product (a b : ℝ) : 
  a > 0 → b > 0 → (1/2 * (8/a) * (8/b) = 8) → a * b = 4 := by
  sorry

end triangle_area_product_l687_68761


namespace divisible_by_30_l687_68703

theorem divisible_by_30 (n : ℤ) : ∃ k : ℤ, n^5 - n = 30 * k := by
  sorry

end divisible_by_30_l687_68703


namespace least_four_digit_divisible_l687_68783

/-- A function that checks if a number has all different digits -/
def has_different_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

/-- A function that checks if a number is divisible by another number -/
def is_divisible_by (n m : ℕ) : Prop :=
  n % m = 0

theorem least_four_digit_divisible :
  ∀ n : ℕ,
    1000 ≤ n                          -- four-digit number
    → n < 10000                       -- four-digit number
    → has_different_digits n          -- all digits are different
    → is_divisible_by n 1             -- divisible by 1
    → is_divisible_by n 2             -- divisible by 2
    → is_divisible_by n 4             -- divisible by 4
    → is_divisible_by n 8             -- divisible by 8
    → 1248 ≤ n                        -- 1248 is the least such number
  := by sorry

end least_four_digit_divisible_l687_68783


namespace name_length_difference_l687_68735

/-- Given that Elida has 5 letters in her name and the total of 10 times the average number
    of letters in both names is 65, prove that Adrianna's name has 3 more letters than Elida's name. -/
theorem name_length_difference (elida_length : ℕ) (adrianna_length : ℕ) : 
  elida_length = 5 →
  10 * ((elida_length + adrianna_length) / 2) = 65 →
  adrianna_length = elida_length + 3 := by
sorry


end name_length_difference_l687_68735


namespace least_four_digit_with_conditions_l687_68745

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_different_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

def contains_digit (n d : ℕ) : Prop :=
  d ∈ n.digits 10

def divisible_by_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0 → n % d = 0

theorem least_four_digit_with_conditions :
  ∀ n : ℕ,
    is_four_digit n ∧
    has_different_digits n ∧
    contains_digit n 5 ∧
    divisible_by_digits n →
    5124 ≤ n :=
by sorry

end least_four_digit_with_conditions_l687_68745


namespace divisor_problem_l687_68706

theorem divisor_problem (d : ℕ+) : 
  (∃ n : ℕ, n % d = 3 ∧ (2 * n) % d = 2) → d = 4 := by
  sorry

end divisor_problem_l687_68706


namespace brady_earnings_correct_l687_68794

/-- Calculates the total earnings for Brady's transcription work -/
def brady_earnings (basic_cards : ℕ) (gourmet_cards : ℕ) : ℚ :=
  let basic_rate : ℚ := 70 / 100
  let gourmet_rate : ℚ := 90 / 100
  let basic_earnings := basic_rate * basic_cards
  let gourmet_earnings := gourmet_rate * gourmet_cards
  let card_earnings := basic_earnings + gourmet_earnings
  let total_cards := basic_cards + gourmet_cards
  let bonus_count := total_cards / 100
  let bonus_base := 10
  let bonus_increment := 5
  let bonus_total := bonus_count * bonus_base + (bonus_count * (bonus_count - 1) / 2) * bonus_increment
  card_earnings + bonus_total

theorem brady_earnings_correct :
  brady_earnings 120 80 = 181 := by
  sorry

end brady_earnings_correct_l687_68794


namespace average_rate_of_change_f_on_1_5_l687_68791

def f (x : ℝ) : ℝ := 2 * x + 1

theorem average_rate_of_change_f_on_1_5 :
  (f 5 - f 1) / (5 - 1) = 2 := by
  sorry

end average_rate_of_change_f_on_1_5_l687_68791


namespace quadruple_solution_l687_68729

theorem quadruple_solution (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a * b * c * d = 1)
  (h2 : a^2012 + 2012 * b = 2012 * c + d^2012)
  (h3 : 2012 * a + b^2012 = c^2012 + 2012 * d) :
  ∃ t : ℝ, t > 0 ∧ a = t ∧ b = 1/t ∧ c = 1/t ∧ d = t :=
sorry

end quadruple_solution_l687_68729


namespace inner_prism_volume_l687_68762

theorem inner_prism_volume (w l h : ℕ+) : 
  w * l * h = 128 ↔ (w : ℕ) * (l : ℕ) * (h : ℕ) = 128 := by sorry

end inner_prism_volume_l687_68762


namespace tuesday_rainfall_l687_68718

/-- Given that it rained 0.9 inches on Monday and Tuesday's rainfall was 0.7 inches less than Monday's,
    prove that it rained 0.2 inches on Tuesday. -/
theorem tuesday_rainfall (monday_rain : ℝ) (tuesday_difference : ℝ) 
  (h1 : monday_rain = 0.9)
  (h2 : tuesday_difference = 0.7) :
  monday_rain - tuesday_difference = 0.2 := by
  sorry

end tuesday_rainfall_l687_68718


namespace mean_of_added_numbers_l687_68744

theorem mean_of_added_numbers (original_list : List ℝ) (x y z : ℝ) :
  original_list.length = 7 →
  original_list.sum / original_list.length = 48 →
  let new_list := original_list ++ [x, y, z]
  new_list.sum / new_list.length = 55 →
  (x + y + z) / 3 = 71 + 1/3 := by
sorry

end mean_of_added_numbers_l687_68744


namespace unique_favorite_number_l687_68776

def is_favorite_number (n : ℕ) : Prop :=
  80 < n ∧ n ≤ 130 ∧
  n % 13 = 0 ∧
  n % 3 ≠ 0 ∧
  (n / 100 + (n / 10) % 10 + n % 10) % 4 = 0

theorem unique_favorite_number : ∃! n, is_favorite_number n :=
  sorry

end unique_favorite_number_l687_68776


namespace telescope_visibility_increase_l687_68701

theorem telescope_visibility_increase (min_without max_without min_with max_with : ℝ) 
  (h1 : min_without = 100)
  (h2 : max_without = 110)
  (h3 : min_with = 150)
  (h4 : max_with = 165) :
  let avg_without := (min_without + max_without) / 2
  let avg_with := (min_with + max_with) / 2
  (avg_with - avg_without) / avg_without * 100 = 50 := by
  sorry

end telescope_visibility_increase_l687_68701


namespace meat_for_hamburgers_l687_68724

/-- Given that Rachelle uses 4 pounds of meat to make 10 hamburgers,
    prove that she needs 12 pounds of meat to make 30 hamburgers. -/
theorem meat_for_hamburgers (meat_for_10 : ℝ) (hamburgers_for_10 : ℕ)
    (meat_for_30 : ℝ) (hamburgers_for_30 : ℕ) :
    meat_for_10 = 4 ∧ hamburgers_for_10 = 10 ∧ hamburgers_for_30 = 30 →
    meat_for_30 = 12 := by
  sorry

end meat_for_hamburgers_l687_68724


namespace circle_configuration_theorem_l687_68723

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents the configuration of three circles C, D, and E -/
structure CircleConfiguration where
  C : Circle
  D : Circle
  E : Circle
  C_radius_is_4 : C.radius = 4
  D_internally_tangent_to_C : True  -- This is a placeholder for the tangency condition
  E_internally_tangent_to_C : True  -- This is a placeholder for the tangency condition
  E_externally_tangent_to_D : True  -- This is a placeholder for the tangency condition
  E_tangent_to_diameter : True      -- This is a placeholder for the tangency condition
  D_radius_twice_E : D.radius = 2 * E.radius

theorem circle_configuration_theorem (config : CircleConfiguration) 
  (p q : ℕ) (h : config.D.radius = Real.sqrt p - q) : 
  p + q = 259 := by
  sorry

end circle_configuration_theorem_l687_68723


namespace field_planting_fraction_l687_68759

theorem field_planting_fraction :
  ∀ (a b c x : ℝ),
  a = 5 ∧ b = 12 ∧ c^2 = a^2 + b^2 →
  x^2 * c = 3 * (a * b) →
  (a * b - x^2) / (a * b) = 7 / 9 := by
sorry

end field_planting_fraction_l687_68759


namespace vanessa_score_l687_68792

/-- Calculates Vanessa's score in a basketball game -/
theorem vanessa_score (total_score : ℕ) (other_players : ℕ) (avg_score : ℕ) : 
  total_score = 72 → other_players = 7 → avg_score = 6 →
  total_score - (other_players * avg_score) = 30 := by
sorry

end vanessa_score_l687_68792


namespace unique_function_satisfying_conditions_l687_68722

-- Define the type of real-valued functions
def RealFunction := ℝ → ℝ

-- Define the properties of f
def IsStrictlyIncreasing (f : RealFunction) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

def HasInverse (f g : RealFunction) : Prop :=
  (∀ x : ℝ, f (g x) = x) ∧ (∀ x : ℝ, g (f x) = x)

def SatisfiesEquation (f g : RealFunction) : Prop :=
  ∀ x : ℝ, f x + g x = 2 * x

-- Main theorem
theorem unique_function_satisfying_conditions :
  ∃! f : RealFunction,
    IsStrictlyIncreasing f ∧
    (∃ g : RealFunction, HasInverse f g ∧ SatisfiesEquation f g) ∧
    (∀ x : ℝ, f x = x) :=
by
  sorry

end unique_function_satisfying_conditions_l687_68722


namespace sphere_surface_area_rectangular_solid_l687_68747

theorem sphere_surface_area_rectangular_solid (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  let R := Real.sqrt ((a^2 + b^2 + c^2) / 4)
  4 * Real.pi * R^2 = 50 * Real.pi := by
  sorry

end sphere_surface_area_rectangular_solid_l687_68747


namespace quadratic_completion_l687_68798

theorem quadratic_completion (b : ℝ) (n : ℝ) : 
  (∀ x, x^2 + b*x + (1/5 : ℝ) = (x + n)^2 + (1/20 : ℝ)) → b < 0 → b = -Real.sqrt (3/5)
:= by sorry

end quadratic_completion_l687_68798


namespace plane_equation_correct_l687_68749

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Parametric representation of a plane -/
def parametricPlane (s t : ℝ) : Point3D :=
  { x := 2 + 2*s - t
    y := 4 - 2*s
    z := 5 - 3*s + 3*t }

/-- Check if a point lies on a plane -/
def pointOnPlane (plane : Plane) (point : Point3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The plane equation we want to prove -/
def targetPlane : Plane :=
  { a := 2
    b := 1
    c := -1
    d := -3 }

theorem plane_equation_correct :
  ∀ s t : ℝ, pointOnPlane targetPlane (parametricPlane s t) := by
  sorry

end plane_equation_correct_l687_68749


namespace square_root_16_l687_68732

theorem square_root_16 (x : ℝ) : (x + 1)^2 = 16 → x = 3 ∨ x = -5 := by
  sorry

end square_root_16_l687_68732


namespace abcdefg_over_defghij_l687_68773

theorem abcdefg_over_defghij (a b c d e f g h i j : ℚ)
  (h1 : a / b = -7 / 3)
  (h2 : b / c = -5 / 2)
  (h3 : c / d = 2)
  (h4 : d / e = -3 / 2)
  (h5 : e / f = 4 / 3)
  (h6 : f / g = -1 / 4)
  (h7 : g / h = 3 / -5)
  (h8 : i ≠ 0) -- Additional hypothesis to avoid division by zero
  : a * b * c * d * e * f * g / (d * e * f * g * h * i * j) = (-21 / 16) * (c / i) := by
  sorry

end abcdefg_over_defghij_l687_68773


namespace equal_roots_quadratic_l687_68708

theorem equal_roots_quadratic (p : ℝ) : 
  (∃! p, ∀ x, x^2 - (p+1)*x + p = 0 → (∃! r, x = r)) := by
  sorry

end equal_roots_quadratic_l687_68708


namespace exact_sunny_days_probability_l687_68750

def num_days : ℕ := 5
def sunny_prob : ℚ := 2/5
def desired_sunny_days : ℕ := 2

theorem exact_sunny_days_probability :
  (num_days.choose desired_sunny_days : ℚ) * sunny_prob ^ desired_sunny_days * (1 - sunny_prob) ^ (num_days - desired_sunny_days) = 4320/15625 := by
  sorry

end exact_sunny_days_probability_l687_68750


namespace triangle_altitude_after_base_extension_l687_68707

theorem triangle_altitude_after_base_extension (area : ℝ) (new_base : ℝ) (h : area = 800) (h_base : new_base = 50) :
  let new_altitude := 2 * area / new_base
  new_altitude = 32 := by
sorry

end triangle_altitude_after_base_extension_l687_68707


namespace quadratic_roots_theorem_l687_68740

-- Define the quadratic equations
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x < 0 ∧ y < 0 ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Define the set (-2,0) ∪ (1,3)
def target_set (m : ℝ) : Prop := (m > -2 ∧ m < 0) ∨ (m > 1 ∧ m < 3)

-- State the theorem
theorem quadratic_roots_theorem (m : ℝ) :
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) ↔ target_set m :=
sorry

end quadratic_roots_theorem_l687_68740


namespace ribbon_pieces_l687_68785

theorem ribbon_pieces (original_length : ℝ) (piece_length : ℝ) (remaining_length : ℝ) : 
  original_length = 51 →
  piece_length = 0.15 →
  remaining_length = 36 →
  (original_length - remaining_length) / piece_length = 100 := by
sorry

end ribbon_pieces_l687_68785


namespace pairing_probability_l687_68736

/-- Represents a student in the classroom -/
structure Student :=
  (name : String)

/-- The probability of a specific event occurring in a random pairing scenario -/
def probability_of_pairing (total_students : ℕ) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / (total_students - 1)

/-- The classroom setup -/
def classroom_setup : Prop :=
  ∃ (students : Finset Student) (margo irma jess kurt : Student),
    students.card = 50 ∧
    margo ∈ students ∧
    irma ∈ students ∧
    jess ∈ students ∧
    kurt ∈ students ∧
    margo ≠ irma ∧ margo ≠ jess ∧ margo ≠ kurt

theorem pairing_probability (h : classroom_setup) :
  probability_of_pairing 50 3 = 3 / 49 := by
  sorry

end pairing_probability_l687_68736


namespace orange_selling_price_l687_68771

/-- Proves that the selling price of each orange is 60 cents given the conditions -/
theorem orange_selling_price (total_cost : ℚ) (num_oranges : ℕ) (profit_per_orange : ℚ) :
  total_cost = 25 / 2 →
  num_oranges = 25 →
  profit_per_orange = 1 / 10 →
  (total_cost / num_oranges + profit_per_orange) * 100 = 60 := by
  sorry

end orange_selling_price_l687_68771


namespace trisha_walk_distance_l687_68765

/-- The total distance Trisha walked during her vacation in New York City -/
def total_distance (hotel_to_postcard postcard_to_tshirt tshirt_to_hotel : ℝ) : ℝ :=
  hotel_to_postcard + postcard_to_tshirt + tshirt_to_hotel

/-- Theorem stating that Trisha walked 0.89 miles in total -/
theorem trisha_walk_distance :
  total_distance 0.11 0.11 0.67 = 0.89 := by
  sorry

end trisha_walk_distance_l687_68765


namespace equation_solution_l687_68714

theorem equation_solution : 
  {x : ℝ | ∃ (a b : ℝ), a^4 = 59 - 2*x ∧ b^4 = 23 + 2*x ∧ a + b = 4} = {-8, 29} := by
  sorry

end equation_solution_l687_68714


namespace car_tire_usage_l687_68748

/-- Represents the usage of tires on a car -/
structure TireUsage where
  total_tires : ℕ
  active_tires : ℕ
  total_miles : ℕ
  miles_per_tire : ℕ

/-- Calculates the miles each tire is used given the total miles driven and number of tires -/
def calculate_miles_per_tire (usage : TireUsage) : Prop :=
  usage.miles_per_tire = usage.total_miles * usage.active_tires / usage.total_tires

/-- Theorem stating that for a car with 5 tires, 4 of which are used at any time, 
    each tire is used for 40,000 miles over a total of 50,000 miles driven -/
theorem car_tire_usage :
  ∀ (usage : TireUsage), 
    usage.total_tires = 5 →
    usage.active_tires = 4 →
    usage.total_miles = 50000 →
    calculate_miles_per_tire usage →
    usage.miles_per_tire = 40000 :=
sorry

end car_tire_usage_l687_68748


namespace no_dapper_numbers_l687_68767

/-- A two-digit positive integer is 'dapper' if it equals the sum of its nonzero tens digit and the cube of its units digit. -/
def is_dapper (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ ∃ (a b : ℕ), n = 10 * a + b ∧ a ≠ 0 ∧ n = a + b^3

/-- There are no two-digit positive integers that are 'dapper'. -/
theorem no_dapper_numbers : ¬∃ (n : ℕ), is_dapper n := by
  sorry

#check no_dapper_numbers

end no_dapper_numbers_l687_68767


namespace distinct_prime_factors_30_factorial_l687_68758

/-- The number of distinct prime factors of 30! -/
def num_distinct_prime_factors_30_factorial : ℕ := 10

/-- Theorem stating that the number of distinct prime factors of 30! is 10 -/
theorem distinct_prime_factors_30_factorial :
  num_distinct_prime_factors_30_factorial = 10 := by sorry

end distinct_prime_factors_30_factorial_l687_68758


namespace sum_of_special_numbers_l687_68777

/-- A function that returns the number of divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a number ends with 8 zeros -/
def ends_with_8_zeros (n : ℕ) : Prop := sorry

/-- The theorem to be proved -/
theorem sum_of_special_numbers :
  ∃ (a b : ℕ), a ≠ b ∧
    ends_with_8_zeros a ∧
    ends_with_8_zeros b ∧
    num_divisors a = 90 ∧
    num_divisors b = 90 ∧
    a + b = 700000000 := by sorry

end sum_of_special_numbers_l687_68777


namespace quadratic_roots_l687_68715

theorem quadratic_roots (d : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + d = 0 ↔ x = (3 + Real.sqrt d) / 2 ∨ x = (3 - Real.sqrt d) / 2) →
  d = 9/5 := by
sorry

end quadratic_roots_l687_68715


namespace sin_four_arcsin_one_fourth_l687_68746

theorem sin_four_arcsin_one_fourth :
  Real.sin (4 * Real.arcsin (1/4)) = 7 * Real.sqrt 15 / 32 := by
  sorry

end sin_four_arcsin_one_fourth_l687_68746


namespace solution_set_for_a_eq_one_range_of_a_l687_68757

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| - |x + 1|

-- Part I
theorem solution_set_for_a_eq_one :
  {x : ℝ | f 1 x ≤ x^2 - x} = {x : ℝ | x ≤ -1 ∨ x ≥ 0} :=
sorry

-- Part II
theorem range_of_a (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : 2*m + n = 1) :
  (∀ x, f a x ≤ 1/m + 2/n) → -9 ≤ a ∧ a ≤ 7 :=
sorry

end solution_set_for_a_eq_one_range_of_a_l687_68757


namespace quadratic_real_roots_l687_68790

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, (k + 1) * x^2 + 4 * x - 1 = 0) ↔ (k ≥ -5 ∧ k ≠ -1) := by
sorry

end quadratic_real_roots_l687_68790


namespace quadratic_inequality_solution_set_l687_68768

/-- Given that the solution set of x² - ax - b < 0 is (2, 3), 
    prove that the solution set of bx² - ax - 1 > 0 is (-1/2, -1/3) -/
theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : ∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) :
  ∀ x, b*x^2 - a*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3 := by
  sorry


end quadratic_inequality_solution_set_l687_68768


namespace no_odd_pieces_all_diagonals_black_squares_count_equivalence_l687_68710

/-- Represents a chess piece on a chessboard --/
structure ChessPiece where
  position : Nat × Nat
  color : Bool

/-- Represents a chessboard with pieces --/
def Chessboard := List ChessPiece

/-- Represents a diagonal on a chessboard --/
inductive Diagonal
| A1H8 : Nat → Diagonal  -- Diagonals parallel to a1-h8
| A8H1 : Nat → Diagonal  -- Diagonals parallel to a8-h1

/-- Returns the number of pieces on a given diagonal --/
def piecesOnDiagonal (board : Chessboard) (diag : Diagonal) : Nat :=
  sorry

/-- Checks if a number is odd --/
def isOdd (n : Nat) : Bool :=
  n % 2 = 1

/-- Main theorem: It's impossible to have an odd number of pieces on all 30 diagonals --/
theorem no_odd_pieces_all_diagonals (board : Chessboard) : 
  ¬(∀ (d : Diagonal), isOdd (piecesOnDiagonal board d)) :=
by
  sorry

/-- Helper function to count pieces on black squares along a1-h8 diagonals --/
def countBlackSquaresA1H8 (board : Chessboard) : Nat :=
  sorry

/-- Helper function to count pieces on black squares along a8-h1 diagonals --/
def countBlackSquaresA8H1 (board : Chessboard) : Nat :=
  sorry

/-- Theorem: The two ways of counting pieces on black squares are equivalent --/
theorem black_squares_count_equivalence (board : Chessboard) :
  countBlackSquaresA1H8 board = countBlackSquaresA8H1 board :=
by
  sorry

end no_odd_pieces_all_diagonals_black_squares_count_equivalence_l687_68710


namespace coin_toss_probability_l687_68731

/-- The number of coin tosses -/
def n : ℕ := 5

/-- The number of heads -/
def k : ℕ := 4

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
def binomial_probability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1/2)^n

theorem coin_toss_probability :
  binomial_probability n k = 5/32 := by
  sorry

end coin_toss_probability_l687_68731


namespace triangle_inequalities_l687_68788

theorem triangle_inequalities (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  let p := (a + b + c) / 2
  (2 * (a + b + c) * (a^2 + b^2 + c^2) ≥ 3 * (a^3 + b^3 + c^3 + 3 * a * b * c)) ∧
  ((a + b + c)^3 ≤ 5 * (b * c * (b + c) + c * a * (c + a) + a * b * (a + b)) - 3 * a * b * c) ∧
  (a * b * c < a^2 * (p - a) + b^2 * (p - b) + c^2 * (p - c) ∧
    a^2 * (p - a) + b^2 * (p - b) + c^2 * (p - c) ≤ 3/2 * a * b * c) ∧
  (1 < Real.cos (π - Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) +
       Real.cos (π - Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) +
       Real.cos (π - Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) ∧
   Real.cos (π - Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) +
       Real.cos (π - Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) +
       Real.cos (π - Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) ≤ 3/2) := by
  sorry

end triangle_inequalities_l687_68788


namespace two_thousand_thirteenth_underlined_pair_l687_68795

/-- The sequence of n values where n and 3^n have the same units digit -/
def underlined_sequence : ℕ → ℕ
| 0 => 1
| n + 1 => underlined_sequence n + 2

/-- The nth pair in the sequence of underlined pairs -/
def nth_underlined_pair (n : ℕ) : ℕ × ℕ :=
  let m := underlined_sequence (n - 1)
  (m, 3^m)

theorem two_thousand_thirteenth_underlined_pair :
  nth_underlined_pair 2013 = (4025, 3^4025) := by
  sorry

end two_thousand_thirteenth_underlined_pair_l687_68795


namespace f_strictly_increasing_l687_68717

-- Define the function f(x) = x³ + x
def f (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem f_strictly_increasing : StrictMono f := by sorry

end f_strictly_increasing_l687_68717


namespace midpoint_x_coordinate_sum_l687_68763

theorem midpoint_x_coordinate_sum (a b c : ℝ) :
  let vertex_sum := a + b + c
  let midpoint1 := (a + b) / 2
  let midpoint2 := (a + c) / 2
  let midpoint3 := (b + c) / 2
  midpoint1 + midpoint2 + midpoint3 = vertex_sum := by
sorry

end midpoint_x_coordinate_sum_l687_68763


namespace pie_remainder_pie_problem_l687_68730

theorem pie_remainder (carlos_portion : Real) (maria_fraction : Real) : Real :=
  let remaining_after_carlos := 1 - carlos_portion
  let maria_portion := maria_fraction * remaining_after_carlos
  let final_remainder := remaining_after_carlos - maria_portion
  
  final_remainder

theorem pie_problem :
  pie_remainder 0.6 0.25 = 0.3 := by
  sorry

end pie_remainder_pie_problem_l687_68730


namespace distribute_5_3_l687_68741

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The theorem stating that there are 5 ways to distribute 5 indistinguishable balls into 3 indistinguishable boxes -/
theorem distribute_5_3 : distribute 5 3 = 5 := by
  sorry

end distribute_5_3_l687_68741


namespace functional_equation_implies_linearity_l687_68775

theorem functional_equation_implies_linearity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x^2 - f x * f y + f y^2)) :
  ∀ x : ℝ, f (2005 * x) = 2005 * f x :=
sorry

end functional_equation_implies_linearity_l687_68775


namespace p_neither_sufficient_nor_necessary_l687_68799

-- Define the propositions
def p (m : ℝ) : Prop := m = -1
def q (m : ℝ) : Prop := ∀ (x y : ℝ), (x - 1 = 0) ∧ (x + m^2 * y = 0) → 
  (∃ (k : ℝ), k ≠ 0 ∧ (1 * k = m^2) ∨ (1 * m^2 = -k))

-- Theorem statement
theorem p_neither_sufficient_nor_necessary :
  (∃ m : ℝ, p m ∧ ¬q m) ∧ (∃ m : ℝ, q m ∧ ¬p m) :=
sorry

end p_neither_sufficient_nor_necessary_l687_68799


namespace range_of_g_l687_68751

theorem range_of_g (x : ℝ) : 3/4 ≤ Real.cos x ^ 4 + Real.sin x ^ 2 ∧ Real.cos x ^ 4 + Real.sin x ^ 2 ≤ 1 := by
  sorry

end range_of_g_l687_68751


namespace construction_materials_cost_l687_68719

/-- The total cost of construction materials for Mr. Zander -/
def total_cost (cement_bags : ℕ) (cement_price : ℕ) (sand_lorries : ℕ) (sand_tons_per_lorry : ℕ) (sand_price : ℕ) : ℕ :=
  cement_bags * cement_price + sand_lorries * sand_tons_per_lorry * sand_price

/-- Theorem stating that the total cost of construction materials for Mr. Zander is $13,000 -/
theorem construction_materials_cost :
  total_cost 500 10 20 10 40 = 13000 := by
  sorry

end construction_materials_cost_l687_68719


namespace sum_of_fractions_l687_68742

theorem sum_of_fractions : 
  (1 : ℚ) / 3 + (1 : ℚ) / 2 + (-5 : ℚ) / 6 + (1 : ℚ) / 5 + (1 : ℚ) / 4 + (-9 : ℚ) / 20 + (-9 : ℚ) / 20 = (-9 : ℚ) / 20 := by
  sorry

end sum_of_fractions_l687_68742


namespace trajectory_and_m_value_l687_68752

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 7

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 7

-- Define the line that intersects C
def intersecting_line (x y m : ℝ) : Prop := x + y - m = 0

-- Define the property that a circle passes through the origin
def circle_through_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem trajectory_and_m_value :
  ∀ (x₀ y₀ x y x₁ y₁ x₂ y₂ m : ℝ),
  (3/2, 0) = ((x₀ + x)/2, (y₀ + y)/2) →  -- A is midpoint of BM
  circle_O x₀ y₀ →  -- B is on circle O
  trajectory_C x y →  -- M is on trajectory C
  intersecting_line x₁ y₁ m →  -- P is on the intersecting line
  intersecting_line x₂ y₂ m →  -- Q is on the intersecting line
  trajectory_C x₁ y₁ →  -- P is on trajectory C
  trajectory_C x₂ y₂ →  -- Q is on trajectory C
  circle_through_origin x₁ y₁ x₂ y₂ →  -- Circle with PQ as diameter passes through origin
  (∀ x y, trajectory_C x y ↔ (x - 3)^2 + y^2 = 7) ∧  -- Trajectory equation is correct
  (m = 1 ∨ m = 2)  -- m value is correct
  := by sorry

end trajectory_and_m_value_l687_68752


namespace parallel_vectors_sin_cos_product_l687_68778

/-- 
Given two vectors in the plane, a = (4, 3) and b = (sin α, cos α),
prove that if a is parallel to b, then sin α * cos α = 12/25.
-/
theorem parallel_vectors_sin_cos_product (α : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ (4, 3) = k • (Real.sin α, Real.cos α)) → 
  Real.sin α * Real.cos α = 12/25 := by
sorry

end parallel_vectors_sin_cos_product_l687_68778


namespace train_length_l687_68754

/-- Calculates the length of a train given its speed and the time and distance it takes to cross a platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 * (5/18) → 
  platform_length = 250 → 
  crossing_time = 30 → 
  (train_speed * crossing_time) - platform_length = 350 := by
  sorry

#check train_length

end train_length_l687_68754


namespace fourth_score_proof_l687_68734

/-- Given four test scores with an average of 94, where three scores are known to be 85, 100, and 94,
    prove that the fourth score must be 97. -/
theorem fourth_score_proof (score1 score2 score3 score4 : ℕ) : 
  score1 = 85 → score2 = 100 → score3 = 94 → 
  (score1 + score2 + score3 + score4) / 4 = 94 →
  score4 = 97 := by sorry

end fourth_score_proof_l687_68734


namespace two_identical_digits_in_2_pow_30_l687_68727

theorem two_identical_digits_in_2_pow_30 :
  ∃ (d : ℕ) (i j : ℕ), i ≠ j ∧ i < 10 ∧ j < 10 ∧
  (2^30 / 10^i) % 10 = d ∧ (2^30 / 10^j) % 10 = d :=
by
  have h1 : 2^30 > 10^9 := sorry
  have h2 : 2^30 < 8 * 10^9 := sorry
  have pigeonhole : ∀ (n m : ℕ), n > m → 
    ∃ (k : ℕ), k < n ∧ (∃ (i j : ℕ), i < m ∧ j < m ∧ i ≠ j ∧
    (n / 10^i) % 10 = k ∧ (n / 10^j) % 10 = k) := sorry
  sorry


end two_identical_digits_in_2_pow_30_l687_68727


namespace cos_arcsin_eight_seventeenths_l687_68700

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8 / 17)) = 15 / 17 := by
  sorry

end cos_arcsin_eight_seventeenths_l687_68700


namespace smallest_denominator_between_fractions_l687_68787

theorem smallest_denominator_between_fractions : 
  ∃ (p q : ℕ), 
    (1 : ℚ) / 2014 < (p : ℚ) / q ∧ 
    (p : ℚ) / q < (1 : ℚ) / 2013 ∧
    q = 4027 ∧
    (∀ (p' q' : ℕ), (1 : ℚ) / 2014 < (p' : ℚ) / q' → (p' : ℚ) / q' < (1 : ℚ) / 2013 → q ≤ q') :=
by sorry

end smallest_denominator_between_fractions_l687_68787


namespace largest_x_floor_div_l687_68720

theorem largest_x_floor_div : ∃ (x : ℝ), x = 63/8 ∧ 
  (∀ (y : ℝ), y > x → ⌊y⌋/y ≠ 8/9) ∧ ⌊x⌋/x = 8/9 := by
  sorry

end largest_x_floor_div_l687_68720


namespace pool_cleaning_tip_percentage_l687_68760

/-- Calculates the tip percentage for pool cleaning sessions -/
theorem pool_cleaning_tip_percentage
  (days_between_cleanings : ℕ)
  (cost_per_cleaning : ℕ)
  (chemical_cost : ℕ)
  (chemical_frequency : ℕ)
  (total_monthly_cost : ℕ)
  (days_in_month : ℕ := 30)  -- Assumption from the problem
  (h1 : days_between_cleanings = 3)
  (h2 : cost_per_cleaning = 150)
  (h3 : chemical_cost = 200)
  (h4 : chemical_frequency = 2)
  (h5 : total_monthly_cost = 2050)
  : (total_monthly_cost - (days_in_month / days_between_cleanings * cost_per_cleaning + chemical_frequency * chemical_cost)) / (days_in_month / days_between_cleanings * cost_per_cleaning) * 100 = 10 :=
by sorry

end pool_cleaning_tip_percentage_l687_68760


namespace tree_height_after_two_years_l687_68713

/-- The height of a tree after a given number of years, given that it triples its height every year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

/-- Theorem stating that if a tree reaches 81 feet after 4 years of tripling its height annually, 
    then its height after 2 years is 9 feet -/
theorem tree_height_after_two_years 
  (h : ∃ initial_height : ℝ, tree_height initial_height 4 = 81) : 
  ∃ initial_height : ℝ, tree_height initial_height 2 = 9 :=
sorry

end tree_height_after_two_years_l687_68713


namespace p_difference_qr_l687_68784

theorem p_difference_qr (p q r : ℕ) : 
  p = 56 → 
  q = p / 8 →
  r = p / 8 →
  p - (q + r) = 42 := by
sorry

end p_difference_qr_l687_68784


namespace geometric_sequence_fourth_term_l687_68704

theorem geometric_sequence_fourth_term 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : a = 1024) 
  (h2 : a * r^5 = 32) : 
  a * r^3 = 128 := by
sorry

end geometric_sequence_fourth_term_l687_68704


namespace sum_of_distances_l687_68709

/-- Given points A, B, and D in a coordinate plane, prove that the sum of distances AD and BD is 2√5 + √130 -/
theorem sum_of_distances (A B D : ℝ × ℝ) : 
  A = (15, 0) → B = (0, 5) → D = (4, 3) → 
  Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 2 * Real.sqrt 5 + Real.sqrt 130 := by
  sorry

end sum_of_distances_l687_68709


namespace functional_equation_solution_l687_68721

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + (x + 1/2) * f (1 - x) = 1

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
    (f 0 = 2 ∧ f 1 = -2) ∧
    (∀ x ≠ 1/2, f x = 2 / (1 - 2*x)) ∧
    f (1/2) = 1/2 := by
  sorry

end functional_equation_solution_l687_68721


namespace division_problem_l687_68766

theorem division_problem : (786 * 74) / 30 = 1938.8 := by sorry

end division_problem_l687_68766


namespace quadratic_factorization_l687_68712

theorem quadratic_factorization (y : ℝ) : 16 * y^2 - 40 * y + 25 = (4 * y - 5)^2 := by
  sorry

end quadratic_factorization_l687_68712


namespace arithmetic_sequence_sum_l687_68725

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 1 = 2 → a 2 + a 5 = 13 → a 5 + a 6 + a 7 = 33 := by
  sorry

end arithmetic_sequence_sum_l687_68725


namespace triangle_inequality_check_l687_68789

theorem triangle_inequality_check (rods : Fin 100 → ℝ) 
  (h_sorted : ∀ i j : Fin 100, i ≤ j → rods i ≤ rods j) :
  (∀ i j k : Fin 100, i ≠ j ∧ j ≠ k ∧ i ≠ k → 
    rods i + rods j > rods k) ↔ 
  (rods 98 + rods 99 > rods 100) :=
sorry

end triangle_inequality_check_l687_68789


namespace max_type_a_mascots_l687_68781

/-- Represents the mascot types -/
inductive MascotType
| A
| B

/-- Represents the mascot purchase scenario -/
structure MascotPurchase where
  totalMascots : ℕ
  totalCost : ℕ
  unitPriceA : ℕ
  unitPriceB : ℕ
  newBudget : ℕ
  newTotalMascots : ℕ

/-- Conditions for the mascot purchase -/
def validMascotPurchase (mp : MascotPurchase) : Prop :=
  mp.totalMascots = 110 ∧
  mp.totalCost = 6000 ∧
  mp.unitPriceA = (6 * mp.unitPriceB) / 5 ∧
  mp.totalCost = mp.totalMascots / 2 * (mp.unitPriceA + mp.unitPriceB) ∧
  mp.newBudget = 16800 ∧
  mp.newTotalMascots = 300

/-- Theorem: The maximum number of type A mascots that can be purchased in the second round is 180 -/
theorem max_type_a_mascots (mp : MascotPurchase) (h : validMascotPurchase mp) :
  ∀ n : ℕ, n ≤ mp.newTotalMascots → n * mp.unitPriceA + (mp.newTotalMascots - n) * mp.unitPriceB ≤ mp.newBudget →
  n ≤ 180 :=
sorry

end max_type_a_mascots_l687_68781


namespace calculate_expression_l687_68770

theorem calculate_expression : 
  75 * (4 + 1/3 - (5 + 1/4)) / (3 + 1/2 + 2 + 1/5) = -5/31 := by
  sorry

end calculate_expression_l687_68770


namespace farm_field_problem_l687_68705

/-- Represents the problem of calculating the farm field area and initial work plan --/
theorem farm_field_problem (planned_daily_rate : ℕ) (actual_daily_rate : ℕ) (extra_days : ℕ) (area_left : ℕ) 
  (h1 : planned_daily_rate = 90)
  (h2 : actual_daily_rate = 85)
  (h3 : extra_days = 2)
  (h4 : area_left = 40) :
  ∃ (total_area : ℕ) (initial_days : ℕ),
    total_area = 3780 ∧ 
    initial_days = 42 ∧
    planned_daily_rate * initial_days = total_area ∧
    actual_daily_rate * (initial_days + extra_days) + area_left = total_area :=
by
  sorry

end farm_field_problem_l687_68705


namespace three_in_range_of_g_l687_68769

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

-- Theorem statement
theorem three_in_range_of_g (a : ℝ) : ∃ x : ℝ, g a x = 3 := by
  sorry

end three_in_range_of_g_l687_68769


namespace sum_of_roots_is_18_l687_68728

def is_symmetric_about_3 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (3 + x) = f (3 - x)

def has_six_distinct_roots_in_arithmetic_sequence (f : ℝ → ℝ) : Prop :=
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ) (d : ℝ),
    r₁ < r₂ ∧ r₂ < r₃ ∧ r₃ < r₄ ∧ r₄ < r₅ ∧ r₅ < r₆ ∧
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0 ∧ f r₄ = 0 ∧ f r₅ = 0 ∧ f r₆ = 0 ∧
    r₂ - r₁ = d ∧ r₃ - r₂ = d ∧ r₄ - r₃ = d ∧ r₅ - r₄ = d ∧ r₆ - r₅ = d

theorem sum_of_roots_is_18 (f : ℝ → ℝ) 
    (h_sym : is_symmetric_about_3 f)
    (h_roots : has_six_distinct_roots_in_arithmetic_sequence f) :
    ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ),
      f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0 ∧ f r₄ = 0 ∧ f r₅ = 0 ∧ f r₆ = 0 ∧
      r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 18 :=
by
  sorry

end sum_of_roots_is_18_l687_68728


namespace quadratic_form_completion_constant_term_value_l687_68733

theorem quadratic_form_completion (x : ℝ) : 
  x^2 - 6*x = (x - 3)^2 - 9 :=
sorry

theorem constant_term_value : 
  ∃ k, ∀ x, x^2 - 6*x = (x - 3)^2 + k ∧ k = -9 :=
sorry

end quadratic_form_completion_constant_term_value_l687_68733


namespace parallel_line_slope_l687_68738

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Define the slope of a line given its equation
def slope_of_line (f : ℝ → ℝ → Prop) : ℝ :=
  sorry

-- Theorem: The slope of a line parallel to 3x - 6y = 12 is 1/2
theorem parallel_line_slope :
  slope_of_line line_equation = 1/2 := by sorry

end parallel_line_slope_l687_68738


namespace gracies_height_l687_68702

/-- Given the heights of Griffin, Grayson, and Gracie, prove Gracie's height -/
theorem gracies_height 
  (griffin_height : ℕ) 
  (grayson_height : ℕ) 
  (gracie_height : ℕ)
  (h1 : griffin_height = 61)
  (h2 : grayson_height = griffin_height + 2)
  (h3 : gracie_height = grayson_height - 7) : 
  gracie_height = 56 := by sorry

end gracies_height_l687_68702


namespace simple_interest_problem_l687_68743

theorem simple_interest_problem (interest : ℚ) (rate : ℚ) (time : ℚ) (principal : ℚ) : 
  interest = 4016.25 →
  rate = 9 / 100 →
  time = 5 →
  principal = 8925 →
  interest = principal * rate * time :=
by sorry

end simple_interest_problem_l687_68743
