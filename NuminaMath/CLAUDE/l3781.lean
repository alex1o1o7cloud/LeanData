import Mathlib

namespace right_angled_triangle_check_l3781_378139

theorem right_angled_triangle_check : 
  let a : ℝ := Real.sqrt 2
  let b : ℝ := Real.sqrt 3
  let c : ℝ := Real.sqrt 5
  (a * a + b * b = c * c) ∧ 
  ¬(1 * 1 + 1 * 1 = Real.sqrt 3 * Real.sqrt 3) ∧
  ¬(0.2 * 0.2 + 0.3 * 0.3 = 0.5 * 0.5) ∧
  ¬((1/3) * (1/3) + (1/4) * (1/4) = (1/5) * (1/5)) := by
  sorry

end right_angled_triangle_check_l3781_378139


namespace division_relation_l3781_378151

theorem division_relation (a b c : ℚ) 
  (h1 : a / b = 2) 
  (h2 : b / c = 3/4) : 
  c / a = 2/3 := by
  sorry

end division_relation_l3781_378151


namespace negation_of_proposition_negation_of_2n_gt_sqrt_n_l3781_378194

theorem negation_of_proposition (p : ℕ → Prop) : 
  (¬∀ n, p n) ↔ (∃ n, ¬p n) :=
by sorry

theorem negation_of_2n_gt_sqrt_n :
  (¬∀ n : ℕ, 2^n > Real.sqrt n) ↔ (∃ n : ℕ, 2^n ≤ Real.sqrt n) :=
by sorry

end negation_of_proposition_negation_of_2n_gt_sqrt_n_l3781_378194


namespace completing_square_result_l3781_378157

theorem completing_square_result (x : ℝ) : 
  (x^2 - 4*x - 1 = 0) ↔ ((x - 2)^2 = 5) :=
by sorry

end completing_square_result_l3781_378157


namespace point_C_coordinates_main_theorem_l3781_378141

-- Define points A, B, and C in ℝ²
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (13, 9)
def C : ℝ × ℝ := (19, 12)

-- Define the vector from A to B
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define the vector from B to C
def BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)

-- Theorem stating that C is the correct point
theorem point_C_coordinates : 
  BC = (1/2 : ℝ) • AB := by sorry

-- Main theorem to prove
theorem main_theorem : C = (19, 12) := by sorry

end point_C_coordinates_main_theorem_l3781_378141


namespace largest_divisible_n_l3781_378125

theorem largest_divisible_n : ∃ (n : ℕ), n > 0 ∧ (n + 12) ∣ (n^3 + 105) ∧ 
  ∀ (m : ℕ), m > n → m > 0 → ¬((m + 12) ∣ (m^3 + 105)) :=
by sorry

end largest_divisible_n_l3781_378125


namespace largest_three_digit_congruence_l3781_378132

theorem largest_three_digit_congruence :
  ∀ n : ℕ,
  100 ≤ n ∧ n ≤ 999 ∧ (75 * n) % 300 = 225 →
  n ≤ 999 ∧
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (75 * m) % 300 = 225 → m ≤ n) :=
by sorry

end largest_three_digit_congruence_l3781_378132


namespace frame_price_ratio_l3781_378102

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


end frame_price_ratio_l3781_378102


namespace max_value_xyz_l3781_378111

theorem max_value_xyz (x y z : ℝ) (h1 : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) 
  (h2 : x + y + z = 1) (h3 : x^2 + y^2 + z^2 = 1) : 
  x + y^3 + z^4 ≤ 1 := by
  sorry

end max_value_xyz_l3781_378111


namespace equation_solution_l3781_378184

/-- Custom multiplication operation for real numbers -/
def custom_mul (a b : ℝ) : ℝ := a^2 - b^2

/-- Theorem stating the solution to the equation -/
theorem equation_solution :
  ∃ x : ℝ, custom_mul (x + 2) 5 = (x - 5) * (5 + x) ∧ x = -1 := by
  sorry

end equation_solution_l3781_378184


namespace f_increasing_l3781_378153

def f (x : ℝ) := 3 * x + 2

theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end f_increasing_l3781_378153


namespace derivative_zero_sufficient_not_necessary_for_extremum_l3781_378105

-- Define a differentiable function f
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define the property of having an extremum at a point
def HasExtremumAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - x₀| < ε → f x ≤ f x₀ ∨ ∀ x, |x - x₀| < ε → f x ≥ f x₀

-- State the theorem
theorem derivative_zero_sufficient_not_necessary_for_extremum :
  (∃ x₀ : ℝ, deriv f x₀ = 0 → HasExtremumAt f x₀) ∧
  (∃ x₀ : ℝ, HasExtremumAt f x₀ ∧ deriv f x₀ ≠ 0) := by
  sorry

end derivative_zero_sufficient_not_necessary_for_extremum_l3781_378105


namespace inscribed_rectangle_area_l3781_378107

theorem inscribed_rectangle_area (square_area : ℝ) (ratio : ℝ) 
  (h_square_area : square_area = 18)
  (h_ratio : ratio = 2)
  (h_positive : square_area > 0) :
  let square_side := Real.sqrt square_area
  let rect_short_side := 2 * square_side / (ratio + 1 + Real.sqrt (ratio^2 + 1))
  let rect_long_side := ratio * rect_short_side
  rect_short_side * rect_long_side = 8 := by
  sorry


end inscribed_rectangle_area_l3781_378107


namespace a_is_negative_l3781_378176

theorem a_is_negative (a b : ℤ) (h1 : a ≠ 0) (h2 : ∃ k : ℤ, 3 + a + b^2 = 6*a*k) : a < 0 := by
  sorry

end a_is_negative_l3781_378176


namespace remaining_half_speed_l3781_378101

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


end remaining_half_speed_l3781_378101


namespace unique_divisibility_function_l3781_378169

/-- A function from positive integers to positive integers -/
def NatFunction := ℕ+ → ℕ+

/-- The property that f(m) + f(n) divides m + n for all m and n -/
def HasDivisibilityProperty (f : NatFunction) : Prop :=
  ∀ m n : ℕ+, (f m + f n) ∣ (m + n)

/-- The identity function on positive integers -/
def identityFunction : NatFunction := fun x => x

/-- Theorem stating that the identity function is the only function satisfying the divisibility property -/
theorem unique_divisibility_function :
  ∀ f : NatFunction, HasDivisibilityProperty f ↔ f = identityFunction := by
  sorry

end unique_divisibility_function_l3781_378169


namespace perpendicular_line_angle_of_inclination_l3781_378148

theorem perpendicular_line_angle_of_inclination 
  (line_eq : ℝ → ℝ → Prop) 
  (h_line_eq : ∀ x y, line_eq x y ↔ x + Real.sqrt 3 * y + 2 = 0) :
  ∃ θ : ℝ, 
    0 ≤ θ ∧ 
    θ < π ∧ 
    (∀ x y, line_eq x y → 
      ∃ m : ℝ, m * Real.tan θ = -1 ∧ 
      ∀ x' y', y' - y = m * (x' - x)) ∧ 
    θ = π / 3 :=
sorry

end perpendicular_line_angle_of_inclination_l3781_378148


namespace hoopit_hands_l3781_378129

/-- Represents the number of toes on each hand of a Hoopit -/
def hoopit_toes_per_hand : ℕ := 3

/-- Represents the number of toes on each hand of a Neglart -/
def neglart_toes_per_hand : ℕ := 2

/-- Represents the number of hands each Neglart has -/
def neglart_hands : ℕ := 5

/-- Represents the number of Hoopit students on the bus -/
def hoopit_students : ℕ := 7

/-- Represents the number of Neglart students on the bus -/
def neglart_students : ℕ := 8

/-- Represents the total number of toes on the bus -/
def total_toes : ℕ := 164

/-- Theorem stating that each Hoopit has 4 hands -/
theorem hoopit_hands : 
  ∃ (h : ℕ), h = 4 ∧ 
  hoopit_students * h * hoopit_toes_per_hand + 
  neglart_students * neglart_hands * neglart_toes_per_hand = total_toes :=
sorry

end hoopit_hands_l3781_378129


namespace ceiling_negative_seven_fourths_squared_l3781_378199

theorem ceiling_negative_seven_fourths_squared : ⌈(-7/4)^2⌉ = 4 := by
  sorry

end ceiling_negative_seven_fourths_squared_l3781_378199


namespace complete_square_with_integer_l3781_378130

theorem complete_square_with_integer (y : ℝ) : ∃ k : ℤ, y^2 + 10*y + 33 = (y + 5)^2 + k := by
  sorry

end complete_square_with_integer_l3781_378130


namespace cubic_function_theorem_l3781_378164

-- Define the cubic function f(x)
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the derivative of f(x)
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_theorem (a b c : ℝ) :
  (∀ x ∈ Set.Icc (-3) 2, f a b c x > 1/c - 1/2) →
  (f' a b 1 = 0 ∧ f' a b (-2) = 0) →
  (a = 3/2 ∧ b = -6 ∧ ((3 - Real.sqrt 13)/2 < c ∧ c < 0) ∨ c > (3 + Real.sqrt 13)/2) :=
sorry

end cubic_function_theorem_l3781_378164


namespace sqrt_product_equality_l3781_378123

theorem sqrt_product_equality : Real.sqrt 3 * Real.sqrt 6 = 3 * Real.sqrt 2 := by
  sorry

end sqrt_product_equality_l3781_378123


namespace susan_walk_distance_l3781_378196

theorem susan_walk_distance (total_distance : ℝ) (difference : ℝ) :
  total_distance = 15 ∧ difference = 3 →
  ∃ susan_distance erin_distance : ℝ,
    susan_distance + erin_distance = total_distance ∧
    erin_distance = susan_distance - difference ∧
    susan_distance = 9 :=
by
  sorry

end susan_walk_distance_l3781_378196


namespace wall_passing_skill_l3781_378149

theorem wall_passing_skill (n : ℕ) : 
  8 * Real.sqrt (8 / n) = Real.sqrt (8 * (8 / n)) → n = 63 :=
by sorry

end wall_passing_skill_l3781_378149


namespace gcf_of_abc_l3781_378138

def a : ℕ := 90
def b : ℕ := 126
def c : ℕ := 180

-- The condition that c is the product of a and b divided by some integer
axiom exists_divisor : ∃ k : ℕ, k ≠ 0 ∧ c * k = a * b

-- Define the greatest common factor function
def gcf (x y z : ℕ) : ℕ := Nat.gcd x (Nat.gcd y z)

theorem gcf_of_abc : gcf a b c = 18 := by sorry

end gcf_of_abc_l3781_378138


namespace intersection_sum_l3781_378122

theorem intersection_sum (a b : ℚ) : 
  (3 = (1/3) * 4 + a) → 
  (4 = (1/2) * 3 + b) → 
  (a + b = 25/6) := by
sorry

end intersection_sum_l3781_378122


namespace hyperbola_equation_theorem_l3781_378180

/-- A hyperbola with focal length 4√3 and one branch intersected by the line y = x - 3 at two points -/
structure Hyperbola where
  /-- The focal length of the hyperbola -/
  focal_length : ℝ
  /-- The line that intersects one branch of the hyperbola at two points -/
  intersecting_line : ℝ → ℝ
  /-- Condition that the focal length is 4√3 -/
  focal_length_cond : focal_length = 4 * Real.sqrt 3
  /-- Condition that the line y = x - 3 intersects one branch at two points -/
  intersecting_line_cond : intersecting_line = fun x => x - 3

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 6 - y^2 / 6 = 1

/-- Theorem stating that the given hyperbola has the equation x²/6 - y²/6 = 1 -/
theorem hyperbola_equation_theorem (h : Hyperbola) :
  ∀ x y, hyperbola_equation h x y ↔ x^2 / 6 - y^2 / 6 = 1 :=
by sorry

end hyperbola_equation_theorem_l3781_378180


namespace counterexample_exists_l3781_378178

theorem counterexample_exists : ∃ (a b c d : ℝ), a < b ∧ c < d ∧ a * c ≥ b * d := by
  sorry

end counterexample_exists_l3781_378178


namespace triangle_area_product_l3781_378108

theorem triangle_area_product (p q : ℝ) : 
  p > 0 → q > 0 → (1/2 * (24/p) * (24/q) = 48) → p * q = 12 := by sorry

end triangle_area_product_l3781_378108


namespace preimage_of_three_one_l3781_378185

/-- The mapping f from ℝ² to ℝ² -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2*p.2, 2*p.1 - p.2)

/-- Theorem stating that (1, 1) is the pre-image of (3, 1) under f -/
theorem preimage_of_three_one :
  f (1, 1) = (3, 1) ∧ ∀ p : ℝ × ℝ, f p = (3, 1) → p = (1, 1) := by
  sorry

end preimage_of_three_one_l3781_378185


namespace max_value_theorem_l3781_378197

-- Define the line l
def line_l (y : ℝ) : Prop := y = 8

-- Define the circle C
def circle_C (x y : ℝ) : Prop := ∃ φ, x = 2 * Real.cos φ ∧ y = 2 + 2 * Real.sin φ

-- Define the ray OM
def ray_OM (θ : ℝ) : Prop := ∃ α, θ = α ∧ 0 < α ∧ α < Real.pi / 2

-- Define the ray ON
def ray_ON (θ : ℝ) : Prop := ∃ α, θ = α + Real.pi / 2 ∧ 0 < α ∧ α < Real.pi / 2

-- Define the theorem
theorem max_value_theorem :
  ∃ (OP OM OQ ON : ℝ),
    (∀ y, line_l y → ∃ x, circle_C x y) →
    (∀ θ, ray_OM θ → ∃ x y, circle_C x y) →
    (∀ θ, ray_ON θ → ∃ x y, circle_C x y) →
    (∀ α, 0 < α → α < Real.pi / 2 →
      ∃ (OP OM OQ ON : ℝ),
        (OP / OM) * (OQ / ON) ≤ 1 / 16) ∧
    (∃ α, 0 < α ∧ α < Real.pi / 2 ∧
      (OP / OM) * (OQ / ON) = 1 / 16) :=
by sorry

end max_value_theorem_l3781_378197


namespace simplify_expression_l3781_378109

theorem simplify_expression (a b : ℝ) : (a + b)^2 - a*(a + 2*b) = b^2 := by
  sorry

end simplify_expression_l3781_378109


namespace lamp_height_difference_l3781_378193

theorem lamp_height_difference (old_height new_height : Real) 
  (h1 : old_height = 1)
  (h2 : new_height = 2.33) :
  new_height - old_height = 1.33 := by
sorry

end lamp_height_difference_l3781_378193


namespace quadratic_sum_l3781_378120

/-- Given a quadratic expression 5x^2 - 20x + 8, when expressed in the form a(x - h)^2 + k,
    the sum a + h + k equals -5. -/
theorem quadratic_sum (x : ℝ) :
  ∃ (a h k : ℝ), (5 * x^2 - 20 * x + 8 = a * (x - h)^2 + k) ∧ (a + h + k = -5) := by
  sorry

end quadratic_sum_l3781_378120


namespace apartment_211_location_l3781_378156

/-- Represents a building with apartments -/
structure Building where
  total_floors : ℕ
  shop_floors : ℕ
  apartments_per_floor : ℕ

/-- Calculates the floor and entrance of an apartment -/
def apartment_location (b : Building) (apartment_number : ℕ) : ℕ × ℕ :=
  sorry

/-- The specific building in the problem -/
def problem_building : Building :=
  { total_floors := 9
  , shop_floors := 1
  , apartments_per_floor := 6 }

theorem apartment_211_location :
  apartment_location problem_building 211 = (5, 5) :=
sorry

end apartment_211_location_l3781_378156


namespace cab_journey_delay_l3781_378165

theorem cab_journey_delay (S : ℝ) (h : S > 0) : 
  let usual_time := 40
  let reduced_speed := (5/6) * S
  let new_time := usual_time * S / reduced_speed
  new_time - usual_time = 8 := by
sorry

end cab_journey_delay_l3781_378165


namespace smallest_delicious_integer_l3781_378177

/-- A delicious integer is an integer A for which there exist consecutive integers starting from A that sum to 2024. -/
def IsDelicious (A : ℤ) : Prop :=
  ∃ n : ℕ+, (n : ℤ) * (2 * A + n - 1) / 2 = 2024

/-- The smallest delicious integer is -2023. -/
theorem smallest_delicious_integer : 
  (IsDelicious (-2023) ∧ ∀ A : ℤ, A < -2023 → ¬IsDelicious A) := by
  sorry

end smallest_delicious_integer_l3781_378177


namespace power_product_reciprocals_l3781_378150

theorem power_product_reciprocals (n : ℕ) : (1 / 4 : ℝ) ^ n * 4 ^ n = 1 := by
  sorry

end power_product_reciprocals_l3781_378150


namespace gcf_lcm_360_210_l3781_378100

theorem gcf_lcm_360_210 : 
  (Nat.gcd 360 210 = 30) ∧ (Nat.lcm 360 210 = 2520) := by
sorry

end gcf_lcm_360_210_l3781_378100


namespace factorization_equality_l3781_378133

theorem factorization_equality (x y : ℝ) :
  (1 - x^2) * (1 - y^2) - 4*x*y = (x*y - 1 + x + y) * (x*y - 1 - x - y) := by
  sorry

end factorization_equality_l3781_378133


namespace simplify_expression_l3781_378110

theorem simplify_expression (x y : ℝ) : (3*x - 5*y) + (4*x + 5*y) = 7*x := by
  sorry

end simplify_expression_l3781_378110


namespace last_two_digits_of_seven_power_l3781_378146

theorem last_two_digits_of_seven_power : 7^30105 ≡ 7 [ZMOD 100] := by
  sorry

end last_two_digits_of_seven_power_l3781_378146


namespace race_problem_l3781_378183

/-- The race problem -/
theorem race_problem (total_distance : ℝ) (time_A : ℝ) (time_B : ℝ) 
  (h1 : total_distance = 70)
  (h2 : time_A = 20)
  (h3 : time_B = 25) :
  total_distance - (total_distance / time_B * time_A) = 14 := by
  sorry

end race_problem_l3781_378183


namespace sum_divisible_by_ten_l3781_378192

theorem sum_divisible_by_ten (n : ℕ) : 
  10 ∣ (n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2) ↔ n % 5 = 1 := by
  sorry

end sum_divisible_by_ten_l3781_378192


namespace cone_lateral_surface_area_l3781_378170

/-- Given a cone with slant height 13 cm and height 12 cm, its lateral surface area is 65π cm². -/
theorem cone_lateral_surface_area (s h r : ℝ) : 
  s = 13 → h = 12 → s^2 = h^2 + r^2 → (π * r * s : ℝ) = 65 * π := by sorry

end cone_lateral_surface_area_l3781_378170


namespace perpendicular_bisector_b_value_l3781_378113

/-- Given that the line x + y = b is the perpendicular bisector of the line segment 
    from (2,4) to (6,10), prove that b = 11. -/
theorem perpendicular_bisector_b_value : 
  let point1 : ℝ × ℝ := (2, 4)
  let point2 : ℝ × ℝ := (6, 10)
  let midpoint : ℝ × ℝ := ((point1.1 + point2.1) / 2, (point1.2 + point2.2) / 2)
  ∃ b : ℝ, (∀ (x y : ℝ), x + y = b ↔ ((x - midpoint.1) ^ 2 + (y - midpoint.2) ^ 2 = 
    (point1.1 - midpoint.1) ^ 2 + (point1.2 - midpoint.2) ^ 2)) → b = 11 :=
by
  sorry


end perpendicular_bisector_b_value_l3781_378113


namespace three_digit_numbers_with_properties_l3781_378187

def satisfiesConditions (N : ℕ) : Prop :=
  N % 2 = 1 ∧ N % 3 = 2 ∧ N % 4 = 3 ∧ N % 5 = 4 ∧ N % 6 = 5

def isThreeDigit (N : ℕ) : Prop :=
  100 ≤ N ∧ N ≤ 999

def solutionSet : Set ℕ :=
  {119, 179, 239, 299, 359, 419, 479, 539, 599, 659, 719, 779, 839, 899, 959}

theorem three_digit_numbers_with_properties :
  {N : ℕ | isThreeDigit N ∧ satisfiesConditions N} = solutionSet := by sorry

end three_digit_numbers_with_properties_l3781_378187


namespace tiles_for_monica_room_l3781_378163

/-- Calculates the total number of tiles needed to cover a rectangular room
    with a border of larger tiles and inner area of smaller tiles. -/
def total_tiles (room_length room_width border_tile_size inner_tile_size : ℕ) : ℕ :=
  let border_area := room_length * room_width - (room_length - 2 * border_tile_size) * (room_width - 2 * border_tile_size)
  let border_tiles := border_area / (border_tile_size * border_tile_size)
  let inner_area := (room_length - 2 * border_tile_size) * (room_width - 2 * border_tile_size)
  let inner_tiles := inner_area / (inner_tile_size * inner_tile_size)
  border_tiles + inner_tiles

/-- Theorem stating that the total number of tiles for the given room dimensions and tile sizes is 318. -/
theorem tiles_for_monica_room : total_tiles 24 18 2 1 = 318 := by
  sorry

end tiles_for_monica_room_l3781_378163


namespace simplify_expression_1_simplify_expression_2_l3781_378137

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) (ha : a ≠ 0) :
  (2 * a^(2/3) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3)) / (-3 * a^(1/6) * b^(5/6)) = 4 * a := by
  sorry

-- Problem 2
theorem simplify_expression_2 :
  (25^(1/3) - 125^(1/2)) / 25^(1/4) = 5^(1/6) - 5 := by
  sorry

end simplify_expression_1_simplify_expression_2_l3781_378137


namespace winnie_balloons_l3781_378158

/-- The number of balloons Winnie keeps for herself -/
def balloons_kept (total_balloons : ℕ) (num_friends : ℕ) : ℕ :=
  total_balloons % num_friends

theorem winnie_balloons :
  let red_balloons : ℕ := 15
  let blue_balloons : ℕ := 42
  let yellow_balloons : ℕ := 54
  let purple_balloons : ℕ := 92
  let total_balloons : ℕ := red_balloons + blue_balloons + yellow_balloons + purple_balloons
  let num_friends : ℕ := 11
  balloons_kept total_balloons num_friends = 5 :=
by sorry

end winnie_balloons_l3781_378158


namespace subtract_inequality_negative_l3781_378144

theorem subtract_inequality_negative (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) : a - b < 0 := by
  sorry

end subtract_inequality_negative_l3781_378144


namespace no_real_solutions_for_equation_l3781_378104

theorem no_real_solutions_for_equation :
  ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 1/a + 1/b = 1/(a+b) :=
by sorry

end no_real_solutions_for_equation_l3781_378104


namespace greatest_abcba_div_11_is_greatest_greatest_abcba_div_11_is_divisible_by_11_l3781_378167

/-- Represents a five-digit number in the form AB,CBA -/
structure ABCBA where
  a : Nat
  b : Nat
  c : Nat
  value : Nat
  h1 : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9
  h2 : a ≠ b ∧ a ≠ c ∧ b ≠ c
  h3 : value = a * 10000 + b * 1000 + c * 100 + b * 10 + a

/-- The greatest ABCBA number divisible by 11 -/
def greatest_abcba_div_11 : ABCBA :=
  { a := 9
  , b := 6
  , c := 5
  , value := 96569
  , h1 := by simp
  , h2 := by simp
  , h3 := by simp
  }

theorem greatest_abcba_div_11_is_greatest :
  ∀ n : ABCBA, n.value % 11 = 0 → n.value ≤ greatest_abcba_div_11.value :=
sorry

theorem greatest_abcba_div_11_is_divisible_by_11 :
  greatest_abcba_div_11.value % 11 = 0 :=
sorry

end greatest_abcba_div_11_is_greatest_greatest_abcba_div_11_is_divisible_by_11_l3781_378167


namespace figure_to_square_possible_l3781_378172

/-- A figure on a grid --/
structure GridFigure where
  -- Add necessary properties of the figure
  area : ℕ

/-- A triangle on a grid --/
structure GridTriangle where
  -- Add necessary properties of a triangle
  area : ℕ

/-- Represents a square --/
structure Square where
  side_length : ℕ

/-- Function to cut a figure into triangles --/
def cut_into_triangles (figure : GridFigure) : List GridTriangle :=
  sorry

/-- Function to check if triangles can form a square --/
def can_form_square (triangles : List GridTriangle) : Bool :=
  sorry

/-- The main theorem --/
theorem figure_to_square_possible (figure : GridFigure) :
  ∃ (triangles : List GridTriangle),
    (triangles.length = 5) ∧
    (cut_into_triangles figure = triangles) ∧
    (can_form_square triangles = true) :=
  sorry

end figure_to_square_possible_l3781_378172


namespace polynomial_equality_l3781_378145

theorem polynomial_equality (a b c : ℝ) :
  (∀ x : ℝ, 4 * x^2 - 3 * x + 1 = a * (x - 1)^2 + b * (x - 1) + c) →
  4 * a + 2 * b + c = 28 := by
  sorry

end polynomial_equality_l3781_378145


namespace savings_multiple_l3781_378174

/-- Given two people's savings A and K satisfying certain conditions,
    prove that doubling K results in 3 times A -/
theorem savings_multiple (A K : ℚ) 
  (h1 : A + K = 750)
  (h2 : A - 150 = (1/3) * K) :
  2 * K = 3 * A := by
  sorry

end savings_multiple_l3781_378174


namespace remainder_sum_l3781_378175

theorem remainder_sum (n : ℤ) : n % 12 = 7 → (n % 3 + n % 4 = 4) := by
  sorry

end remainder_sum_l3781_378175


namespace cookie_boxes_l3781_378143

theorem cookie_boxes (n : Nat) (h : n = 392) : 
  (Finset.filter (fun p => 1 < p ∧ p < n ∧ n / p > 3) (Finset.range (n + 1))).card = 11 := by
  sorry

end cookie_boxes_l3781_378143


namespace circle_tangency_problem_l3781_378134

theorem circle_tangency_problem :
  let max_radius : ℕ := 36
  let valid_radius (s : ℕ) : Prop := 1 ≤ s ∧ s < max_radius ∧ max_radius % s = 0
  (Finset.filter valid_radius (Finset.range max_radius)).card = 8 := by
  sorry

end circle_tangency_problem_l3781_378134


namespace rectangle_point_distances_l3781_378189

-- Define the rectangle and point P
def Rectangle (A B C D : ℝ × ℝ) : Prop :=
  -- Add conditions for a rectangle here
  True

def InsideRectangle (P : ℝ × ℝ) (A B C D : ℝ × ℝ) : Prop :=
  -- Add conditions for P being inside the rectangle here
  True

-- Define the distance function
def distance (P Q : ℝ × ℝ) : ℝ :=
  -- Add definition for Euclidean distance here
  0

-- Theorem statement
theorem rectangle_point_distances 
  (A B C D P : ℝ × ℝ) 
  (h_rect : Rectangle A B C D) 
  (h_inside : InsideRectangle P A B C D) 
  (h_PA : distance P A = 5)
  (h_PD : distance P D = 12)
  (h_PC : distance P C = 13) :
  distance P B = 5 * Real.sqrt 2 :=
sorry

end rectangle_point_distances_l3781_378189


namespace owen_final_turtles_l3781_378140

/-- Represents the number of turtles each person has at different times --/
structure TurtleCount where
  owen_initial : ℕ
  johanna_initial : ℕ
  owen_after_month : ℕ
  johanna_after_month : ℕ
  owen_final : ℕ

/-- Calculates the final number of turtles Owen has --/
def calculate_final_turtles (t : TurtleCount) : Prop :=
  t.owen_initial = 21 ∧
  t.johanna_initial = t.owen_initial - 5 ∧
  t.owen_after_month = 2 * t.owen_initial ∧
  t.johanna_after_month = t.johanna_initial / 2 ∧
  t.owen_final = t.owen_after_month + t.johanna_after_month ∧
  t.owen_final = 50

theorem owen_final_turtles :
  ∃ t : TurtleCount, calculate_final_turtles t :=
sorry

end owen_final_turtles_l3781_378140


namespace intersection_plane_sphere_sum_l3781_378116

/-- Given a plane x + 2y + 3z = 6 that passes through a point (a, b, c) and intersects
    the coordinate axes at points A, B, C distinct from the origin O,
    prove that a/p + b/q + c/r = 2, where (p, q, r) is the center of the sphere
    passing through A, B, C, and O. -/
theorem intersection_plane_sphere_sum (a b c p q r : ℝ) : 
  (∃ (x y z : ℝ), x + 2*y + 3*z = 6 ∧ 
                   a + 2*b + 3*c = 6 ∧
                   (x = 0 ∨ y = 0 ∨ z = 0) ∧
                   (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) →
  (p^2 + q^2 + r^2 = (p - 6)^2 + q^2 + r^2 ∧
   p^2 + q^2 + r^2 = p^2 + (q - 3)^2 + r^2 ∧
   p^2 + q^2 + r^2 = p^2 + q^2 + (r - 2)^2) →
  a/p + b/q + c/r = 2 := by
  sorry

end intersection_plane_sphere_sum_l3781_378116


namespace reciprocal_of_negative_2022_l3781_378152

theorem reciprocal_of_negative_2022 : (1 : ℚ) / (-2022 : ℚ) = -1 / 2022 := by sorry

end reciprocal_of_negative_2022_l3781_378152


namespace jason_has_21_toys_l3781_378126

-- Define the number of toys for each person
def rachel_toys : ℕ := 1
def john_toys : ℕ := rachel_toys + 6
def jason_toys : ℕ := 3 * john_toys

-- Theorem statement
theorem jason_has_21_toys : jason_toys = 21 := by
  sorry

end jason_has_21_toys_l3781_378126


namespace class_average_weight_l3781_378191

/-- Given two sections A and B in a class, calculate the average weight of the whole class. -/
theorem class_average_weight 
  (students_A : ℕ) 
  (students_B : ℕ) 
  (avg_weight_A : ℝ) 
  (avg_weight_B : ℝ) : 
  students_A = 50 → 
  students_B = 40 → 
  avg_weight_A = 50 → 
  avg_weight_B = 70 → 
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B) = 
    (50 * 50 + 70 * 40) / (50 + 40) := by
  sorry

#eval (50 * 50 + 70 * 40) / (50 + 40)  -- This will evaluate to approximately 58.89

end class_average_weight_l3781_378191


namespace odometer_problem_l3781_378106

theorem odometer_problem (a b c : ℕ) (n : ℕ+) :
  100 ≤ 100 * a + 10 * b + c →
  100 * a + 10 * b + c ≤ 999 →
  a ≥ 1 →
  a + b + c ≤ 7 →
  100 * c + 10 * b + a - (100 * a + 10 * b + c) = 55 * n →
  a^2 + b^2 + c^2 = 37 := by
sorry

end odometer_problem_l3781_378106


namespace pascal_triangle_48th_number_l3781_378127

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of elements in the row of Pascal's triangle -/
def row_size : ℕ := 51

/-- The index of the number we're looking for in the row -/
def target_index : ℕ := 48

/-- The theorem stating that the 48th number in the row with 51 numbers 
    of Pascal's triangle is 19600 -/
theorem pascal_triangle_48th_number : 
  binomial (row_size - 1) (target_index - 1) = 19600 := by sorry

end pascal_triangle_48th_number_l3781_378127


namespace partnership_investment_timing_l3781_378147

theorem partnership_investment_timing 
  (x : ℝ) 
  (m : ℝ) 
  (total_gain : ℝ) 
  (a_share : ℝ) 
  (h1 : total_gain = 18600) 
  (h2 : a_share = 6200) 
  (h3 : a_share / total_gain = 1 / 3) 
  (h4 : x * 12 = (1 / 3) * (x * 12 + 2 * x * (12 - m) + 3 * x * 4)) : 
  m = 6 := by
  sorry

end partnership_investment_timing_l3781_378147


namespace correlation_coefficient_comparison_l3781_378168

def x : List ℝ := [1, 2, 3, 4, 5]
def y : List ℝ := [3, 5.3, 6.9, 9.1, 10.8]
def U : List ℝ := [1, 2, 3, 4, 5]
def V : List ℝ := [12.7, 10.2, 7, 3.6, 1]

def r1 : ℝ := sorry
def r2 : ℝ := sorry

theorem correlation_coefficient_comparison : r2 < 0 ∧ 0 < r1 := by sorry

end correlation_coefficient_comparison_l3781_378168


namespace lives_lost_l3781_378154

theorem lives_lost (initial_lives : ℕ) (lives_gained : ℕ) (final_lives : ℕ) 
  (h1 : initial_lives = 14)
  (h2 : lives_gained = 36)
  (h3 : final_lives = 46) :
  ∃ (lives_lost : ℕ), initial_lives - lives_lost + lives_gained = final_lives ∧ lives_lost = 4 := by
  sorry

end lives_lost_l3781_378154


namespace fraction_denominator_l3781_378131

theorem fraction_denominator (x y : ℝ) (h : x / y = 7 / 3) :
  ∃ z : ℝ, (x + y) / z = 2.5 ∧ z = 4 * y / 3 :=
by sorry

end fraction_denominator_l3781_378131


namespace f_properties_l3781_378118

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - |x - a| + 1

theorem f_properties :
  (∀ x ∈ Set.Icc 0 2, f 0 x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 2, f 0 x = 3) ∧
  (∀ x ∈ Set.Icc 0 2, f 0 x ≥ 3/4) ∧
  (∃ x ∈ Set.Icc 0 2, f 0 x = 3/4) ∧
  (∀ a < 0, ∀ x, f a x ≥ 3/4 + a) ∧
  (∀ a < 0, ∃ x, f a x = 3/4 + a) ∧
  (∀ a ≥ 0, ∀ x, f a x ≥ 3/4 - a) ∧
  (∀ a ≥ 0, ∃ x, f a x = 3/4 - a) :=
by sorry

end f_properties_l3781_378118


namespace min_value_and_inequality_l3781_378155

def f (x : ℝ) := 2 * abs (x + 1) + abs (x + 2)

theorem min_value_and_inequality :
  (∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x₀ : ℝ, f x₀ = m) ∧ m = 1) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    (a^2 + b^2) / c + (c^2 + a^2) / b + (b^2 + c^2) / a ≥ 2) :=
by sorry

end min_value_and_inequality_l3781_378155


namespace calculate_expression_l3781_378103

theorem calculate_expression : 5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 := by
  sorry

end calculate_expression_l3781_378103


namespace largest_integer_divisible_by_18_with_sqrt_between_24_and_24_5_l3781_378135

theorem largest_integer_divisible_by_18_with_sqrt_between_24_and_24_5 :
  ∃ n : ℕ, n > 0 ∧ 18 ∣ n ∧ 24 < Real.sqrt n ∧ Real.sqrt n < 24.5 ∧
  ∀ m : ℕ, m > 0 → 18 ∣ m → 24 < Real.sqrt m → Real.sqrt m < 24.5 → m ≤ n :=
by
  sorry

end largest_integer_divisible_by_18_with_sqrt_between_24_and_24_5_l3781_378135


namespace books_sold_in_garage_sale_l3781_378181

theorem books_sold_in_garage_sale :
  ∀ (initial_books given_to_friend remaining_books : ℕ),
    initial_books = 108 →
    given_to_friend = 35 →
    remaining_books = 62 →
    initial_books - given_to_friend - remaining_books = 11 :=
by
  sorry

end books_sold_in_garage_sale_l3781_378181


namespace simplify_and_sum_l3781_378142

theorem simplify_and_sum : ∃ (a b : ℕ), 
  (a > 0) ∧ (b > 0) ∧ 
  ((2^10 * 5^2)^(1/4) : ℝ) = a * (b^(1/4) : ℝ) ∧ 
  a + b = 104 := by sorry

end simplify_and_sum_l3781_378142


namespace sequence_double_plus_one_greater_l3781_378190

/-- Definition of the property $\{a_n\} > M$ -/
def sequence_greater_than (a : ℕ → ℝ) (M : ℝ) : Prop :=
  ∀ n : ℕ, a n ≥ M ∨ a (n + 1) ≥ M

/-- Main theorem -/
theorem sequence_double_plus_one_greater (a : ℕ → ℝ) (M : ℝ) :
  sequence_greater_than a M → sequence_greater_than (fun n ↦ 2 * a n + 1) (2 * M + 1) := by
  sorry

end sequence_double_plus_one_greater_l3781_378190


namespace segment_length_product_l3781_378179

theorem segment_length_product (a : ℝ) :
  (∃ a₁ a₂ : ℝ, 
    (∀ a : ℝ, (3*a - 5)^2 + (a - 3)^2 = 117 ↔ (a = a₁ ∨ a = a₂)) ∧
    a₁ * a₂ = -8.32) := by
  sorry

end segment_length_product_l3781_378179


namespace circle_radius_theorem_l3781_378166

theorem circle_radius_theorem (r : ℝ) (h : r > 0) :
  3 * (2 * Real.pi * r) = Real.pi * r^2 → r = 6 := by
  sorry

end circle_radius_theorem_l3781_378166


namespace bread_for_double_meat_sandwiches_l3781_378117

/-- Given the following conditions:
  - Two pieces of bread are needed for one regular sandwich.
  - Three pieces of bread are needed for a double meat sandwich.
  - There are 14 regular sandwiches.
  - A total of 64 pieces of bread are used.
Prove that the number of bread pieces used for double meat sandwiches is 36. -/
theorem bread_for_double_meat_sandwiches :
  let regular_sandwich_bread : ℕ := 2
  let double_meat_sandwich_bread : ℕ := 3
  let regular_sandwiches : ℕ := 14
  let total_bread : ℕ := 64
  let double_meat_bread := total_bread - regular_sandwich_bread * regular_sandwiches
  double_meat_bread = 36 := by
  sorry

end bread_for_double_meat_sandwiches_l3781_378117


namespace complement_union_A_B_l3781_378186

-- Define the sets A and B
def A : Set ℝ := {x | x < 0}
def B : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem complement_union_A_B :
  (A ∪ B)ᶜ = {x : ℝ | 0 ≤ x ∧ x < 2} := by sorry

end complement_union_A_B_l3781_378186


namespace parallelogram_area_l3781_378171

/-- The area of a parallelogram with base 22 cm and height 21 cm is 462 square centimeters. -/
theorem parallelogram_area : 
  let base : ℝ := 22
  let height : ℝ := 21
  let area := base * height
  area = 462 :=
by sorry

end parallelogram_area_l3781_378171


namespace largest_geometric_digit_sequence_l3781_378114

/-- Checks if the given three digits form a geometric sequence -/
def is_geometric_sequence (a b c : ℕ) : Prop :=
  ∃ r : ℚ, b = r * a ∧ c = r * b

/-- Checks if the given number is a valid solution -/
def is_valid_solution (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  100 ≤ n ∧ n < 1000 ∧  -- Three-digit integer
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧  -- Distinct digits
  is_geometric_sequence a b c ∧  -- Geometric sequence
  b % 2 = 0  -- Tens digit is even

theorem largest_geometric_digit_sequence :
  ∀ n : ℕ, is_valid_solution n → n ≤ 964 :=
sorry

end largest_geometric_digit_sequence_l3781_378114


namespace sufficient_not_necessary_condition_l3781_378162

open Real

-- Define the logarithm base 10
noncomputable def log10 (x : ℝ) : ℝ := log x / log 10

-- Define the set {1, 2}
def set_1_2 : Set ℝ := {1, 2}

-- Theorem statement
theorem sufficient_not_necessary_condition :
  (∀ m ∈ set_1_2, log10 m < 1) ∧
  (∃ m : ℝ, log10 m < 1 ∧ m ∉ set_1_2) :=
by sorry

end sufficient_not_necessary_condition_l3781_378162


namespace jakes_weight_l3781_378161

/-- Proves Jake's current weight given the conditions of the problem -/
theorem jakes_weight (jake sister brother : ℝ) 
  (h1 : jake - 20 = 2 * sister)
  (h2 : brother = 0.5 * jake)
  (h3 : jake + sister + brother = 330) :
  jake = 170 := by
sorry

end jakes_weight_l3781_378161


namespace reciprocal_power_l3781_378198

theorem reciprocal_power (a : ℝ) (h : a⁻¹ = -1) : a^2023 = -1 := by
  sorry

end reciprocal_power_l3781_378198


namespace inequality_proof_l3781_378124

def f (x a : ℝ) : ℝ := |x + a| + 2 * |x - 1|

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : ∀ x ∈ Set.Icc 1 2, f x a > x^2 - b + 1) :
  (a + 1/2)^2 + (b + 1/2)^2 > 2 := by
sorry

end inequality_proof_l3781_378124


namespace min_vertical_distance_l3781_378159

noncomputable def f (x : ℝ) : ℝ := |x - 1|
noncomputable def g (x : ℝ) : ℝ := -x^2 - 4*x - 3

theorem min_vertical_distance : 
  ∃ (x₀ : ℝ), ∀ (x : ℝ), |f x - g x| ≥ |f x₀ - g x₀| ∧ |f x₀ - g x₀| = 10 :=
sorry

end min_vertical_distance_l3781_378159


namespace bottle_caps_wrappers_difference_l3781_378195

/-- The number of wrappers Danny found at the park -/
def wrappers_found : ℕ := 46

/-- The number of bottle caps Danny found at the park -/
def bottle_caps_found : ℕ := 50

/-- The number of bottle caps Danny now has in his collection -/
def bottle_caps_in_collection : ℕ := 21

/-- The number of wrappers Danny now has in his collection -/
def wrappers_in_collection : ℕ := 52

/-- Theorem stating the difference between bottle caps and wrappers found at the park -/
theorem bottle_caps_wrappers_difference : 
  bottle_caps_found - wrappers_found = 4 := by
  sorry

end bottle_caps_wrappers_difference_l3781_378195


namespace range_of_a_given_three_integer_solutions_l3781_378119

/-- The inequality (2x-1)^2 < ax^2 has exactly three integer solutions -/
def has_three_integer_solutions (a : ℝ) : Prop :=
  ∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (∀ (w : ℤ), (2*w - 1)^2 < a*w^2 ↔ w = x ∨ w = y ∨ w = z)

/-- The theorem stating the range of a given the condition -/
theorem range_of_a_given_three_integer_solutions :
  ∀ a : ℝ, has_three_integer_solutions a ↔ 25/9 < a ∧ a ≤ 49/16 := by sorry

end range_of_a_given_three_integer_solutions_l3781_378119


namespace binomial_20_10_l3781_378121

theorem binomial_20_10 (h1 : Nat.choose 18 8 = 43758) 
                       (h2 : Nat.choose 18 9 = 48620) 
                       (h3 : Nat.choose 18 10 = 43758) : 
  Nat.choose 20 10 = 184756 := by
  sorry

end binomial_20_10_l3781_378121


namespace school2_selection_l3781_378128

/-- Represents the number of students selected from a school in a system sampling. -/
def studentsSelected (schoolSize totalStudents selectedStudents : ℕ) : ℚ :=
  (schoolSize : ℚ) * (selectedStudents : ℚ) / (totalStudents : ℚ)

/-- The main theorem about the number of students selected from School 2. -/
theorem school2_selection :
  let totalStudents : ℕ := 360
  let school1Size : ℕ := 123
  let school2Size : ℕ := 123
  let school3Size : ℕ := 114
  let totalSelected : ℕ := 60
  let remainingSelected : ℕ := totalSelected - 1
  let remainingStudents : ℕ := totalStudents - 1
  Int.ceil (studentsSelected school2Size remainingStudents remainingSelected) = 20 := by
  sorry

#check school2_selection

end school2_selection_l3781_378128


namespace square_root_of_sixteen_l3781_378173

theorem square_root_of_sixteen : 
  {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end square_root_of_sixteen_l3781_378173


namespace inequality_solution_implies_k_range_l3781_378136

theorem inequality_solution_implies_k_range :
  ∀ k : ℝ,
  (∀ x : ℝ, x > 1/2 ↔ (k^2 - 2*k + 3/2)^x < (k^2 - 2*k + 3/2)^(1-x)) →
  (1 - Real.sqrt 2 / 2 < k ∧ k < 1 + Real.sqrt 2 / 2) :=
by sorry

end inequality_solution_implies_k_range_l3781_378136


namespace no_integer_solution_l3781_378112

theorem no_integer_solution : ∀ x y : ℤ, x^5 + y^5 + 1 ≠ (x+2)^5 + (y-3)^5 := by
  sorry

end no_integer_solution_l3781_378112


namespace right_triangle_existence_l3781_378188

theorem right_triangle_existence (a b c d : ℕ+) 
  (h1 : a * b = c * d) 
  (h2 : a + b = c - d) : 
  ∃ x y z : ℕ+, x^2 + y^2 = z^2 ∧ (1/2 : ℚ) * x * y = a * b := by
sorry

end right_triangle_existence_l3781_378188


namespace last_digit_is_11_l3781_378115

def fibonacci_mod_12 : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => (fibonacci_mod_12 (n + 1) + fibonacci_mod_12 n) % 12

def digit_appears (d : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ n ∧ fibonacci_mod_12 k = d

theorem last_digit_is_11 :
  ∀ d : ℕ, d < 12 →
    ∃ n : ℕ, digit_appears d n ∧
      ¬∃ m : ℕ, m > n ∧ digit_appears 11 m ∧ ¬digit_appears 11 n :=
by sorry

end last_digit_is_11_l3781_378115


namespace min_value_theorem_l3781_378160

/-- A monotonically increasing function on ℝ of the form f(x) = a^x + b -/
noncomputable def f (a b : ℝ) : ℝ → ℝ := λ x ↦ a^x + b

/-- The theorem stating the minimum value of the expression -/
theorem min_value_theorem (a b : ℝ) (h1 : a > 1) (h2 : b > 0) (h3 : f a b 1 = 3) :
  (4 / (a - 1) + 1 / b) ≥ 9/2 :=
sorry

end min_value_theorem_l3781_378160


namespace abs_equation_roots_l3781_378182

def abs_equation (x : ℝ) : Prop :=
  |x|^2 + |x| - 6 = 0

theorem abs_equation_roots :
  ∃ (r₁ r₂ : ℝ),
    (abs_equation r₁ ∧ abs_equation r₂) ∧
    (∀ x, abs_equation x → (x = r₁ ∨ x = r₂)) ∧
    (r₁ + r₂ = 0) ∧
    (r₁ * r₂ = -4) :=
by sorry

end abs_equation_roots_l3781_378182
