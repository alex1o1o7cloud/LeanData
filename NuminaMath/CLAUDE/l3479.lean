import Mathlib

namespace obtain_100_with_fewer_sevens_l3479_347905

theorem obtain_100_with_fewer_sevens : ∃ (expr : ℕ), 
  (expr = 100) ∧ 
  (∃ (a b c d e f g h i : ℕ), 
    (a + b + c + d + e + f + g + h + i < 10) ∧
    (expr = (777 / 7 - 77 / 7) ∨ 
     expr = (7 * 7 + 7 * 7 + 7 / 7 + 7 / 7))) :=
by sorry

end obtain_100_with_fewer_sevens_l3479_347905


namespace smallest_determinant_and_minimal_pair_l3479_347947

def determinant (a b : ℤ) : ℤ := 36 * b - 81 * a

theorem smallest_determinant_and_minimal_pair :
  (∃ c : ℕ+, ∀ a b : ℤ, determinant a b ≠ 0 → c ≤ |determinant a b|) ∧
  (∃ a b : ℕ, determinant a b = 9 ∧
    ∀ a' b' : ℕ, determinant a' b' = 9 → a + b ≤ a' + b') :=
by sorry

end smallest_determinant_and_minimal_pair_l3479_347947


namespace meat_remaining_l3479_347955

theorem meat_remaining (initial_meat : ℝ) (meatball_fraction : ℝ) (spring_roll_meat : ℝ) :
  initial_meat = 20 →
  meatball_fraction = 1/4 →
  spring_roll_meat = 3 →
  initial_meat - (meatball_fraction * initial_meat + spring_roll_meat) = 12 := by
  sorry

end meat_remaining_l3479_347955


namespace pink_crayons_count_l3479_347954

/-- Represents the number of crayons of each color in a crayon box. -/
structure CrayonBox where
  total : ℕ
  red : ℕ
  blue : ℕ
  green : ℕ
  pink : ℕ

/-- Theorem stating the number of pink crayons in the given crayon box. -/
theorem pink_crayons_count (box : CrayonBox) : box.pink = 6 :=
  by
  have h1 : box.total = 24 := by sorry
  have h2 : box.red = 8 := by sorry
  have h3 : box.blue = 6 := by sorry
  have h4 : box.green = 4 := by sorry
  have h5 : box.green = (2 * box.blue) / 3 := by sorry
  have h6 : box.total = box.red + box.blue + box.green + box.pink := by sorry
  sorry


end pink_crayons_count_l3479_347954


namespace min_fixed_amount_l3479_347901

def fixed_amount (F : ℝ) : Prop :=
  ∀ (S : ℝ), S ≥ 7750 → F + 0.04 * S ≥ 500

theorem min_fixed_amount :
  ∃ (F : ℝ), F ≥ 190 ∧ fixed_amount F :=
sorry

end min_fixed_amount_l3479_347901


namespace intersection_of_A_and_B_l3479_347933

-- Define set A
def A : Set ℝ := {y | ∃ x, y = Real.cos x}

-- Define set B
def B : Set ℝ := {x | x^2 + x ≥ 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc 0 1 ∪ {-1} := by
  sorry

end intersection_of_A_and_B_l3479_347933


namespace complement_A_intersect_B_l3479_347938

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x > 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.sin x}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = Set.Icc 0 1 := by sorry

end complement_A_intersect_B_l3479_347938


namespace pure_imaginary_complex_number_l3479_347991

theorem pure_imaginary_complex_number (x : ℝ) :
  let z : ℂ := Complex.mk (x^2 - 3*x + 2) (x - 1)
  (z.re = 0 ∧ z.im ≠ 0) → x = 2 := by
  sorry

end pure_imaginary_complex_number_l3479_347991


namespace garden_area_is_2400_l3479_347912

/-- Represents a rectangular garden with given properties -/
structure Garden where
  length : ℝ
  width : ℝ
  length_walk : ℕ
  perimeter_walk : ℕ
  total_distance : ℝ
  len_condition : length * length_walk = total_distance
  peri_condition : (2 * length + 2 * width) * perimeter_walk = total_distance

/-- Calculates the area of a rectangular garden -/
def garden_area (g : Garden) : ℝ :=
  g.length * g.width

/-- Theorem stating that a garden with the given properties has an area of 2400 square meters -/
theorem garden_area_is_2400 (g : Garden) 
  (h1 : g.length_walk = 50)
  (h2 : g.perimeter_walk = 15)
  (h3 : g.total_distance = 3000) : 
  garden_area g = 2400 := by
  sorry

end garden_area_is_2400_l3479_347912


namespace function_property_l3479_347909

-- Define the functions
def f1 (x : ℝ) := |x|
def f2 (x : ℝ) := x - |x|
def f3 (x : ℝ) := x + 1
def f4 (x : ℝ) := -x

-- Define the property we're checking
def satisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 * x) = 2 * f x

-- Theorem statement
theorem function_property :
  satisfiesProperty f1 ∧
  satisfiesProperty f2 ∧
  ¬satisfiesProperty f3 ∧
  satisfiesProperty f4 :=
sorry

end function_property_l3479_347909


namespace square_area_from_diagonal_l3479_347972

/-- The area of a square with diagonal length 8√2 is 64 -/
theorem square_area_from_diagonal : 
  ∀ (s : ℝ), s > 0 → s * s * 2 = (8 * Real.sqrt 2) ^ 2 → s * s = 64 := by
sorry

end square_area_from_diagonal_l3479_347972


namespace hyperbola_triangle_perimeter_l3479_347982

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

/-- The distance between foci -/
def focal_distance : ℝ := 10

/-- Point P is on the hyperbola -/
def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola_equation P.1 P.2

/-- The distance from P to the right focus F₂ -/
def distance_PF₂ : ℝ := 7

/-- The perimeter of triangle F₁PF₂ -/
def triangle_perimeter (d_PF₁ : ℝ) : ℝ :=
  d_PF₁ + distance_PF₂ + focal_distance

theorem hyperbola_triangle_perimeter :
  ∀ P : ℝ × ℝ, point_on_hyperbola P →
  ∃ d_PF₁ : ℝ, triangle_perimeter d_PF₁ = 30 :=
sorry

end hyperbola_triangle_perimeter_l3479_347982


namespace system_solution_sum_l3479_347961

theorem system_solution_sum (a b : ℝ) : 
  (1 : ℝ) * a + 2 = -1 ∧ 2 * (1 : ℝ) - b * 2 = 0 → a + b = -2 := by
  sorry

end system_solution_sum_l3479_347961


namespace smallest_four_digit_negative_congruent_to_one_mod_37_l3479_347917

theorem smallest_four_digit_negative_congruent_to_one_mod_37 :
  ∀ x : ℤ, x < 0 ∧ x ≥ -9999 ∧ x ≡ 1 [ZMOD 37] → x ≥ -1034 :=
by sorry

end smallest_four_digit_negative_congruent_to_one_mod_37_l3479_347917


namespace area_at_stage_8_l3479_347996

/-- The side length of each square -/
def squareSide : ℕ := 4

/-- The area of each square -/
def squareArea : ℕ := squareSide * squareSide

/-- The number of squares at a given stage -/
def numSquaresAtStage (stage : ℕ) : ℕ := stage

/-- The total area of the rectangle at a given stage -/
def totalAreaAtStage (stage : ℕ) : ℕ := numSquaresAtStage stage * squareArea

/-- The theorem stating that the area of the rectangle at Stage 8 is 128 square inches -/
theorem area_at_stage_8 : totalAreaAtStage 8 = 128 := by sorry

end area_at_stage_8_l3479_347996


namespace train_crossing_time_l3479_347913

/-- Proves that a train with given length and speed takes the calculated time to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) (crossing_time : Real) :
  train_length = 2500 ∧ 
  train_speed_kmh = 90 →
  crossing_time = 100 := by
  sorry

#check train_crossing_time

end train_crossing_time_l3479_347913


namespace triangle_abc_properties_l3479_347993

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given condition
  2 * a * Real.sin B - Real.sqrt 5 * b * Real.cos A = 0 →
  -- Theorem 1: cos A = 2/3
  Real.cos A = 2/3 ∧
  -- Theorem 2: If a = √5 and b = 2, area = √5
  (a = Real.sqrt 5 ∧ b = 2 → 
    (1/2) * a * b * Real.sin C = Real.sqrt 5) :=
by sorry

end triangle_abc_properties_l3479_347993


namespace ferry_route_ratio_l3479_347988

-- Define the parameters
def ferry_p_speed : ℝ := 6
def ferry_p_time : ℝ := 3
def ferry_q_speed_difference : ℝ := 3
def ferry_q_time_difference : ℝ := 3

-- Define the theorem
theorem ferry_route_ratio :
  let ferry_p_distance := ferry_p_speed * ferry_p_time
  let ferry_q_speed := ferry_p_speed + ferry_q_speed_difference
  let ferry_q_time := ferry_p_time + ferry_q_time_difference
  let ferry_q_distance := ferry_q_speed * ferry_q_time
  ferry_q_distance / ferry_p_distance = 3 := by
  sorry


end ferry_route_ratio_l3479_347988


namespace field_length_width_ratio_l3479_347987

/-- Proves that the ratio of a rectangular field's length to its width is 2:1,
    given specific conditions about the field and a pond within it. -/
theorem field_length_width_ratio :
  ∀ (field_length field_width pond_side : ℝ),
  field_length = 48 →
  pond_side = 8 →
  pond_side * pond_side = (field_length * field_width) / 18 →
  field_length / field_width = 2 := by
  sorry

end field_length_width_ratio_l3479_347987


namespace min_value_sum_and_reciprocals_l3479_347934

theorem min_value_sum_and_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b + 1/a + 1/b ≥ 4 ∧ (a + b + 1/a + 1/b = 4 ↔ a = 1 ∧ b = 1) := by
  sorry

end min_value_sum_and_reciprocals_l3479_347934


namespace triangle_properties_l3479_347974

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Two vectors are parallel -/
def parallel (v w : Vector2D) : Prop :=
  ∃ (k : ℝ), v.x * w.y = k * v.y * w.x

variable (ABC : Triangle)
variable (m n : Vector2D)

/-- The given conditions -/
axiom cond1 : m = ⟨2 * Real.sin ABC.B, -Real.sqrt 3⟩
axiom cond2 : n = ⟨Real.cos (2 * ABC.B), 2 * (Real.cos ABC.B)^2 - 1⟩
axiom cond3 : parallel m n
axiom cond4 : ABC.b = 2

/-- The theorem to be proved -/
theorem triangle_properties :
  ABC.B = Real.pi / 3 ∧
  (∀ (S : ℝ), S = 1/2 * ABC.a * ABC.c * Real.sin ABC.B → S ≤ Real.sqrt 3) :=
sorry

end triangle_properties_l3479_347974


namespace car_part_payment_l3479_347924

theorem car_part_payment (remaining_payment : ℝ) (part_payment_percentage : ℝ) 
  (h1 : remaining_payment = 5700)
  (h2 : part_payment_percentage = 0.05) : 
  (remaining_payment / (1 - part_payment_percentage)) * part_payment_percentage = 300 := by
  sorry

end car_part_payment_l3479_347924


namespace black_blue_difference_l3479_347923

/-- Represents Sam's pen collection -/
structure PenCollection where
  black : ℕ
  blue : ℕ
  red : ℕ
  pencils : ℕ

/-- Conditions for Sam's pen collection -/
def validCollection (c : PenCollection) : Prop :=
  c.black > c.blue ∧
  c.blue = 2 * c.pencils ∧
  c.pencils = 8 ∧
  c.red = c.pencils - 2 ∧
  c.black + c.blue + c.red = 48

/-- Theorem stating the difference between black and blue pens -/
theorem black_blue_difference (c : PenCollection) 
  (h : validCollection c) : c.black - c.blue = 10 := by
  sorry


end black_blue_difference_l3479_347923


namespace equation_solutions_l3479_347910

theorem equation_solutions (x : ℝ) : 
  (1 / x^2 + 2 / x = 5/4) ↔ (x = 2 ∨ x = -2/5) :=
by sorry

end equation_solutions_l3479_347910


namespace inscribed_sphere_theorem_l3479_347903

/-- A right cone with a sphere inscribed in it -/
structure InscribedSphere where
  cone_base_radius : ℝ
  cone_height : ℝ
  sphere_radius : ℝ

/-- The condition that the sphere is inscribed in the cone -/
def is_inscribed (s : InscribedSphere) : Prop :=
  s.sphere_radius * (s.cone_base_radius^2 + s.cone_height^2).sqrt =
    s.cone_base_radius * (s.cone_height - s.sphere_radius)

/-- The theorem to be proved -/
theorem inscribed_sphere_theorem (b d : ℝ) :
  let s := InscribedSphere.mk 15 20 (b * d.sqrt - b)
  is_inscribed s → b + d = 12 := by
  sorry


end inscribed_sphere_theorem_l3479_347903


namespace d_values_l3479_347929

def a (n : ℕ) : ℕ := 20 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem d_values : {n : ℕ | n > 0} → {d n | n : ℕ} = {1, 3, 9, 27, 81} := by sorry

end d_values_l3479_347929


namespace rectangle_area_and_ratio_l3479_347952

/-- Given a rectangle with original length a and width b -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The new rectangle after increasing dimensions -/
def new_rectangle (r : Rectangle) : Rectangle :=
  { length := 1.12 * r.length,
    width := 1.15 * r.width }

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: The area increase and length-to-width ratio of the rectangle -/
theorem rectangle_area_and_ratio (r : Rectangle) :
  (area (new_rectangle r) = 1.288 * area r) ∧
  (perimeter (new_rectangle r) = 1.13 * perimeter r → r.length = 2 * r.width) := by
  sorry


end rectangle_area_and_ratio_l3479_347952


namespace language_course_enrollment_l3479_347999

theorem language_course_enrollment (total : ℕ) (french : ℕ) (german : ℕ) (spanish : ℕ)
  (french_german : ℕ) (french_spanish : ℕ) (german_spanish : ℕ) (all_three : ℕ) :
  total = 120 →
  french = 52 →
  german = 35 →
  spanish = 48 →
  french_german = 15 →
  french_spanish = 20 →
  german_spanish = 12 →
  all_three = 6 →
  total - (french + german + spanish - french_german - french_spanish - german_spanish + all_three) = 32 := by
sorry

end language_course_enrollment_l3479_347999


namespace train_true_speed_l3479_347970

/-- The true speed of a train given its length, crossing time, and opposing wind speed -/
theorem train_true_speed (train_length : ℝ) (crossing_time : ℝ) (wind_speed : ℝ) :
  train_length = 200 →
  crossing_time = 20 →
  wind_speed = 5 →
  (train_length / crossing_time) + wind_speed = 15 := by
  sorry


end train_true_speed_l3479_347970


namespace initial_number_of_persons_l3479_347904

theorem initial_number_of_persons (n : ℕ) 
  (h1 : (3 : ℝ) * n = 24) : n = 8 := by
  sorry

#check initial_number_of_persons

end initial_number_of_persons_l3479_347904


namespace decimal_to_fraction_l3479_347963

theorem decimal_to_fraction : (2.35 : ℚ) = 47 / 20 := by sorry

end decimal_to_fraction_l3479_347963


namespace fraction_subtraction_equality_l3479_347997

theorem fraction_subtraction_equality : (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end fraction_subtraction_equality_l3479_347997


namespace hydrogen_chloride_production_l3479_347946

/-- Represents the balanced chemical equation for the reaction between methane and chlorine -/
structure BalancedEquation where
  methane : ℕ
  chlorine : ℕ
  tetrachloromethane : ℕ
  hydrogen_chloride : ℕ
  balanced : methane = 1 ∧ chlorine = 4 ∧ tetrachloromethane = 1 ∧ hydrogen_chloride = 4

/-- Represents the given reaction conditions -/
structure ReactionConditions where
  methane : ℕ
  chlorine : ℕ
  tetrachloromethane : ℕ
  methane_eq : methane = 3
  chlorine_eq : chlorine = 12
  tetrachloromethane_eq : tetrachloromethane = 3

/-- Theorem stating that given the reaction conditions, 12 moles of hydrogen chloride are produced -/
theorem hydrogen_chloride_production 
  (balanced : BalancedEquation) 
  (conditions : ReactionConditions) : 
  conditions.methane * balanced.hydrogen_chloride = 12 := by
  sorry

end hydrogen_chloride_production_l3479_347946


namespace tan_pi_minus_alpha_l3479_347915

theorem tan_pi_minus_alpha (α : Real) (h : 3 * Real.sin (α - Real.pi) = Real.cos α) :
  Real.tan (Real.pi - α) = 1/3 := by
  sorry

end tan_pi_minus_alpha_l3479_347915


namespace y_value_when_x_is_one_l3479_347937

-- Define the inverse square relationship between x and y
def inverse_square_relation (k : ℝ) (x y : ℝ) : Prop :=
  x = k / (y ^ 2)

-- Theorem statement
theorem y_value_when_x_is_one 
  (k : ℝ) 
  (h1 : inverse_square_relation k 0.1111111111111111 6) 
  (h2 : k > 0) :
  ∃ y : ℝ, inverse_square_relation k 1 y ∧ y = 2 :=
by
  sorry

end y_value_when_x_is_one_l3479_347937


namespace system_solution_l3479_347931

theorem system_solution (a b c : ℝ) :
  ∃ x y z : ℝ,
  (a * x^3 + b * y = c * z^5 ∧
   a * z^3 + b * x = c * y^5 ∧
   a * y^3 + b * z = c * x^5) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨
   ∃ s t : ℝ, s^2 = (a + t * Real.sqrt (a^2 + 4*b*c)) / (2*c) ∧
             (x = s ∧ y = s ∧ z = s) ∧
             (t = 1 ∨ t = -1)) :=
by sorry

end system_solution_l3479_347931


namespace function_value_at_3000_l3479_347978

/-- Given a function f: ℕ → ℕ satisfying the following properties:
  1) f(0) = 1
  2) For all x, f(x + 3) = f(x) + 2x + 3
  Prove that f(3000) = 3000001 -/
theorem function_value_at_3000 (f : ℕ → ℕ) 
  (h1 : f 0 = 1) 
  (h2 : ∀ x, f (x + 3) = f x + 2 * x + 3) : 
  f 3000 = 3000001 := by
  sorry

end function_value_at_3000_l3479_347978


namespace min_k_value_l3479_347916

theorem min_k_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ k : ℝ, 1 / a + 1 / b + k / (a + b) ≥ 0) → 
  ∀ k : ℝ, k ≥ -4 ∧ ∃ k₀ : ℝ, k₀ = -4 ∧ 1 / a + 1 / b + k₀ / (a + b) ≥ 0 :=
by sorry

end min_k_value_l3479_347916


namespace integral_polynomial_l3479_347940

theorem integral_polynomial (x : ℝ) :
  deriv (fun x => x^3 - x^2 + 5*x) x = 3*x^2 - 2*x + 5 := by
  sorry

end integral_polynomial_l3479_347940


namespace exists_hexagonal_2016_l3479_347956

/-- The n-th hexagonal number -/
def hexagonal (n : ℕ) : ℕ := 2 * n^2 - n

/-- 2016 is a hexagonal number -/
theorem exists_hexagonal_2016 : ∃ n : ℕ, n > 0 ∧ hexagonal n = 2016 := by
  sorry

end exists_hexagonal_2016_l3479_347956


namespace matrices_are_inverses_l3479_347930

theorem matrices_are_inverses : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -7; -5, 9]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![9, 7; 5, 4]
  A * B = 1 ∧ B * A = 1 := by
  sorry

end matrices_are_inverses_l3479_347930


namespace budgets_equal_in_1996_l3479_347994

/-- Represents the year when the budgets of two projects become equal -/
def year_budgets_equal (initial_q initial_v increase_q decrease_v : ℕ) : ℕ :=
  let n : ℕ := (initial_v - initial_q) / (increase_q + decrease_v)
  1990 + n

/-- Theorem stating that the budgets become equal in 1996 -/
theorem budgets_equal_in_1996 :
  year_budgets_equal 540000 780000 30000 10000 = 1996 := by
  sorry

end budgets_equal_in_1996_l3479_347994


namespace equation_solution_l3479_347964

theorem equation_solution :
  ∃ x : ℝ, (8 : ℝ) ^ (2 * x - 9) = 2 ^ (-2 * x - 3) ∧ x = 3 := by
  sorry

end equation_solution_l3479_347964


namespace least_prime_factor_of_11_5_minus_11_4_l3479_347914

theorem least_prime_factor_of_11_5_minus_11_4 :
  Nat.minFac (11^5 - 11^4) = 2 := by sorry

end least_prime_factor_of_11_5_minus_11_4_l3479_347914


namespace modified_geometric_progression_sum_of_squares_l3479_347980

/-- The sum of squares of a modified geometric progression -/
theorem modified_geometric_progression_sum_of_squares
  (b c s : ℝ) (h : abs s < 1) :
  let modifiedSum := (c^2 * b^2 * s^4) / (1 - s)
  let modifiedSequence := fun n => if n < 3 then b * s^(n-1) else c * b * s^(n-1)
  ∑' n, (modifiedSequence n)^2 = modifiedSum :=
sorry

end modified_geometric_progression_sum_of_squares_l3479_347980


namespace line_contains_point_l3479_347990

/-- The value of k for which the line 2 - 2kx = -4y contains the point (3, -2) -/
theorem line_contains_point (k : ℝ) : 
  (2 - 2 * k * 3 = -4 * (-2)) ↔ k = -1 := by sorry

end line_contains_point_l3479_347990


namespace candy_left_l3479_347936

theorem candy_left (houses : ℕ) (candies_per_house : ℕ) (people : ℕ) (candies_eaten_per_person : ℕ) : 
  houses = 15 → 
  candies_per_house = 8 → 
  people = 3 → 
  candies_eaten_per_person = 6 → 
  houses * candies_per_house - people * candies_eaten_per_person = 102 := by
sorry

end candy_left_l3479_347936


namespace completing_square_equivalence_l3479_347985

theorem completing_square_equivalence (x : ℝ) :
  (x^2 - 6*x + 4 = 0) ↔ ((x - 3)^2 = 5) := by
  sorry

end completing_square_equivalence_l3479_347985


namespace transformed_function_eq_g_l3479_347959

/-- The original quadratic function -/
def f (x : ℝ) : ℝ := 2 * (x - 3)^2 + 4

/-- The transformed quadratic function -/
def g (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

/-- The horizontal shift transformation -/
def shift_left (h : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + h)

/-- The vertical shift transformation -/
def shift_down (k : ℝ) (f : ℝ → ℝ) (x : ℝ) : ℝ := f x - k

/-- Theorem stating that the transformed function is equivalent to g -/
theorem transformed_function_eq_g :
  ∀ x, shift_down 3 (shift_left 2 f) x = g x := by sorry

end transformed_function_eq_g_l3479_347959


namespace nancy_football_games_l3479_347944

theorem nancy_football_games (games_this_month games_last_month games_next_month total_games : ℕ) :
  games_this_month = 9 →
  games_last_month = 8 →
  games_next_month = 7 →
  total_games = 24 →
  games_this_month + games_last_month + games_next_month = total_games :=
by sorry

end nancy_football_games_l3479_347944


namespace student_selection_theorem_l3479_347919

def number_of_boys : ℕ := 4
def number_of_girls : ℕ := 3
def total_to_select : ℕ := 3

theorem student_selection_theorem :
  (Nat.choose number_of_boys 2 * Nat.choose number_of_girls 1) +
  (Nat.choose number_of_boys 1 * Nat.choose number_of_girls 2) = 30 := by
  sorry

end student_selection_theorem_l3479_347919


namespace greatest_common_factor_372_72_under_50_l3479_347922

def is_greatest_common_factor (n : ℕ) : Prop :=
  n ∣ 372 ∧ n < 50 ∧ n ∣ 72 ∧
  ∀ m : ℕ, m ∣ 372 → m < 50 → m ∣ 72 → m ≤ n

theorem greatest_common_factor_372_72_under_50 :
  is_greatest_common_factor 12 := by
sorry

end greatest_common_factor_372_72_under_50_l3479_347922


namespace symmetric_points_sum_l3479_347965

/-- Given two points A and B that are symmetric with respect to the x-axis,
    prove that the sum of their y-coordinate and x-coordinate respectively is 5. -/
theorem symmetric_points_sum (a b : ℝ) : 
  (2 : ℝ) = b ∧ a = 3 → a + b = 5 := by sorry

end symmetric_points_sum_l3479_347965


namespace pizza_division_l3479_347948

theorem pizza_division (total_pizza : ℚ) (num_employees : ℕ) :
  total_pizza = 5 / 8 ∧ num_employees = 4 →
  total_pizza / num_employees = 5 / 32 := by
  sorry

end pizza_division_l3479_347948


namespace quadratic_equation_distinct_roots_l3479_347958

theorem quadratic_equation_distinct_roots (p q : ℚ) : 
  (∀ x : ℚ, x^2 + p*x + q = 0 ↔ x = 2*p ∨ x = p + q) ∧ 
  (2*p ≠ p + q) → 
  p = 2/3 ∧ q = -8/3 :=
sorry

end quadratic_equation_distinct_roots_l3479_347958


namespace coefficient_of_linear_term_l3479_347921

theorem coefficient_of_linear_term (a b c : ℝ) : 
  (fun x : ℝ => a * x^2 + b * x + c) = (fun x : ℝ => x^2 - 2*x + 3) → 
  b = -2 := by
sorry

end coefficient_of_linear_term_l3479_347921


namespace wire_length_for_square_field_l3479_347906

-- Define the area of the square field
def field_area : ℝ := 24336

-- Define the number of times the wire goes around the field
def num_rounds : ℕ := 13

-- Theorem statement
theorem wire_length_for_square_field :
  let side_length := Real.sqrt field_area
  let perimeter := 4 * side_length
  let wire_length := num_rounds * perimeter
  wire_length = 8112 := by sorry

end wire_length_for_square_field_l3479_347906


namespace triangle_area_and_perimeter_l3479_347932

theorem triangle_area_and_perimeter 
  (DE FD : ℝ) 
  (h_DE : DE = 12) 
  (h_FD : FD = 20) 
  (h_right_angle : DE * FD = 2 * (1/2 * DE * FD)) : 
  let EF := Real.sqrt (DE^2 + FD^2)
  (1/2 * DE * FD = 120) ∧ (DE + FD + EF = 32 + 2 * Real.sqrt 136) :=
by sorry

end triangle_area_and_perimeter_l3479_347932


namespace max_ages_for_given_params_l3479_347902

/-- Calculates the maximum number of different integer ages within one standard deviation of the average age. -/
def max_different_ages (average_age : ℤ) (std_dev : ℤ) : ℕ :=
  let lower_bound := average_age - std_dev
  let upper_bound := average_age + std_dev
  (upper_bound - lower_bound + 1).toNat

/-- Theorem stating that for an average age of 10 and standard deviation of 8,
    the maximum number of different integer ages within one standard deviation is 17. -/
theorem max_ages_for_given_params :
  max_different_ages 10 8 = 17 := by
  sorry

#eval max_different_ages 10 8

end max_ages_for_given_params_l3479_347902


namespace units_digit_of_27_times_36_l3479_347908

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_27_times_36 : unitsDigit (27 * 36) = 2 := by
  sorry

end units_digit_of_27_times_36_l3479_347908


namespace max_min_values_l3479_347960

-- Define the constraint function
def constraint (x y : ℝ) : Prop :=
  |5 * x + y| + |5 * x - y| = 20

-- Define the expression to be maximized/minimized
def expr (x y : ℝ) : ℝ :=
  x^2 - x*y + y^2

-- Statement of the theorem
theorem max_min_values :
  (∃ x y : ℝ, constraint x y ∧ expr x y = 124) ∧
  (∃ x y : ℝ, constraint x y ∧ expr x y = 3) ∧
  (∀ x y : ℝ, constraint x y → 3 ≤ expr x y ∧ expr x y ≤ 124) :=
sorry

end max_min_values_l3479_347960


namespace alicia_score_l3479_347953

theorem alicia_score (total_score : ℕ) (other_players : ℕ) (avg_score : ℕ) (alicia_score : ℕ) : 
  total_score = 75 →
  other_players = 8 →
  avg_score = 6 →
  total_score = other_players * avg_score + alicia_score →
  alicia_score = 27 := by
sorry

end alicia_score_l3479_347953


namespace sum_of_differences_l3479_347983

def S : Finset ℕ := Finset.range 9

def diff_sum (s : Finset ℕ) : ℕ :=
  s.sum (fun i => s.sum (fun j => if i > j then 2^i - 2^j else 0))

theorem sum_of_differences : diff_sum S = 3096 := by
  sorry

end sum_of_differences_l3479_347983


namespace smallest_n_divisible_by_two_primes_l3479_347971

/-- A function that returns true if a number is divisible by at least two different primes -/
def divisible_by_two_primes (x : ℕ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ∣ x ∧ q ∣ x

/-- The theorem stating that 5 is the smallest positive integer n ≥ 5 such that n^2 - n + 6 is divisible by at least two different primes -/
theorem smallest_n_divisible_by_two_primes :
  ∀ n : ℕ, n ≥ 5 → (divisible_by_two_primes (n^2 - n + 6) → n ≥ 5) ∧
  (n = 5 → divisible_by_two_primes (5^2 - 5 + 6)) :=
by sorry

#check smallest_n_divisible_by_two_primes

end smallest_n_divisible_by_two_primes_l3479_347971


namespace camp_average_age_of_adults_l3479_347911

theorem camp_average_age_of_adults 
  (total_members : ℕ) 
  (overall_average : ℝ) 
  (num_girls num_boys num_adults : ℕ) 
  (avg_age_girls avg_age_boys : ℝ) 
  (h1 : total_members = 40)
  (h2 : overall_average = 17)
  (h3 : num_girls = 20)
  (h4 : num_boys = 15)
  (h5 : num_adults = 5)
  (h6 : avg_age_girls = 15)
  (h7 : avg_age_boys = 16)
  (h8 : total_members = num_girls + num_boys + num_adults) :
  (total_members : ℝ) * overall_average - 
  (num_girls : ℝ) * avg_age_girls - 
  (num_boys : ℝ) * avg_age_boys = 
  (num_adults : ℝ) * 28 :=
by sorry

end camp_average_age_of_adults_l3479_347911


namespace regular_triangular_pyramid_volume_l3479_347942

/-- 
Given a regular triangular pyramid with angle α between a lateral edge and a side of the base,
and a cross-section of area S made through the midpoint of a lateral edge parallel to the lateral face,
the volume V of the pyramid is (8√3 S cos²α) / (3 sin(2α)), where π/6 < α < π/2.
-/
theorem regular_triangular_pyramid_volume 
  (α : Real) 
  (S : Real) 
  (h1 : π/6 < α) 
  (h2 : α < π/2) 
  (h3 : S > 0) : 
  ∃ V : Real, V = (8 * Real.sqrt 3 * S * (Real.cos α)^2) / (3 * Real.sin (2 * α)) := by
  sorry

#check regular_triangular_pyramid_volume

end regular_triangular_pyramid_volume_l3479_347942


namespace average_weight_of_students_l3479_347900

theorem average_weight_of_students (girls_count boys_count : ℕ) 
  (girls_avg_weight boys_avg_weight : ℝ) :
  girls_count = 5 →
  boys_count = 5 →
  girls_avg_weight = 45 →
  boys_avg_weight = 55 →
  let total_count := girls_count + boys_count
  let total_weight := girls_count * girls_avg_weight + boys_count * boys_avg_weight
  (total_weight / total_count : ℝ) = 50 := by
sorry

end average_weight_of_students_l3479_347900


namespace triangle_side_lengths_l3479_347920

theorem triangle_side_lengths (a b c : ℝ) (angleC : ℝ) (area : ℝ) :
  a = 3 →
  angleC = 2 * Real.pi / 3 →
  area = 3 * Real.sqrt 3 / 4 →
  1/2 * a * b * Real.sin angleC = area →
  Real.cos angleC = (a^2 + b^2 - c^2) / (2 * a * b) →
  b = 1 ∧ c = Real.sqrt 13 := by
  sorry


end triangle_side_lengths_l3479_347920


namespace pencil_distribution_l3479_347907

/-- The number of ways to distribute n identical objects among k people, 
    where each person gets at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 8 identical pencils among 4 friends, 
    where each friend has at least one pencil. -/
theorem pencil_distribution : distribute 8 4 = 35 := by sorry

end pencil_distribution_l3479_347907


namespace rogers_reading_rate_l3479_347986

/-- Roger's book reading problem -/
theorem rogers_reading_rate (total_books : ℕ) (weeks : ℕ) (books_per_week : ℕ) 
  (h1 : total_books = 30)
  (h2 : weeks = 5)
  (h3 : books_per_week * weeks = total_books) :
  books_per_week = 6 := by
sorry

end rogers_reading_rate_l3479_347986


namespace four_numbers_sum_product_l3479_347992

/-- Given four real numbers x₁, x₂, x₃, x₄, if the sum of any one number and the product 
    of the other three is equal to 2, then the only possible solutions are 
    (1, 1, 1, 1) and (-1, -1, -1, 3) and its permutations. -/
theorem four_numbers_sum_product (x₁ x₂ x₃ x₄ : ℝ) : 
  (x₁ + x₂ * x₃ * x₄ = 2) ∧ 
  (x₂ + x₃ * x₄ * x₁ = 2) ∧ 
  (x₃ + x₄ * x₁ * x₂ = 2) ∧ 
  (x₄ + x₁ * x₂ * x₃ = 2) →
  ((x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1 ∧ x₄ = 1) ∨
   (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = 3) ∨
   (x₁ = -1 ∧ x₂ = -1 ∧ x₃ = 3 ∧ x₄ = -1) ∨
   (x₁ = -1 ∧ x₂ = 3 ∧ x₃ = -1 ∧ x₄ = -1) ∨
   (x₁ = 3 ∧ x₂ = -1 ∧ x₃ = -1 ∧ x₄ = -1)) :=
by sorry

end four_numbers_sum_product_l3479_347992


namespace max_pairs_sum_l3479_347968

theorem max_pairs_sum (k : ℕ) (a b : ℕ → ℕ) : 
  (∀ i : ℕ, i < k → a i < b i) →
  (∀ i j : ℕ, i < k → j < k → i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j) →
  (∀ i : ℕ, i < k → a i ∈ Finset.range 4019 ∧ b i ∈ Finset.range 4019) →
  (∀ i : ℕ, i < k → a i + b i ≤ 4019) →
  (∀ i j : ℕ, i < k → j < k → i ≠ j → a i + b i ≠ a j + b j) →
  k ≤ 1607 :=
sorry

end max_pairs_sum_l3479_347968


namespace answer_key_combinations_l3479_347926

/-- The number of ways to answer a single true-false question -/
def true_false_options : ℕ := 2

/-- The number of true-false questions in the quiz -/
def num_true_false : ℕ := 4

/-- The number of ways to answer a single multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- The number of multiple-choice questions in the quiz -/
def num_multiple_choice : ℕ := 2

/-- The total number of possible answer combinations for true-false questions -/
def total_true_false_combinations : ℕ := true_false_options ^ num_true_false

/-- The number of invalid true-false combinations (all true or all false) -/
def invalid_true_false_combinations : ℕ := 2

/-- The number of valid true-false combinations -/
def valid_true_false_combinations : ℕ := total_true_false_combinations - invalid_true_false_combinations

/-- The number of ways to answer all multiple-choice questions -/
def multiple_choice_combinations : ℕ := multiple_choice_options ^ num_multiple_choice

/-- The total number of ways to create an answer key for the quiz -/
def total_answer_key_combinations : ℕ := valid_true_false_combinations * multiple_choice_combinations

theorem answer_key_combinations : total_answer_key_combinations = 224 := by
  sorry

end answer_key_combinations_l3479_347926


namespace instantaneous_velocity_at_3_seconds_l3479_347951

-- Define the motion equation
def s (t : ℝ) : ℝ := 1 - t + t^2

-- Define the instantaneous velocity function
def v (t : ℝ) : ℝ := -1 + 2*t

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 5 := by
  sorry

end instantaneous_velocity_at_3_seconds_l3479_347951


namespace f_inequality_l3479_347925

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem f_inequality (h1 : f 1 = 1) (h2 : ∀ x, deriv f x < 2) :
  ∀ x, f x < 2 * x - 1 ↔ x > 1 := by
  sorry

end f_inequality_l3479_347925


namespace sandy_change_proof_l3479_347973

/-- Calculates the change received from a purchase given the payment amount and the costs of individual items. -/
def calculate_change (payment : ℚ) (item1_cost : ℚ) (item2_cost : ℚ) : ℚ :=
  payment - (item1_cost + item2_cost)

/-- Proves that given a $20 bill payment and purchases of $9.24 and $8.25, the change received is $2.51. -/
theorem sandy_change_proof :
  calculate_change 20 9.24 8.25 = 2.51 := by
  sorry

end sandy_change_proof_l3479_347973


namespace f_min_max_l3479_347984

-- Define the function
def f (x y : ℝ) : ℝ := 3 * x^2 * y - 2 * x * y^2

-- State the theorem
theorem f_min_max :
  (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → f x y ≥ -1/3) ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 → f x y ≤ 9/8) ∧
  (∃ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ f x y = -1/3) ∧
  (∃ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ f x y = 9/8) :=
by sorry

end f_min_max_l3479_347984


namespace marks_future_age_l3479_347943

def amy_age : ℕ := 15
def age_difference : ℕ := 7
def years_in_future : ℕ := 5

theorem marks_future_age :
  amy_age + age_difference + years_in_future = 27 := by
  sorry

end marks_future_age_l3479_347943


namespace optimal_time_correct_l3479_347995

/-- The optimal time for Vasya and Petya to cover the distance -/
def optimal_time : ℝ := 0.5

/-- The total distance to be covered -/
def total_distance : ℝ := 3

/-- Vasya's running speed -/
def vasya_run_speed : ℝ := 4

/-- Vasya's skating speed -/
def vasya_skate_speed : ℝ := 8

/-- Petya's running speed -/
def petya_run_speed : ℝ := 5

/-- Petya's skating speed -/
def petya_skate_speed : ℝ := 10

/-- Theorem stating that the optimal time is correct -/
theorem optimal_time_correct :
  ∃ (x : ℝ), 
    0 ≤ x ∧ x ≤ total_distance ∧
    (x / vasya_skate_speed + (total_distance - x) / vasya_run_speed = optimal_time) ∧
    ((total_distance - x) / petya_skate_speed + x / petya_run_speed = optimal_time) ∧
    ∀ (y : ℝ), 0 ≤ y ∧ y ≤ total_distance →
      max (y / vasya_skate_speed + (total_distance - y) / vasya_run_speed)
          ((total_distance - y) / petya_skate_speed + y / petya_run_speed) ≥ optimal_time :=
by
  sorry


end optimal_time_correct_l3479_347995


namespace total_spent_proof_l3479_347975

-- Define the original prices and discount rates
def tshirt_price : ℚ := 20
def tshirt_discount : ℚ := 0.4
def hat_price : ℚ := 15
def hat_discount : ℚ := 0.6
def accessory_price : ℚ := 10
def bracelet_discount : ℚ := 0.3
def belt_discount : ℚ := 0.5
def sales_tax : ℚ := 0.05

-- Define the number of friends and their purchases
def total_friends : ℕ := 4
def bracelet_buyers : ℕ := 1
def belt_buyers : ℕ := 3

-- Define the function to calculate discounted price
def discounted_price (original_price : ℚ) (discount : ℚ) : ℚ :=
  original_price * (1 - discount)

-- Define the theorem
theorem total_spent_proof :
  let tshirt_discounted := discounted_price tshirt_price tshirt_discount
  let hat_discounted := discounted_price hat_price hat_discount
  let bracelet_discounted := discounted_price accessory_price bracelet_discount
  let belt_discounted := discounted_price accessory_price belt_discount
  let bracelet_total := tshirt_discounted + hat_discounted + bracelet_discounted
  let belt_total := tshirt_discounted + hat_discounted + belt_discounted
  let subtotal := bracelet_total * bracelet_buyers + belt_total * belt_buyers
  let total := subtotal * (1 + sales_tax)
  total = 98.7 := by
    sorry

end total_spent_proof_l3479_347975


namespace largest_circular_pool_diameter_l3479_347969

/-- Given a rectangular garden with area 180 square meters and length three times its width,
    the diameter of the largest circular pool that can be outlined by the garden's perimeter
    is 16√15/π meters. -/
theorem largest_circular_pool_diameter (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 3 * width →
  width * length = 180 →
  (2 * (width + length)) / π = 16 * Real.sqrt 15 / π :=
by sorry

end largest_circular_pool_diameter_l3479_347969


namespace rogers_final_amount_l3479_347979

def rogers_money (initial : ℕ) (gift : ℕ) (spent : ℕ) : ℕ :=
  initial + gift - spent

theorem rogers_final_amount :
  rogers_money 16 28 25 = 19 := by
  sorry

end rogers_final_amount_l3479_347979


namespace problem_solution_l3479_347957

noncomputable def f (a x : ℝ) : ℝ := x + Real.exp (x - a)

noncomputable def g (a x : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem problem_solution (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ - g a x₀ = 3) → a = -Real.log 2 - 1 := by
  sorry

end problem_solution_l3479_347957


namespace remainder_theorem_l3479_347962

theorem remainder_theorem (n : ℕ) (h : n % 7 = 5) : (3 * n + 2)^2 % 11 = 3 := by
  sorry

end remainder_theorem_l3479_347962


namespace atomic_number_relation_l3479_347989

-- Define the compound Y₂X₃
structure Compound where
  X : ℕ  -- Atomic number of X
  Y : ℕ  -- Atomic number of Y

-- Define the property of X being a short-period non-metal element
def isShortPeriodNonMetal (x : ℕ) : Prop :=
  x ≤ 18  -- Assuming short-period elements have atomic numbers up to 18

-- Define the compound formation rule
def formsCompound (c : Compound) : Prop :=
  isShortPeriodNonMetal c.X ∧ c.Y > 0

-- Theorem statement
theorem atomic_number_relation (n : ℕ) :
  ∀ c : Compound, formsCompound c → c.X = n → c.Y ≠ n + 2 := by
  sorry

end atomic_number_relation_l3479_347989


namespace women_count_is_twenty_l3479_347949

/-- Represents a social event with dancing participants -/
structure DancingEvent where
  num_men : ℕ
  num_women : ℕ
  dances_per_man : ℕ
  dances_per_woman : ℕ

/-- The number of women at the event given the conditions -/
def women_count (event : DancingEvent) : ℕ :=
  (event.num_men * event.dances_per_man) / event.dances_per_woman

/-- Theorem stating that the number of women at the event is 20 -/
theorem women_count_is_twenty (event : DancingEvent) 
  (h1 : event.num_men = 15)
  (h2 : event.dances_per_man = 4)
  (h3 : event.dances_per_woman = 3) :
  women_count event = 20 := by
  sorry

end women_count_is_twenty_l3479_347949


namespace greatest_n_condition_l3479_347939

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k * k

def condition (n : ℕ) : Prop :=
  is_perfect_square (sum_of_squares n * (sum_of_squares (2 * n) - sum_of_squares n))

theorem greatest_n_condition :
  (1921 ≤ 2023) ∧ 
  condition 1921 ∧
  ∀ m : ℕ, (m > 1921 ∧ m ≤ 2023) → ¬(condition m) :=
sorry

end greatest_n_condition_l3479_347939


namespace rectangular_field_width_l3479_347998

/-- Proves that the width of a rectangular field is 1400/29 meters given specific conditions -/
theorem rectangular_field_width (w : ℝ) : 
  w > 0 → -- width is positive
  (2*w + 2*(7/5*w) + w = 280) → -- combined perimeter equation
  w = 1400/29 := by
sorry

end rectangular_field_width_l3479_347998


namespace bag_cost_theorem_l3479_347966

def total_money : ℕ := 50
def tshirt_cost : ℕ := 8
def keychain_cost : ℚ := 2 / 3
def tshirts_bought : ℕ := 2
def bags_bought : ℕ := 2
def keychains_bought : ℕ := 21

theorem bag_cost_theorem :
  ∃ (bag_cost : ℚ),
    bag_cost * bags_bought = 
      total_money - 
      (tshirt_cost * tshirts_bought) - 
      (keychain_cost * keychains_bought) ∧
    bag_cost = 10 := by
  sorry

end bag_cost_theorem_l3479_347966


namespace directed_segment_length_equal_l3479_347950

-- Define a vector space
variable {V : Type*} [NormedAddCommGroup V]

-- Define two points in the vector space
variable (M N : V)

-- Define the directed line segment from M to N
def directed_segment (M N : V) : V := N - M

-- Theorem statement
theorem directed_segment_length_equal :
  ‖directed_segment M N‖ = ‖directed_segment N M‖ := by sorry

end directed_segment_length_equal_l3479_347950


namespace bob_guaranteed_victory_l3479_347927

/-- Represents a grid in the game -/
def Grid := Matrix (Fin 2011) (Fin 2011) ℕ

/-- The size of the grid -/
def gridSize : ℕ := 2011

/-- The total number of grids Alice has -/
def aliceGridCount : ℕ := 2010

/-- Checks if a grid is valid (strictly increasing across rows and down columns) -/
def isValidGrid (g : Grid) : Prop :=
  ∀ i j k, i < j → g i k < g j k ∧ g k i < g k j

/-- Checks if two grids are different -/
def areDifferentGrids (g1 g2 : Grid) : Prop :=
  ∃ i j, g1 i j ≠ g2 i j

/-- Checks if Bob wins against a given grid -/
def bobWins (bobGrid aliceGrid : Grid) : Prop :=
  ∃ i j k, aliceGrid i j = bobGrid k i ∧ aliceGrid i k = bobGrid k j

/-- Theorem: Bob can guarantee victory with at most 1 swap -/
theorem bob_guaranteed_victory :
  ∃ (initialBobGrid : Grid) (swappedBobGrid : Grid),
    isValidGrid initialBobGrid ∧
    isValidGrid swappedBobGrid ∧
    (∀ (aliceGrids : Fin aliceGridCount → Grid),
      (∀ i, isValidGrid (aliceGrids i)) →
      (∀ i j, i ≠ j → areDifferentGrids (aliceGrids i) (aliceGrids j)) →
      (bobWins initialBobGrid (aliceGrids i) ∨
       bobWins swappedBobGrid (aliceGrids i))) :=
sorry

end bob_guaranteed_victory_l3479_347927


namespace min_sum_squares_l3479_347977

def S : Finset Int := {-8, -6, -4, -1, 1, 3, 7, 12}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (∀ a b c d e f g h : Int, 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧
    g ≠ h →
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S →
    (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 10) ∧
  (p + q + r + s)^2 + (t + u + v + w)^2 = 10 :=
sorry

end min_sum_squares_l3479_347977


namespace double_division_remainder_l3479_347976

def p (x : ℝ) : ℝ := x^10

def q1 (x : ℝ) : ℝ := 
  x^9 + 2*x^8 + 4*x^7 + 8*x^6 + 16*x^5 + 32*x^4 + 64*x^3 + 128*x^2 + 256*x + 512

theorem double_division_remainder (x : ℝ) : 
  ∃ (q2 : ℝ → ℝ) (r2 : ℝ), p x = (x - 2) * ((x - 2) * q2 x + q1 2) + r2 ∧ r2 = 5120 := by
  sorry

end double_division_remainder_l3479_347976


namespace flag_stripes_l3479_347928

theorem flag_stripes :
  ∀ (S : ℕ), 
    S > 0 →
    (10 * (1 + (S - 1) / 2 : ℚ) = 70) →
    S = 13 := by
  sorry

end flag_stripes_l3479_347928


namespace modified_cube_surface_area_l3479_347981

/-- Represents a cube with its dimensions -/
structure Cube where
  size : Nat

/-- Represents the large cube and its properties -/
structure LargeCube where
  size : Nat
  smallCubeSize : Nat
  totalSmallCubes : Nat

/-- Calculates the surface area of the modified structure -/
def calculateSurfaceArea (lc : LargeCube) : Nat :=
  sorry

/-- Theorem stating the surface area of the modified structure -/
theorem modified_cube_surface_area 
  (lc : LargeCube) 
  (h1 : lc.size = 12) 
  (h2 : lc.smallCubeSize = 3) 
  (h3 : lc.totalSmallCubes = 64) : 
  calculateSurfaceArea lc = 2454 := by
  sorry

end modified_cube_surface_area_l3479_347981


namespace usual_time_calculation_l3479_347941

/-- Proves that given a constant distance and the fact that at 60% of usual speed 
    it takes 35 minutes more, the usual time to cover the distance is 52.5 minutes. -/
theorem usual_time_calculation (distance : ℝ) (usual_speed : ℝ) (usual_time : ℝ) 
    (h1 : usual_speed > 0) 
    (h2 : usual_time > 0)
    (h3 : distance = usual_speed * usual_time)
    (h4 : distance = (0.6 * usual_speed) * (usual_time + 35/60)) :
  usual_time = 52.5 / 60 := by
sorry

end usual_time_calculation_l3479_347941


namespace no_non_zero_solution_l3479_347967

theorem no_non_zero_solution (a b c n : ℤ) :
  6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 := by
  sorry

end no_non_zero_solution_l3479_347967


namespace pentagon_probability_l3479_347918

/-- A type representing the points on the pentagon --/
inductive PentagonPoint
| Vertex : Fin 5 → PentagonPoint
| Midpoint : Fin 5 → PentagonPoint

/-- The total number of points on the pentagon --/
def total_points : ℕ := 10

/-- A function to determine if two points are exactly one side apart --/
def one_side_apart (p q : PentagonPoint) : Prop :=
  match p, q with
  | PentagonPoint.Vertex i, PentagonPoint.Vertex j => (j - i) % 5 = 2 ∨ (i - j) % 5 = 2
  | _, _ => False

/-- The number of ways to choose 2 points from the total points --/
def total_choices : ℕ := (total_points.choose 2)

/-- The number of point pairs that are one side apart --/
def favorable_choices : ℕ := 10

theorem pentagon_probability :
  (favorable_choices : ℚ) / total_choices = 2 / 9 := by sorry

end pentagon_probability_l3479_347918


namespace some_number_value_l3479_347935

theorem some_number_value (x : ℝ) : 65 + 5 * x / (180 / 3) = 66 → x = 12 := by
  sorry

end some_number_value_l3479_347935


namespace nineteenth_term_is_zero_l3479_347945

/-- A sequence with specific properties -/
def special_sequence (a : ℕ → ℚ) : Prop :=
  a 3 = 2 ∧ 
  a 7 = 1 ∧ 
  ∃ d : ℚ, ∀ n : ℕ, (1 / (a (n + 1) + 1) - 1 / (a n + 1)) = d

/-- The 19th term of the special sequence is 0 -/
theorem nineteenth_term_is_zero (a : ℕ → ℚ) (h : special_sequence a) : 
  a 19 = 0 := by
sorry

end nineteenth_term_is_zero_l3479_347945
