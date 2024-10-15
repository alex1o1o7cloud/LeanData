import Mathlib

namespace NUMINAMATH_CALUDE_cos_2017pi_over_6_l2481_248195

theorem cos_2017pi_over_6 : Real.cos (2017 * Real.pi / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_2017pi_over_6_l2481_248195


namespace NUMINAMATH_CALUDE_intersection_of_isosceles_and_right_angled_l2481_248129

-- Define the set of all triangles
def Triangle : Type := sorry

-- Define the property of being isosceles
def IsIsosceles (t : Triangle) : Prop := sorry

-- Define the property of being right-angled
def IsRightAngled (t : Triangle) : Prop := sorry

-- Define the set of isosceles triangles
def M : Set Triangle := {t : Triangle | IsIsosceles t}

-- Define the set of right-angled triangles
def N : Set Triangle := {t : Triangle | IsRightAngled t}

-- Define the property of being both isosceles and right-angled
def IsIsoscelesRightAngled (t : Triangle) : Prop := IsIsosceles t ∧ IsRightAngled t

-- Theorem statement
theorem intersection_of_isosceles_and_right_angled :
  M ∩ N = {t : Triangle | IsIsoscelesRightAngled t} := by sorry

end NUMINAMATH_CALUDE_intersection_of_isosceles_and_right_angled_l2481_248129


namespace NUMINAMATH_CALUDE_parabola_directrix_l2481_248105

/-- A parabola C with equation y² = mx passing through the point (-2, √3) has directrix x = 3/8 -/
theorem parabola_directrix (m : ℝ) : 
  (3 : ℝ) = m * (-2) → -- Condition: parabola passes through (-2, √3)
  (∀ x y : ℝ, y^2 = m*x → -- Definition of parabola C
    (x = 3/8 ↔ -- Equation of directrix
      ∃ p : ℝ × ℝ, 
        p.2^2 = m*p.1 ∧ -- Point on parabola
        (x - p.1)^2 = (y - p.2)^2 + (3/8 - x)^2)) -- Distance condition for directrix
  := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2481_248105


namespace NUMINAMATH_CALUDE_four_digit_multiples_of_five_l2481_248166

theorem four_digit_multiples_of_five : 
  (Finset.filter (fun n => n % 5 = 0) (Finset.range 9000)).card = 1800 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_multiples_of_five_l2481_248166


namespace NUMINAMATH_CALUDE_baby_grab_outcomes_l2481_248128

theorem baby_grab_outcomes (educational_items living_items entertainment_items : ℕ) 
  (h1 : educational_items = 4)
  (h2 : living_items = 3)
  (h3 : entertainment_items = 4) :
  educational_items + living_items + entertainment_items = 11 := by
  sorry

end NUMINAMATH_CALUDE_baby_grab_outcomes_l2481_248128


namespace NUMINAMATH_CALUDE_corresponding_sides_proportional_in_similar_triangles_l2481_248197

-- Define what it means for two triangles to be similar
def similar_triangles (t1 t2 : Triangle) : Prop := sorry

-- Define what it means for sides to be corresponding
def corresponding_sides (s1 : Segment) (t1 : Triangle) (s2 : Segment) (t2 : Triangle) : Prop := sorry

-- Define what it means for two segments to be proportional
def proportional (s1 s2 : Segment) : Prop := sorry

-- Theorem statement
theorem corresponding_sides_proportional_in_similar_triangles 
  (t1 t2 : Triangle) (s1 s3 : Segment) (s2 s4 : Segment) :
  similar_triangles t1 t2 →
  corresponding_sides s1 t1 s2 t2 →
  corresponding_sides s3 t1 s4 t2 →
  proportional s1 s2 ∧ proportional s3 s4 := by sorry

end NUMINAMATH_CALUDE_corresponding_sides_proportional_in_similar_triangles_l2481_248197


namespace NUMINAMATH_CALUDE_number_of_students_in_class_l2481_248171

/-- Represents the problem of calculating the number of students in a class based on their payments for a science project. -/
theorem number_of_students_in_class
  (full_payment : ℕ)
  (half_payment : ℕ)
  (num_half_payers : ℕ)
  (total_collected : ℕ)
  (h1 : full_payment = 50)
  (h2 : half_payment = 25)
  (h3 : num_half_payers = 4)
  (h4 : total_collected = 1150) :
  ∃ (num_students : ℕ),
    num_students * full_payment - num_half_payers * (full_payment - half_payment) = total_collected ∧
    num_students = 25 :=
by sorry

end NUMINAMATH_CALUDE_number_of_students_in_class_l2481_248171


namespace NUMINAMATH_CALUDE_largest_guaranteed_divisor_l2481_248157

def die_numbers : Finset Nat := {1, 2, 3, 4, 5, 6}

def is_valid_product (P : Nat) : Prop :=
  ∃ (S : Finset Nat), S ⊆ die_numbers ∧ S.card = 5 ∧ P = S.prod id

theorem largest_guaranteed_divisor :
  ∀ P, is_valid_product P → (12 ∣ P) ∧ ∀ n, n > 12 → ¬∀ Q, is_valid_product Q → (n ∣ Q) :=
by sorry

end NUMINAMATH_CALUDE_largest_guaranteed_divisor_l2481_248157


namespace NUMINAMATH_CALUDE_laptop_price_increase_l2481_248121

theorem laptop_price_increase (P₀ : ℝ) : 
  let P₂ := P₀ * (1 + 0.06)^2
  P₂ > 56358 :=
by
  sorry

#check laptop_price_increase

end NUMINAMATH_CALUDE_laptop_price_increase_l2481_248121


namespace NUMINAMATH_CALUDE_factorial_10_mod_13_l2481_248124

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem factorial_10_mod_13 : factorial 10 % 13 = 6 := by sorry

end NUMINAMATH_CALUDE_factorial_10_mod_13_l2481_248124


namespace NUMINAMATH_CALUDE_trains_distance_before_meeting_l2481_248104

/-- The distance between two trains one hour before they meet -/
def distance_before_meeting (speed_A speed_B : ℝ) : ℝ :=
  speed_A + speed_B

theorem trains_distance_before_meeting 
  (speed_A speed_B total_distance : ℝ)
  (h1 : speed_A = 60)
  (h2 : speed_B = 40)
  (h3 : total_distance ≤ 250) :
  distance_before_meeting speed_A speed_B = 100 := by
  sorry

#check trains_distance_before_meeting

end NUMINAMATH_CALUDE_trains_distance_before_meeting_l2481_248104


namespace NUMINAMATH_CALUDE_perimeter_PQR_l2481_248126

/-- Represents a triangle with three points -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculates the distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Calculates the perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ :=
  distance t.A t.B + distance t.B t.C + distance t.C t.A

/-- Theorem: Perimeter of PQR in the given configuration -/
theorem perimeter_PQR (ABC : Triangle)
  (P : ℝ × ℝ) (Q : ℝ × ℝ) (R : ℝ × ℝ)
  (h_AB : distance ABC.A ABC.B = 13)
  (h_BC : distance ABC.B ABC.C = 14)
  (h_CA : distance ABC.C ABC.A = 15)
  (h_P_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • ABC.B + t • ABC.C)
  (h_Q_on_CA : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • ABC.C + t • ABC.A)
  (h_R_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ R = (1 - t) • ABC.A + t • ABC.B)
  (h_equal_perimeters : perimeter ⟨ABC.A, Q, R⟩ = perimeter ⟨ABC.B, P, R⟩ ∧
                        perimeter ⟨ABC.A, Q, R⟩ = perimeter ⟨ABC.C, P, Q⟩)
  (h_ratio : perimeter ⟨ABC.A, Q, R⟩ = 4/5 * perimeter ⟨P, Q, R⟩) :
  perimeter ⟨P, Q, R⟩ = 30 := by sorry

end NUMINAMATH_CALUDE_perimeter_PQR_l2481_248126


namespace NUMINAMATH_CALUDE_train_speed_problem_l2481_248196

theorem train_speed_problem (distance : ℝ) (time_diff : ℝ) (speed_increase : ℝ) 
  (h1 : distance = 600)
  (h2 : time_diff = 4)
  (h3 : speed_increase = 12) :
  ∃ (normal_speed : ℝ),
    normal_speed > 0 ∧
    (distance / normal_speed) - (distance / (normal_speed + speed_increase)) = time_diff ∧
    normal_speed = 37 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l2481_248196


namespace NUMINAMATH_CALUDE_mixed_solution_purity_l2481_248135

/-- Calculates the purity of a mixed solution given two initial solutions with different purities -/
theorem mixed_solution_purity
  (purity1 purity2 : ℚ)
  (volume1 volume2 : ℚ)
  (h1 : purity1 = 30 / 100)
  (h2 : purity2 = 60 / 100)
  (h3 : volume1 = 40)
  (h4 : volume2 = 20)
  (h5 : volume1 + volume2 = 60) :
  (purity1 * volume1 + purity2 * volume2) / (volume1 + volume2) = 40 / 100 := by
  sorry

#check mixed_solution_purity

end NUMINAMATH_CALUDE_mixed_solution_purity_l2481_248135


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l2481_248174

/-- The equation of a circle in polar coordinates -/
def circle_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

/-- The equation of a line in polar coordinates -/
def line_equation (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

/-- Theorem stating that the line is tangent to the circle -/
theorem line_tangent_to_circle :
  ∃ (ρ₀ θ₀ : ℝ), circle_equation ρ₀ θ₀ ∧ line_equation ρ₀ θ₀ ∧
  ∀ (ρ θ : ℝ), circle_equation ρ θ ∧ line_equation ρ θ → (ρ, θ) = (ρ₀, θ₀) :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l2481_248174


namespace NUMINAMATH_CALUDE_playground_insects_l2481_248132

def remaining_insects (spiders ants initial_ladybugs departed_ladybugs : ℕ) : ℕ :=
  spiders + ants + initial_ladybugs - departed_ladybugs

theorem playground_insects :
  remaining_insects 3 12 8 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_playground_insects_l2481_248132


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2481_248188

theorem diophantine_equation_solutions :
  ∀ x y : ℕ, 2 * y^2 - x * y - x^2 + 2 * y + 7 * x - 84 = 0 ↔ (x = 1 ∧ y = 6) ∨ (x = 14 ∧ y = 13) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2481_248188


namespace NUMINAMATH_CALUDE_sector_central_angle_l2481_248107

theorem sector_central_angle (area : Real) (radius : Real) (h1 : area = 3 * Real.pi / 8) (h2 : radius = 1) :
  let central_angle := 2 * area / (radius ^ 2)
  central_angle = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2481_248107


namespace NUMINAMATH_CALUDE_min_value_expression_l2481_248186

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let expr := (|2*a - b + 2*a*(b - a)| + |b + 2*a - a*(b + 4*a)|) / Real.sqrt (4*a^2 + b^2)
  ∃ (min_val : ℝ), (∀ (a' b' : ℝ), a' > 0 → b' > 0 → expr ≥ min_val) ∧ min_val = Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2481_248186


namespace NUMINAMATH_CALUDE_right_handed_players_count_l2481_248158

theorem right_handed_players_count (total_players : ℕ) (throwers : ℕ) : 
  total_players = 70 →
  throwers = 46 →
  (total_players - throwers) / 3 * 2 + throwers = 62 :=
by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l2481_248158


namespace NUMINAMATH_CALUDE_jimmy_has_five_figures_l2481_248190

/-- Represents the collection of action figures Jimmy has --/
structure ActionFigures where
  regular : ℕ  -- number of regular figures worth $15
  special : ℕ  -- number of special figures worth $20
  h_special : special = 1  -- there is exactly one special figure

/-- The total value of the collection before the price reduction --/
def total_value (af : ActionFigures) : ℕ :=
  15 * af.regular + 20 * af.special

/-- The total earnings after selling all figures with $5 discount --/
def total_earnings (af : ActionFigures) : ℕ :=
  10 * af.regular + 15 * af.special

/-- Theorem stating that Jimmy has 5 action figures in total --/
theorem jimmy_has_five_figures :
  ∃ (af : ActionFigures), total_earnings af = 55 ∧ af.regular + af.special = 5 :=
sorry

end NUMINAMATH_CALUDE_jimmy_has_five_figures_l2481_248190


namespace NUMINAMATH_CALUDE_discriminant_divisibility_l2481_248164

theorem discriminant_divisibility (a b : ℝ) (n : ℤ) : 
  (∃ x₁ x₂ : ℝ, (2018 * x₁^2 + a * x₁ + b = 0) ∧ 
                (2018 * x₂^2 + a * x₂ + b = 0) ∧ 
                (x₁ - x₂ = n)) → 
  ∃ k : ℤ, a^2 - 4 * 2018 * b = 2018^2 * k := by
sorry

end NUMINAMATH_CALUDE_discriminant_divisibility_l2481_248164


namespace NUMINAMATH_CALUDE_original_people_count_l2481_248141

theorem original_people_count (x : ℕ) : 
  (x / 2 : ℕ) = 18 → 
  x = 36 :=
by
  sorry

#check original_people_count

end NUMINAMATH_CALUDE_original_people_count_l2481_248141


namespace NUMINAMATH_CALUDE_jacket_price_restoration_l2481_248176

theorem jacket_price_restoration (initial_price : ℝ) (h_pos : initial_price > 0) :
  let price_after_first_reduction := initial_price * (1 - 0.15)
  let price_after_second_reduction := price_after_first_reduction * (1 - 0.30)
  let required_increase := (initial_price / price_after_second_reduction) - 1
  abs (required_increase - 0.6807) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_restoration_l2481_248176


namespace NUMINAMATH_CALUDE_circle_equation_with_diameter_l2481_248117

def M : ℝ × ℝ := (-2, 0)
def N : ℝ × ℝ := (2, 0)

theorem circle_equation_with_diameter (x y : ℝ) :
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 16 →
  (x - (M.1 + N.1) / 2)^2 + (y - (M.2 + N.2) / 2)^2 = ((M.1 - N.1)^2 + (M.2 - N.2)^2) / 4 ↔
  x^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_with_diameter_l2481_248117


namespace NUMINAMATH_CALUDE_difference_of_numbers_l2481_248161

theorem difference_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 580)
  (ratio_eq : x / y = 0.75) : 
  y - x = 83 := by
sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l2481_248161


namespace NUMINAMATH_CALUDE_ashley_cocktail_calories_l2481_248131

/-- Represents the ingredients of Ashley's cocktail -/
structure Cocktail :=
  (mango_juice : ℝ)
  (honey : ℝ)
  (water : ℝ)
  (vodka : ℝ)

/-- Calculates the total calories in the cocktail -/
def total_calories (c : Cocktail) : ℝ :=
  c.mango_juice * 0.6 + c.honey * 6.4 + c.vodka * 0.7

/-- Calculates the total weight of the cocktail -/
def total_weight (c : Cocktail) : ℝ :=
  c.mango_juice + c.honey + c.water + c.vodka

/-- Ashley's cocktail recipe -/
def ashley_cocktail : Cocktail :=
  { mango_juice := 150
  , honey := 200
  , water := 300
  , vodka := 100 }

/-- Theorem stating that 300g of Ashley's cocktail contains 576 calories -/
theorem ashley_cocktail_calories :
  (300 / total_weight ashley_cocktail) * total_calories ashley_cocktail = 576 := by
  sorry


end NUMINAMATH_CALUDE_ashley_cocktail_calories_l2481_248131


namespace NUMINAMATH_CALUDE_calculate_fraction_l2481_248137

/-- A function satisfying the given property for all real numbers -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, b^2 * f a = a^2 * f b

/-- The main theorem stating the result of the calculation -/
theorem calculate_fraction (f : ℝ → ℝ) (h1 : special_function f) (h2 : f 2 ≠ 0) :
  (f 6 - f 3) / f 2 = 6.75 := by
  sorry

end NUMINAMATH_CALUDE_calculate_fraction_l2481_248137


namespace NUMINAMATH_CALUDE_prop_1_prop_2_prop_3_prop_4_all_props_correct_l2481_248112

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Proposition 1
theorem prop_1 (m : ℝ) (a b : V) : m • (a - b) = m • a - m • b := by sorry

-- Proposition 2
theorem prop_2 (m n : ℝ) (a : V) : (m - n) • a = m • a - n • a := by sorry

-- Proposition 3
theorem prop_3 (m : ℝ) (a b : V) (h : m ≠ 0) : m • a = m • b → a = b := by sorry

-- Proposition 4
theorem prop_4 (m n : ℝ) (a : V) (h : a ≠ 0) : m • a = n • a → m = n := by sorry

-- All propositions are correct
theorem all_props_correct : 
  (∀ m : ℝ, ∀ a b : V, m • (a - b) = m • a - m • b) ∧
  (∀ m n : ℝ, ∀ a : V, (m - n) • a = m • a - n • a) ∧
  (∀ m : ℝ, ∀ a b : V, m ≠ 0 → (m • a = m • b → a = b)) ∧
  (∀ m n : ℝ, ∀ a : V, a ≠ 0 → (m • a = n • a → m = n)) := by sorry

end NUMINAMATH_CALUDE_prop_1_prop_2_prop_3_prop_4_all_props_correct_l2481_248112


namespace NUMINAMATH_CALUDE_ancient_chinese_math_problem_l2481_248150

theorem ancient_chinese_math_problem (x y : ℕ) : 
  (8 * x = y + 3) → (7 * x = y - 4) → ((y + 3) / 8 : ℚ) = ((y - 4) / 7 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ancient_chinese_math_problem_l2481_248150


namespace NUMINAMATH_CALUDE_field_trip_vans_l2481_248182

/-- The number of vans needed for a field trip --/
def vans_needed (students : ℕ) (adults : ℕ) (van_capacity : ℕ) : ℕ :=
  ((students + adults + van_capacity - 1) / van_capacity : ℕ)

/-- Theorem: For 33 students, 9 adults, and vans with capacity 7, 6 vans are needed --/
theorem field_trip_vans : vans_needed 33 9 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_vans_l2481_248182


namespace NUMINAMATH_CALUDE_eight_percent_problem_l2481_248109

theorem eight_percent_problem (x : ℝ) : (8 / 100) * x = 64 → x = 800 := by
  sorry

end NUMINAMATH_CALUDE_eight_percent_problem_l2481_248109


namespace NUMINAMATH_CALUDE_circle_points_count_l2481_248159

/-- A circle with n equally spaced points, labeled from 1 to n. -/
structure LabeledCircle where
  n : ℕ
  points : Fin n → ℕ
  labeled_from_1_to_n : ∀ i, points i = i.val + 1

/-- Two points are diametrically opposite if their distance is half the total number of points. -/
def diametrically_opposite (c : LabeledCircle) (i j : Fin c.n) : Prop :=
  (j.val - i.val) % c.n = c.n / 2

/-- The main theorem: if points 7 and 35 are diametrically opposite in a labeled circle, then n = 56. -/
theorem circle_points_count (c : LabeledCircle) 
  (h : ∃ (i j : Fin c.n), c.points i = 7 ∧ c.points j = 35 ∧ diametrically_opposite c i j) : 
  c.n = 56 := by
  sorry

end NUMINAMATH_CALUDE_circle_points_count_l2481_248159


namespace NUMINAMATH_CALUDE_x_plus_y_values_l2481_248110

theorem x_plus_y_values (x y : ℝ) 
  (hx : |x| = 2) 
  (hy : |y| = 5) 
  (hxy : x * y < 0) : 
  (x + y = -3) ∨ (x + y = 3) := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l2481_248110


namespace NUMINAMATH_CALUDE_triangle_reflection_translation_l2481_248116

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the reflection across y-axis operation
def reflectAcrossYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

-- Define the translation upwards operation
def translateUpwards (p : Point2D) (units : ℝ) : Point2D :=
  { x := p.x, y := p.y + units }

-- Define the combined operation
def reflectAndTranslate (p : Point2D) (units : ℝ) : Point2D :=
  translateUpwards (reflectAcrossYAxis p) units

-- Theorem statement
theorem triangle_reflection_translation :
  let D : Point2D := { x := 3, y := 4 }
  let E : Point2D := { x := 5, y := 6 }
  let F : Point2D := { x := 5, y := 1 }
  let F' : Point2D := reflectAndTranslate F 3
  F'.x = -5 ∧ F'.y = 4 := by sorry

end NUMINAMATH_CALUDE_triangle_reflection_translation_l2481_248116


namespace NUMINAMATH_CALUDE_bobby_candy_total_l2481_248106

/-- The total number of pieces of candy Bobby ate over two days -/
def total_candy (initial : ℕ) (next_day : ℕ) : ℕ :=
  initial + next_day

/-- Theorem stating that Bobby ate 241 pieces of candy in total -/
theorem bobby_candy_total : total_candy 89 152 = 241 := by
  sorry

end NUMINAMATH_CALUDE_bobby_candy_total_l2481_248106


namespace NUMINAMATH_CALUDE_ceiling_times_self_156_l2481_248147

theorem ceiling_times_self_156 :
  ∃! x : ℝ, ⌈x⌉ * x = 156 ∧ x = 12 := by sorry

end NUMINAMATH_CALUDE_ceiling_times_self_156_l2481_248147


namespace NUMINAMATH_CALUDE_point_on_x_axis_l2481_248152

/-- 
If a point P(a+2, a-3) lies on the x-axis, then its coordinates are (5, 0).
-/
theorem point_on_x_axis (a : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = a + 2 ∧ P.2 = a - 3 ∧ P.2 = 0) → 
  (∃ P : ℝ × ℝ, P = (5, 0)) :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l2481_248152


namespace NUMINAMATH_CALUDE_probability_diamond_then_ace_or_king_l2481_248123

/-- The number of cards in a combined deck of two standard decks -/
def total_cards : ℕ := 104

/-- The number of diamond cards in a combined deck of two standard decks -/
def diamond_cards : ℕ := 26

/-- The number of ace or king cards in a combined deck of two standard decks -/
def ace_or_king_cards : ℕ := 16

/-- The number of diamond cards that are not ace or king -/
def non_ace_king_diamond : ℕ := 22

/-- The number of diamond cards that are ace or king -/
def ace_king_diamond : ℕ := 4

theorem probability_diamond_then_ace_or_king :
  (diamond_cards * ace_or_king_cards - ace_king_diamond) / (total_cards * (total_cards - 1)) = 103 / 2678 := by
  sorry

end NUMINAMATH_CALUDE_probability_diamond_then_ace_or_king_l2481_248123


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2481_248178

-- Define the polynomials A, B, and C
def A (x : ℝ) : ℝ := 5 * x^2 + 4 * x - 1
def B (x : ℝ) : ℝ := -x^2 - 3 * x + 3
def C (x : ℝ) : ℝ := 8 - 7 * x - 6 * x^2

-- Theorem statement
theorem polynomial_simplification (x : ℝ) : A x - B x + C x = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2481_248178


namespace NUMINAMATH_CALUDE_expression_simplification_l2481_248103

theorem expression_simplification (a y : ℝ) : 
  ((1 : ℝ) * (3 * a^2 - 2 * a) + 2 * (a^2 - a + 2) = 5 * a^2 - 4 * a + 4) ∧ 
  ((2 : ℝ) * (2 * y^2 - 1/2 + 3 * y) - 2 * (y - y^2 + 1/2) = 4 * y^2 + y - 3/2) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2481_248103


namespace NUMINAMATH_CALUDE_unique_positive_solution_l2481_248139

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ Real.cos (Real.arcsin (Real.tan (Real.arccos x))) = x :=
by
  -- The unique solution is x = 1
  use 1
  constructor
  · -- Prove that x = 1 satisfies the equation
    sorry
  · -- Prove that x = 1 is the only positive solution
    sorry

#check unique_positive_solution

end NUMINAMATH_CALUDE_unique_positive_solution_l2481_248139


namespace NUMINAMATH_CALUDE_min_sum_squares_l2481_248113

theorem min_sum_squares (x y : ℝ) (h : (x - 2)^2 + (y - 3)^2 = 1) : 
  ∃ (z : ℝ), z = x^2 + y^2 ∧ (∀ (a b : ℝ), (a - 2)^2 + (b - 3)^2 = 1 → a^2 + b^2 ≥ z) ∧ z = 14 - 2 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2481_248113


namespace NUMINAMATH_CALUDE_median_intersection_locus_l2481_248122

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a trihedral angle -/
structure TrihedralAngle where
  vertex : Point3D
  edge1 : Point3D → Prop
  edge2 : Point3D → Prop
  edge3 : Point3D → Prop

/-- The locus of median intersections in a trihedral angle -/
def medianIntersectionLocus (angle : TrihedralAngle) (A : Point3D) : Plane3D :=
  sorry

/-- Main theorem: The locus of median intersections is a plane parallel to OBC and 1/3 away from A -/
theorem median_intersection_locus 
  (angle : TrihedralAngle) 
  (A : Point3D) 
  (h1 : angle.edge1 A) :
  ∃ (plane : Plane3D),
    (medianIntersectionLocus angle A = plane) ∧ 
    (∃ (B C : Point3D), 
      angle.edge2 B ∧ 
      angle.edge3 C ∧ 
      (plane.a * B.x + plane.b * B.y + plane.c * B.z + plane.d = 0) ∧
      (plane.a * C.x + plane.b * C.y + plane.c * C.z + plane.d = 0)) ∧
    (∃ (k : ℝ), k = 1/3 ∧ 
      (plane.a * A.x + plane.b * A.y + plane.c * A.z + plane.d = k * 
       (plane.a * angle.vertex.x + plane.b * angle.vertex.y + plane.c * angle.vertex.z + plane.d))) :=
by sorry

end NUMINAMATH_CALUDE_median_intersection_locus_l2481_248122


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l2481_248111

theorem reciprocal_of_negative_2023 :
  (1 : ℚ) / (-2023 : ℚ) = -(1 / 2023) := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l2481_248111


namespace NUMINAMATH_CALUDE_cosine_power_sum_l2481_248114

theorem cosine_power_sum (θ : ℝ) (x : ℂ) (n : ℤ) 
  (h1 : 0 < θ) (h2 : θ < π) (h3 : x + 1/x = 2 * Real.cos θ) : 
  x^n + 1/x^n = 2 * Real.cos (n * θ) := by
  sorry

end NUMINAMATH_CALUDE_cosine_power_sum_l2481_248114


namespace NUMINAMATH_CALUDE_min_value_x2_plus_y2_l2481_248138

theorem min_value_x2_plus_y2 (x y : ℝ) (h : x^2 + 2*x*y - y^2 = 7) :
  ∃ (m : ℝ), m = (7 * Real.sqrt 2) / 2 ∧ x^2 + y^2 ≥ m ∧ ∃ (x' y' : ℝ), x'^2 + 2*x'*y' - y'^2 = 7 ∧ x'^2 + y'^2 = m :=
sorry

end NUMINAMATH_CALUDE_min_value_x2_plus_y2_l2481_248138


namespace NUMINAMATH_CALUDE_taxi_ride_cost_l2481_248168

def base_fee : ℚ := 1.5
def cost_per_mile : ℚ := 0.25
def ride1_distance : ℕ := 5
def ride2_distance : ℕ := 8
def ride3_distance : ℕ := 3

theorem taxi_ride_cost : 
  (base_fee + cost_per_mile * ride1_distance) + 
  (base_fee + cost_per_mile * ride2_distance) + 
  (base_fee + cost_per_mile * ride3_distance) = 8.5 := by
sorry

end NUMINAMATH_CALUDE_taxi_ride_cost_l2481_248168


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2481_248194

theorem trigonometric_identities (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  (Real.cos α)^2 = 1/5 ∧
  (Real.sin α / (Real.sin α + Real.cos α) = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2481_248194


namespace NUMINAMATH_CALUDE_no_real_solutions_l2481_248143

theorem no_real_solutions (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≠ 0) :
  (∀ x : ℝ, a * x^2 + a * x + a ≠ b) ↔ (a = 0 ∨ (a ≠ 0 ∧ 4 * a * b - 3 * a^2 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2481_248143


namespace NUMINAMATH_CALUDE_teaching_competition_score_l2481_248151

theorem teaching_competition_score (teaching_design_weight : ℝ) 
                                   (on_site_demo_weight : ℝ) 
                                   (teaching_design_score : ℝ) 
                                   (on_site_demo_score : ℝ) 
                                   (h1 : teaching_design_weight = 0.2)
                                   (h2 : on_site_demo_weight = 0.8)
                                   (h3 : teaching_design_score = 90)
                                   (h4 : on_site_demo_score = 95) :
  teaching_design_weight * teaching_design_score + 
  on_site_demo_weight * on_site_demo_score = 94 := by
sorry

end NUMINAMATH_CALUDE_teaching_competition_score_l2481_248151


namespace NUMINAMATH_CALUDE_bernardo_win_smallest_number_l2481_248130

def game_winner (N : ℕ) : Prop :=
  N ≤ 999 ∧ 8 * N + 600 < 1000 ∧ 8 * N + 700 ≥ 1000

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem bernardo_win_smallest_number :
  ∃ (N : ℕ), N = 38 ∧ game_winner N ∧
  (∀ (M : ℕ), M < N → ¬game_winner M) ∧
  sum_of_digits N = 11 :=
sorry

end NUMINAMATH_CALUDE_bernardo_win_smallest_number_l2481_248130


namespace NUMINAMATH_CALUDE_polynomial_with_specific_roots_l2481_248181

theorem polynomial_with_specific_roots :
  ∃ (P : ℂ → ℂ) (r s : ℤ),
    (∀ x, P x = x^4 + (a : ℤ) * x^3 + (b : ℤ) * x^2 + (c : ℤ) * x + (d : ℤ)) ∧
    (P r = 0) ∧ (P s = 0) ∧ (P ((1 + Complex.I * Real.sqrt 15) / 2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_with_specific_roots_l2481_248181


namespace NUMINAMATH_CALUDE_course_ratio_l2481_248193

theorem course_ratio (max_courses sid_courses : ℕ) (m : ℚ) : 
  max_courses = 40 →
  max_courses + sid_courses = 200 →
  sid_courses = m * max_courses →
  m = 4 ∧ sid_courses / max_courses = 4 := by
  sorry

end NUMINAMATH_CALUDE_course_ratio_l2481_248193


namespace NUMINAMATH_CALUDE_bankers_discount_l2481_248153

/-- Banker's discount calculation -/
theorem bankers_discount (bankers_gain : ℚ) (time : ℕ) (rate : ℚ) : 
  bankers_gain = 360 ∧ time = 3 ∧ rate = 12/100 → 
  ∃ (bankers_discount : ℚ), bankers_discount = 5625/10 :=
by
  sorry

end NUMINAMATH_CALUDE_bankers_discount_l2481_248153


namespace NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l2481_248115

-- Problem 1
theorem calculation_proof : |(-2)| - 2 * Real.sin (30 * π / 180) + (2023 ^ 0) = 2 := by sorry

-- Problem 2
theorem inequality_system_solution :
  (∀ x : ℝ, (3 * x - 1 > -7 ∧ 2 * x < x + 2) ↔ (-2 < x ∧ x < 2)) := by sorry

end NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l2481_248115


namespace NUMINAMATH_CALUDE_consecutive_card_picks_standard_deck_l2481_248170

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (face_cards_per_suit : ℕ)
  (number_cards_per_suit : ℕ)

/-- Calculates the number of ways to pick two consecutive cards from the same suit,
    where one is a face card and the other is a number card -/
def consecutive_card_picks (d : Deck) : ℕ :=
  d.num_suits * (d.face_cards_per_suit * d.number_cards_per_suit * 2)

/-- Theorem stating that for a standard deck, there are 240 ways to pick two consecutive
    cards from the same suit, where one is a face card and the other is a number card -/
theorem consecutive_card_picks_standard_deck :
  let d : Deck := {
    total_cards := 48,
    num_suits := 4,
    cards_per_suit := 12,
    face_cards_per_suit := 3,
    number_cards_per_suit := 10
  }
  consecutive_card_picks d = 240 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_card_picks_standard_deck_l2481_248170


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_5_and_9_with_even_digits_l2481_248179

def is_even_digit (d : Nat) : Prop := d % 2 = 0 ∧ d < 10

def has_only_even_digits (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_even_digit d

def is_four_digit (n : Nat) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_by_5_and_9_with_even_digits :
  ∀ n : Nat,
    is_four_digit n ∧
    has_only_even_digits n ∧
    n % 5 = 0 ∧
    n % 9 = 0 →
    2880 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_5_and_9_with_even_digits_l2481_248179


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l2481_248149

-- Define the function f
def f (x : ℝ) : ℝ := x^4 - x^2 + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4*x^3 - 2*x

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 2*x - 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l2481_248149


namespace NUMINAMATH_CALUDE_polyhedron_edges_l2481_248173

theorem polyhedron_edges (F V E : ℕ) : F + V - E = 2 → F = 6 → V = 8 → E = 12 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_edges_l2481_248173


namespace NUMINAMATH_CALUDE_watch_cost_price_l2481_248189

/-- The cost price of a watch, given certain selling conditions. -/
def cost_price : ℝ := 1166.67

/-- The selling price at a loss. -/
def selling_price_loss : ℝ := 0.90 * cost_price

/-- The selling price at a gain. -/
def selling_price_gain : ℝ := 1.02 * cost_price

/-- Theorem stating the cost price of the watch given the selling conditions. -/
theorem watch_cost_price :
  (selling_price_loss = 0.90 * cost_price) ∧
  (selling_price_gain = 1.02 * cost_price) ∧
  (selling_price_gain - selling_price_loss = 140) →
  cost_price = 1166.67 :=
by sorry

end NUMINAMATH_CALUDE_watch_cost_price_l2481_248189


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2481_248119

theorem ellipse_eccentricity (a b c : ℝ) (θ : ℝ) : 
  a > b ∧ b > 0 ∧  -- conditions for ellipse
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → (x-c)^2 + y^2 ≤ 1) ∧  -- circle inside ellipse
  (π/3 ≤ θ ∧ θ ≤ π/2) →  -- angle condition
  c/a = 3 - 2 * Real.sqrt 2 :=  -- eccentricity
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2481_248119


namespace NUMINAMATH_CALUDE_angle_range_from_cosine_bounds_l2481_248187

theorem angle_range_from_cosine_bounds (A : Real) (h_acute : 0 < A ∧ A < Real.pi / 2) 
  (h_cos_bounds : 1 / 2 < Real.cos A ∧ Real.cos A < Real.sqrt 3 / 2) : 
  Real.pi / 6 < A ∧ A < Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_angle_range_from_cosine_bounds_l2481_248187


namespace NUMINAMATH_CALUDE_expression_equals_zero_l2481_248120

theorem expression_equals_zero : 2 * 2^5 - 8^58 / 8^56 = 0 := by sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l2481_248120


namespace NUMINAMATH_CALUDE_largest_divisible_n_l2481_248169

theorem largest_divisible_n : ∃ (n : ℕ), 
  (∀ (m : ℕ), m > n → ¬((m + 20) ∣ (m^3 - 100))) ∧ 
  ((n + 20) ∣ (n^3 - 100)) ∧ 
  n = 2080 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l2481_248169


namespace NUMINAMATH_CALUDE_coast_guard_overtakes_at_2_15pm_l2481_248118

/-- Represents the time of day in hours and minutes -/
structure TimeOfDay where
  hours : ℕ
  minutes : ℕ
  hValid : hours < 24 ∧ minutes < 60

/-- Represents the chase scenario -/
structure ChaseScenario where
  initialDistance : ℝ
  initialTime : TimeOfDay
  smugglerInitialSpeed : ℝ
  coastGuardSpeed : ℝ
  smugglerReducedSpeed : ℝ
  malfunctionTime : ℝ

/-- Calculates the time when the coast guard overtakes the smuggler -/
def overtakeTime (scenario : ChaseScenario) : TimeOfDay :=
  sorry

/-- The main theorem to prove -/
theorem coast_guard_overtakes_at_2_15pm
  (scenario : ChaseScenario)
  (h1 : scenario.initialDistance = 15)
  (h2 : scenario.initialTime = ⟨10, 0, sorry⟩)
  (h3 : scenario.smugglerInitialSpeed = 18)
  (h4 : scenario.coastGuardSpeed = 20)
  (h5 : scenario.smugglerReducedSpeed = 16)
  (h6 : scenario.malfunctionTime = 1) :
  overtakeTime scenario = ⟨14, 15, sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_coast_guard_overtakes_at_2_15pm_l2481_248118


namespace NUMINAMATH_CALUDE_chess_tournament_matches_chess_tournament_problem_l2481_248175

/-- Represents a single elimination chess tournament --/
structure ChessTournament where
  total_players : ℕ
  bye_players : ℕ
  matches_played : ℕ

/-- Theorem stating the number of matches in the given tournament --/
theorem chess_tournament_matches 
  (tournament : ChessTournament) 
  (h1 : tournament.total_players = 128) 
  (h2 : tournament.bye_players = 32) : 
  tournament.matches_played = 127 := by
  sorry

/-- Main theorem to be proved --/
theorem chess_tournament_problem : 
  ∃ (t : ChessTournament), t.total_players = 128 ∧ t.bye_players = 32 ∧ t.matches_played = 127 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_matches_chess_tournament_problem_l2481_248175


namespace NUMINAMATH_CALUDE_club_assignment_count_l2481_248133

/-- Represents a club -/
inductive Club
| LittleGrassLiteratureSociety
| StreetDanceClub
| FootballHouse
| CyclingClub

/-- Represents a student -/
inductive Student
| A
| B
| C
| D
| E

/-- A valid club assignment is a function from Student to Club -/
def ClubAssignment := Student → Club

/-- Predicate to check if a club assignment is valid -/
def isValidAssignment (assignment : ClubAssignment) : Prop :=
  (∀ c : Club, ∃ s : Student, assignment s = c) ∧
  (assignment Student.A ≠ Club.StreetDanceClub)

/-- The number of valid club assignments -/
def numValidAssignments : ℕ := sorry

theorem club_assignment_count :
  numValidAssignments = 180 := by sorry

end NUMINAMATH_CALUDE_club_assignment_count_l2481_248133


namespace NUMINAMATH_CALUDE_work_completion_time_l2481_248142

/-- Given that:
  - A can do a work in 9 days
  - A and B together can do the work in 6 days
  Prove that B can do the work alone in 18 days -/
theorem work_completion_time (a b : ℝ) (ha : a = 9) (hab : 1 / a + 1 / b = 1 / 6) : b = 18 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2481_248142


namespace NUMINAMATH_CALUDE_two_common_tangents_l2481_248185

/-- The number of common tangents between two circles -/
def num_common_tangents (C₁ C₂ : ℝ → ℝ → Prop) : ℕ :=
  sorry

/-- Circle C₁ equation -/
def C₁ (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 8*y - 8 = 0

/-- Circle C₂ equation -/
def C₂ (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 4*y - 1 = 0

/-- Theorem stating that there are 2 common tangents between C₁ and C₂ -/
theorem two_common_tangents : num_common_tangents C₁ C₂ = 2 :=
  sorry

end NUMINAMATH_CALUDE_two_common_tangents_l2481_248185


namespace NUMINAMATH_CALUDE_shopping_trip_cost_theorem_l2481_248101

def shopping_trip_cost (t_shirt_price : ℝ) (t_shirt_count : ℕ) 
                       (jeans_price : ℝ) (jeans_count : ℕ)
                       (socks_price : ℝ) (socks_count : ℕ)
                       (t_shirt_discount : ℝ) (jeans_discount : ℝ)
                       (sales_tax : ℝ) : ℝ :=
  let t_shirt_total := t_shirt_price * t_shirt_count
  let jeans_total := jeans_price * jeans_count
  let socks_total := socks_price * socks_count
  let t_shirt_discounted := t_shirt_total * (1 - t_shirt_discount)
  let jeans_discounted := jeans_total * (1 - jeans_discount)
  let subtotal := t_shirt_discounted + jeans_discounted + socks_total
  subtotal * (1 + sales_tax)

theorem shopping_trip_cost_theorem :
  shopping_trip_cost 9.65 12 29.95 3 4.50 5 0.15 0.10 0.08 = 217.93 := by
  sorry

end NUMINAMATH_CALUDE_shopping_trip_cost_theorem_l2481_248101


namespace NUMINAMATH_CALUDE_cube_sum_of_cyclic_matrix_cube_is_identity_l2481_248163

/-- N is a 3x3 matrix with real entries x, y, z -/
def N (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![x, y, z; y, z, x; z, x, y]

/-- The theorem statement -/
theorem cube_sum_of_cyclic_matrix_cube_is_identity
  (x y z : ℝ) (h1 : N x y z ^ 3 = 1) (h2 : x * y * z = -1) :
  x^3 + y^3 + z^3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_of_cyclic_matrix_cube_is_identity_l2481_248163


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2481_248148

/-- Given a quadratic equation x^2 - (m+1)x + 2m = 0 where 3 is a root,
    and an isosceles triangle ABC where two sides have lengths equal to the roots of the equation,
    prove that the perimeter of the triangle is either 10 or 11. -/
theorem isosceles_triangle_perimeter (m : ℝ) :
  (3^2 - (m+1)*3 + 2*m = 0) →
  ∃ (a b : ℝ), (a^2 - (m+1)*a + 2*m = 0) ∧ (b^2 - (m+1)*b + 2*m = 0) ∧ 
  ((a + a + b = 10) ∨ (a + a + b = 11) ∨ (b + b + a = 10) ∨ (b + b + a = 11)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2481_248148


namespace NUMINAMATH_CALUDE_four_numbers_sum_product_l2481_248184

def satisfies_condition (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  x₁ + x₂ * x₃ * x₄ = 2 ∧
  x₂ + x₁ * x₃ * x₄ = 2 ∧
  x₃ + x₁ * x₂ * x₄ = 2 ∧
  x₄ + x₁ * x₂ * x₃ = 2

def is_permutation (a b c d : ℝ) (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  (x₁ = a ∧ x₂ = b ∧ x₃ = c ∧ x₄ = d) ∨
  (x₁ = a ∧ x₂ = b ∧ x₃ = d ∧ x₄ = c) ∨
  (x₁ = a ∧ x₂ = c ∧ x₃ = b ∧ x₄ = d) ∨
  (x₁ = a ∧ x₂ = c ∧ x₃ = d ∧ x₄ = b) ∨
  (x₁ = a ∧ x₂ = d ∧ x₃ = b ∧ x₄ = c) ∨
  (x₁ = a ∧ x₂ = d ∧ x₃ = c ∧ x₄ = b) ∨
  (x₁ = b ∧ x₂ = a ∧ x₃ = c ∧ x₄ = d) ∨
  (x₁ = b ∧ x₂ = a ∧ x₃ = d ∧ x₄ = c) ∨
  (x₁ = b ∧ x₂ = c ∧ x₃ = a ∧ x₄ = d) ∨
  (x₁ = b ∧ x₂ = c ∧ x₃ = d ∧ x₄ = a) ∨
  (x₁ = b ∧ x₂ = d ∧ x₃ = a ∧ x₄ = c) ∨
  (x₁ = b ∧ x₂ = d ∧ x₃ = c ∧ x₄ = a) ∨
  (x₁ = c ∧ x₂ = a ∧ x₃ = b ∧ x₄ = d) ∨
  (x₁ = c ∧ x₂ = a ∧ x₃ = d ∧ x₄ = b) ∨
  (x₁ = c ∧ x₂ = b ∧ x₃ = a ∧ x₄ = d) ∨
  (x₁ = c ∧ x₂ = b ∧ x₃ = d ∧ x₄ = a) ∨
  (x₁ = c ∧ x₂ = d ∧ x₃ = a ∧ x₄ = b) ∨
  (x₁ = c ∧ x₂ = d ∧ x₃ = b ∧ x₄ = a) ∨
  (x₁ = d ∧ x₂ = a ∧ x₃ = b ∧ x₄ = c) ∨
  (x₁ = d ∧ x₂ = a ∧ x₃ = c ∧ x₄ = b) ∨
  (x₁ = d ∧ x₂ = b ∧ x₃ = a ∧ x₄ = c) ∨
  (x₁ = d ∧ x₂ = b ∧ x₃ = c ∧ x₄ = a) ∨
  (x₁ = d ∧ x₂ = c ∧ x₃ = a ∧ x₄ = b) ∨
  (x₁ = d ∧ x₂ = c ∧ x₃ = b ∧ x₄ = a)

theorem four_numbers_sum_product (x₁ x₂ x₃ x₄ : ℝ) :
  satisfies_condition x₁ x₂ x₃ x₄ ↔
  (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1 ∧ x₄ = 1) ∨
  is_permutation 3 (-1) (-1) (-1) x₁ x₂ x₃ x₄ :=
sorry

end NUMINAMATH_CALUDE_four_numbers_sum_product_l2481_248184


namespace NUMINAMATH_CALUDE_equation_subtraction_result_l2481_248100

theorem equation_subtraction_result :
  let eq1 : ℝ → ℝ → ℝ := fun x y => 2*x + 5*y
  let eq2 : ℝ → ℝ → ℝ := fun x y => 2*x - 3*y
  let result : ℝ → ℝ := fun y => 8*y
  ∀ x y : ℝ, eq1 x y = 9 ∧ eq2 x y = 6 →
    result y = 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_subtraction_result_l2481_248100


namespace NUMINAMATH_CALUDE_balls_in_boxes_l2481_248146

theorem balls_in_boxes (n : ℕ) (k : ℕ) : n = 5 ∧ k = 4 → k^n = 1024 := by
  sorry

end NUMINAMATH_CALUDE_balls_in_boxes_l2481_248146


namespace NUMINAMATH_CALUDE_function_properties_l2481_248199

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x + (1/2) * m * x^2 - 2

theorem function_properties (m : ℝ) :
  (((Real.exp 1)⁻¹ + m = -(1/2)) → m = -(3/2)) ∧
  (∀ x > 0, f m x + 2 ≤ m * x^2 + (m - 1) * x - 1) →
  m ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_function_properties_l2481_248199


namespace NUMINAMATH_CALUDE_percentage_of_returns_l2481_248160

/-- Calculate the percentage of customers returning books -/
theorem percentage_of_returns (total_customers : ℕ) (price_per_book : ℚ) (sales_after_returns : ℚ) :
  total_customers = 1000 →
  price_per_book = 15 →
  sales_after_returns = 9450 →
  (((total_customers : ℚ) * price_per_book - sales_after_returns) / price_per_book) / (total_customers : ℚ) * 100 = 37 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_returns_l2481_248160


namespace NUMINAMATH_CALUDE_ratio_M_N_l2481_248134

theorem ratio_M_N (P Q R M N : ℝ) 
  (hM : M = 0.4 * Q)
  (hQ : Q = 0.25 * P)
  (hN : N = 0.75 * R)
  (hR : R = 0.6 * P)
  (hP : P ≠ 0) : 
  M / N = 2 / 9 := by
sorry

end NUMINAMATH_CALUDE_ratio_M_N_l2481_248134


namespace NUMINAMATH_CALUDE_problem_one_l2481_248156

theorem problem_one : 
  64.83 - 5 * (18/19 : ℚ) + 35.17 - 44 * (1/19 : ℚ) = 50 := by sorry

end NUMINAMATH_CALUDE_problem_one_l2481_248156


namespace NUMINAMATH_CALUDE_train_interval_l2481_248136

-- Define the train route times
def northern_route_time : ℝ := 17
def southern_route_time : ℝ := 11

-- Define the average time difference between counterclockwise and clockwise trains
def train_arrival_difference : ℝ := 1.25

-- Define the commute time difference
def commute_time_difference : ℝ := 1

-- Theorem statement
theorem train_interval (p : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 1) 
  (hcommute : southern_route_time * p + northern_route_time * (1 - p) + 1 = 
              northern_route_time * p + southern_route_time * (1 - p))
  (htrain_diff : (1 - p) * 3 = train_arrival_difference) : 
  3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_interval_l2481_248136


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2481_248125

/-- Tetrahedron ABCD with given properties -/
structure Tetrahedron where
  /-- Length of edge AB in cm -/
  ab_length : ℝ
  /-- Area of face ABC in cm² -/
  abc_area : ℝ
  /-- Area of face ABD in cm² -/
  abd_area : ℝ
  /-- Angle between faces ABC and ABD in radians -/
  face_angle : ℝ

/-- Volume of the tetrahedron in cm³ -/
def tetrahedron_volume (t : Tetrahedron) : ℝ := sorry

/-- Theorem stating the volume of the specific tetrahedron -/
theorem specific_tetrahedron_volume :
  ∃ t : Tetrahedron,
    t.ab_length = 5 ∧
    t.abc_area = 20 ∧
    t.abd_area = 18 ∧
    t.face_angle = π / 4 ∧
    tetrahedron_volume t = 24 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2481_248125


namespace NUMINAMATH_CALUDE_square_1011_position_l2481_248167

-- Define the possible positions of the square
inductive SquarePosition
| ABCD
| BCDA
| DCBA

-- Define the transformation function
def transform (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.BCDA
  | SquarePosition.BCDA => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.ABCD

-- Define the function to get the nth position
def nthPosition (n : Nat) : SquarePosition :=
  match n % 4 with
  | 0 => SquarePosition.ABCD
  | 1 => SquarePosition.BCDA
  | 2 => SquarePosition.DCBA
  | _ => SquarePosition.ABCD

-- Theorem stating that the 1011th square is in position DCBA
theorem square_1011_position :
  nthPosition 1011 = SquarePosition.DCBA := by
  sorry

end NUMINAMATH_CALUDE_square_1011_position_l2481_248167


namespace NUMINAMATH_CALUDE_inequality_solution_l2481_248102

theorem inequality_solution (x : ℝ) : (x^2 - 9) / (x^2 - 1) > 0 ↔ x > 3 ∨ x < -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2481_248102


namespace NUMINAMATH_CALUDE_intersection_with_complement_l2481_248177

def U : Finset Nat := {0, 1, 2, 3, 4}
def A : Finset Nat := {0, 1, 2, 3}
def B : Finset Nat := {0, 2, 4}

theorem intersection_with_complement :
  A ∩ (U \ B) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l2481_248177


namespace NUMINAMATH_CALUDE_log_sum_equality_l2481_248165

-- Define the theorem
theorem log_sum_equality (p q : ℝ) (h : q ≠ 1) :
  Real.log p + Real.log q = Real.log (p + 2*q) → p = 2*q / (q - 1) := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l2481_248165


namespace NUMINAMATH_CALUDE_additional_trays_is_ten_l2481_248154

/-- Represents the number of eggs in a tray -/
def eggs_per_tray : ℕ := 30

/-- Represents the initial number of trays to be collected -/
def initial_trays : ℕ := 10

/-- Represents the number of trays dropped -/
def dropped_trays : ℕ := 2

/-- Represents the total number of eggs sold -/
def total_eggs_sold : ℕ := 540

/-- Calculates the number of additional trays needed -/
def additional_trays : ℕ :=
  (total_eggs_sold - (initial_trays - dropped_trays) * eggs_per_tray) / eggs_per_tray

/-- Theorem stating that the number of additional trays is 10 -/
theorem additional_trays_is_ten : additional_trays = 10 := by
  sorry

end NUMINAMATH_CALUDE_additional_trays_is_ten_l2481_248154


namespace NUMINAMATH_CALUDE_glenn_spends_35_dollars_l2481_248108

/-- The cost of a movie ticket on Monday -/
def monday_price : ℕ := 5

/-- The cost of a movie ticket on Wednesday -/
def wednesday_price : ℕ := 2 * monday_price

/-- The cost of a movie ticket on Saturday -/
def saturday_price : ℕ := 5 * monday_price

/-- The total amount Glenn spends on movie tickets -/
def glenn_total_spent : ℕ := wednesday_price + saturday_price

/-- Theorem stating that Glenn spends $35 on movie tickets -/
theorem glenn_spends_35_dollars : glenn_total_spent = 35 := by
  sorry

end NUMINAMATH_CALUDE_glenn_spends_35_dollars_l2481_248108


namespace NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l2481_248155

theorem empty_solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 2| + |x + 3| < a)) → a ∈ Set.Iic 5 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_implies_a_range_l2481_248155


namespace NUMINAMATH_CALUDE_f_inequality_l2481_248162

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x ≠ 1, (x - 1) * (deriv f x) < 0)
variable (h2 : ∀ x, f (x + 1) = f (-x + 1))

-- Define the theorem
theorem f_inequality (x₁ x₂ : ℝ) (h3 : |x₁ - 1| < |x₂ - 1|) : f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l2481_248162


namespace NUMINAMATH_CALUDE_prob_one_from_each_jurisdiction_prob_at_least_one_from_xiaogan_l2481_248145

/-- Represents the total number of intermediate stations -/
def total_stations : ℕ := 7

/-- Represents the number of stations in Wuhan's jurisdiction -/
def wuhan_stations : ℕ := 4

/-- Represents the number of stations in Xiaogan's jurisdiction -/
def xiaogan_stations : ℕ := 3

/-- Represents the number of stations to be selected for research -/
def selected_stations : ℕ := 2

/-- Theorem for the probability of selecting one station from each jurisdiction -/
theorem prob_one_from_each_jurisdiction :
  (total_stations.choose selected_stations : ℚ) / (total_stations.choose selected_stations) =
  (wuhan_stations * xiaogan_stations : ℚ) / (total_stations.choose selected_stations) := by sorry

/-- Theorem for the probability of selecting at least one station within Xiaogan's jurisdiction -/
theorem prob_at_least_one_from_xiaogan :
  1 - (wuhan_stations.choose selected_stations : ℚ) / (total_stations.choose selected_stations) =
  5 / 7 := by sorry

end NUMINAMATH_CALUDE_prob_one_from_each_jurisdiction_prob_at_least_one_from_xiaogan_l2481_248145


namespace NUMINAMATH_CALUDE_total_textbook_cost_l2481_248192

/-- The total cost of textbooks given specific pricing conditions -/
theorem total_textbook_cost : 
  ∀ (sale_price : ℕ) (online_total : ℕ) (sale_count online_count bookstore_count : ℕ),
    sale_price = 10 →
    online_total = 40 →
    sale_count = 5 →
    online_count = 2 →
    bookstore_count = 3 →
    sale_count * sale_price + online_total + bookstore_count * online_total = 210 :=
by sorry

end NUMINAMATH_CALUDE_total_textbook_cost_l2481_248192


namespace NUMINAMATH_CALUDE_hyperbola_slope_product_l2481_248183

/-- The product of slopes for a hyperbola -/
theorem hyperbola_slope_product (a b c : ℝ) (P Q : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  b^2 = a * c →
  ((P.1^2 / a^2) - (P.2^2 / b^2) = 1) →
  ((Q.1^2 / a^2) - (Q.2^2 / b^2) = 1) →
  let M := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  let k_PQ := (Q.2 - P.2) / (Q.1 - P.1)
  let k_OM := M.2 / M.1
  k_PQ ≠ 0 →
  M.1 ≠ 0 →
  k_PQ * k_OM = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_slope_product_l2481_248183


namespace NUMINAMATH_CALUDE_ant_expected_moves_l2481_248144

/-- Represents the possible parities of the ant's position -/
inductive Parity
  | Even
  | Odd

/-- Defines the ant's position on the coordinate plane -/
structure AntPosition :=
  (x : Parity)
  (y : Parity)

/-- Calculates the expected number of moves to reach an anthill from a given position -/
noncomputable def expectedMoves (pos : AntPosition) : ℝ :=
  match pos with
  | ⟨Parity.Even, Parity.Even⟩ => 4
  | ⟨Parity.Odd, Parity.Odd⟩ => 0
  | _ => 3

/-- The main theorem to be proved -/
theorem ant_expected_moves :
  let initialPos : AntPosition := ⟨Parity.Even, Parity.Even⟩
  expectedMoves initialPos = 4 := by sorry

end NUMINAMATH_CALUDE_ant_expected_moves_l2481_248144


namespace NUMINAMATH_CALUDE_cassie_parrot_count_l2481_248172

/-- Represents the number of nails Cassie needs to cut for her pets -/
def total_nails : ℕ := 113

/-- Represents the number of dogs Cassie has -/
def num_dogs : ℕ := 4

/-- Represents the number of nails each dog has -/
def nails_per_dog : ℕ := 16

/-- Represents the number of claws each regular parrot has -/
def claws_per_parrot : ℕ := 6

/-- Represents the number of claws the special parrot with an extra toe has -/
def claws_special_parrot : ℕ := 7

/-- Theorem stating that the number of parrots Cassie has is 8 -/
theorem cassie_parrot_count : 
  ∃ (num_parrots : ℕ), 
    num_parrots * claws_per_parrot + 
    (claws_special_parrot - claws_per_parrot) + 
    (num_dogs * nails_per_dog) = total_nails ∧ 
    num_parrots = 8 := by
  sorry

end NUMINAMATH_CALUDE_cassie_parrot_count_l2481_248172


namespace NUMINAMATH_CALUDE_sequence_properties_l2481_248198

def sequence_a (n : ℕ) : ℚ := sorry

def sequence_S (n : ℕ) : ℚ := sorry

axiom a_1 : sequence_a 1 = 3

axiom S_def : ∀ n : ℕ, n ≥ 2 → 2 * sequence_a n = sequence_S n * sequence_S (n - 1)

theorem sequence_properties :
  (∃ d : ℚ, ∀ n : ℕ, n ≥ 1 → (1 / sequence_S (n + 1) - 1 / sequence_S n = d)) ∧
  (∀ n : ℕ, n ≥ 2 → sequence_a n = 18 / ((5 - 3 * n) * (8 - 3 * n))) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l2481_248198


namespace NUMINAMATH_CALUDE_inverse_function_point_l2481_248140

-- Define a monotonic function f
variable (f : ℝ → ℝ)
variable (h_mono : Monotone f)

-- Define the condition that f(x+1) passes through (-2, 1)
variable (h_point : f (-1) = 1)

-- State the theorem
theorem inverse_function_point :
  (Function.invFun f) 3 = -1 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_point_l2481_248140


namespace NUMINAMATH_CALUDE_max_enclosed_area_l2481_248191

/-- Represents the side length of the square garden -/
def s : ℕ := 49

/-- Represents the non-shared side length of the rectangular garden -/
def x : ℕ := 2

/-- The total perimeter of both gardens combined -/
def total_perimeter : ℕ := 200

/-- The maximum area that can be enclosed -/
def max_area : ℕ := 2499

/-- Theorem stating the maximum area that can be enclosed given the constraints -/
theorem max_enclosed_area :
  (4 * s + 2 * x = total_perimeter) → 
  (∀ s' x' : ℕ, (4 * s' + 2 * x' = total_perimeter) → (s' * s' + s' * x' ≤ max_area)) →
  (s * s + s * x = max_area) := by
  sorry

end NUMINAMATH_CALUDE_max_enclosed_area_l2481_248191


namespace NUMINAMATH_CALUDE_lenny_video_game_spending_l2481_248127

def video_game_expenditure (initial_amount grocery_spending remaining_amount : ℕ) : ℕ :=
  initial_amount - grocery_spending - remaining_amount

theorem lenny_video_game_spending :
  video_game_expenditure 84 21 39 = 24 :=
by sorry

end NUMINAMATH_CALUDE_lenny_video_game_spending_l2481_248127


namespace NUMINAMATH_CALUDE_forum_posts_l2481_248180

/-- Calculates the total number of questions and answers posted on a forum in a day -/
def total_posts_per_day (members : ℕ) (questions_per_hour : ℕ) (answer_ratio : ℕ) : ℕ :=
  let questions_per_day := questions_per_hour * 24
  let total_questions := members * questions_per_day
  let total_answers := members * questions_per_day * answer_ratio
  total_questions + total_answers

/-- Theorem stating the total number of posts on the forum in a day -/
theorem forum_posts :
  total_posts_per_day 500 5 4 = 300000 := by
  sorry

end NUMINAMATH_CALUDE_forum_posts_l2481_248180
