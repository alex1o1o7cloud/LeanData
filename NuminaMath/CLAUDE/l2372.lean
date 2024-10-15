import Mathlib

namespace NUMINAMATH_CALUDE_sector_radius_and_angle_l2372_237292

/-- Given a sector with perimeter 4 and area 1, prove its radius is 1 and central angle is 2 -/
theorem sector_radius_and_angle (r θ : ℝ) 
  (h_perimeter : 2 * r + θ * r = 4)
  (h_area : 1/2 * θ * r^2 = 1) : 
  r = 1 ∧ θ = 2 := by sorry

end NUMINAMATH_CALUDE_sector_radius_and_angle_l2372_237292


namespace NUMINAMATH_CALUDE_toothpicks_in_specific_grid_l2372_237205

/-- Calculates the number of toothpicks in a modified grid -/
def toothpicks_in_modified_grid (length width corner_size : ℕ) : ℕ :=
  let vertical_lines := length + 1
  let horizontal_lines := width + 1
  let corner_lines := corner_size + 1
  let total_without_corner := vertical_lines * width + horizontal_lines * length
  let corner_toothpicks := corner_lines * corner_size * 2
  total_without_corner - corner_toothpicks

/-- Theorem stating the number of toothpicks in the specific grid described in the problem -/
theorem toothpicks_in_specific_grid :
  toothpicks_in_modified_grid 70 45 5 = 6295 :=
by sorry

end NUMINAMATH_CALUDE_toothpicks_in_specific_grid_l2372_237205


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_1729_l2372_237296

theorem smallest_prime_factor_of_1729 :
  (Nat.minFac 1729 = 7) := by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_1729_l2372_237296


namespace NUMINAMATH_CALUDE_statue_carving_l2372_237269

theorem statue_carving (initial_weight : ℝ) (first_week_cut : ℝ) (second_week_cut : ℝ) (final_weight : ℝ) :
  initial_weight = 250 →
  first_week_cut = 0.3 →
  second_week_cut = 0.2 →
  final_weight = 105 →
  let weight_after_first_week := initial_weight * (1 - first_week_cut)
  let weight_after_second_week := weight_after_first_week * (1 - second_week_cut)
  let third_week_cut := (weight_after_second_week - final_weight) / weight_after_second_week
  third_week_cut = 0.25 := by
sorry

end NUMINAMATH_CALUDE_statue_carving_l2372_237269


namespace NUMINAMATH_CALUDE_replacement_sugar_percentage_l2372_237203

/-- Represents a sugar solution with a given weight and sugar percentage -/
structure SugarSolution where
  weight : ℝ
  sugarPercentage : ℝ

/-- Calculates the amount of sugar in a solution -/
def sugarAmount (solution : SugarSolution) : ℝ :=
  solution.weight * solution.sugarPercentage

theorem replacement_sugar_percentage
  (original : SugarSolution)
  (replacement : SugarSolution)
  (final : SugarSolution)
  (h1 : original.sugarPercentage = 0.10)
  (h2 : final.sugarPercentage = 0.14)
  (h3 : final.weight = original.weight)
  (h4 : replacement.weight = original.weight / 4)
  (h5 : sugarAmount final = sugarAmount original - sugarAmount original / 4 + sugarAmount replacement) :
  replacement.sugarPercentage = 0.26 := by
sorry

end NUMINAMATH_CALUDE_replacement_sugar_percentage_l2372_237203


namespace NUMINAMATH_CALUDE_binaryOp_solution_l2372_237290

/-- A binary operation on positive real numbers -/
def binaryOp : (ℝ → ℝ → ℝ) := sorry

/-- The binary operation is continuous -/
axiom binaryOp_continuous : Continuous (Function.uncurry binaryOp)

/-- The binary operation is commutative -/
axiom binaryOp_comm : ∀ a b : ℝ, a > 0 → b > 0 → binaryOp a b = binaryOp b a

/-- The binary operation is distributive across multiplication -/
axiom binaryOp_distrib : ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 
  binaryOp a (b * c) = (binaryOp a b) * (binaryOp a c)

/-- The binary operation satisfies 2 ⊗ 2 = 4 -/
axiom binaryOp_two_two : binaryOp 2 2 = 4

/-- The main theorem: if x ⊗ y = x for x > 1, then y = √2 -/
theorem binaryOp_solution {x y : ℝ} (hx : x > 1) (h : binaryOp x y = x) : 
  y = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_binaryOp_solution_l2372_237290


namespace NUMINAMATH_CALUDE_billy_age_l2372_237256

theorem billy_age (billy joe : ℕ) 
  (h1 : billy = 3 * joe) 
  (h2 : billy + joe = 60) : 
  billy = 45 := by
sorry

end NUMINAMATH_CALUDE_billy_age_l2372_237256


namespace NUMINAMATH_CALUDE_sequence_general_term_l2372_237243

/-- Given a sequence {a_n} where the sum of the first n terms S_n satisfies S_n = (3/2)a_n - 3,
    prove that the general term formula is a_n = 2 * 3^n. -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (3/2) * a n - 3) →
  ∃ C, ∀ n, a n = C * 3^n :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2372_237243


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2372_237288

-- Define a geometric sequence
def isGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) (h : isGeometric a) :
  a 4 * a 6 * a 8 * a 10 * a 12 = 32 → a 10^2 / a 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2372_237288


namespace NUMINAMATH_CALUDE_product_72516_9999_l2372_237239

theorem product_72516_9999 : 72516 * 9999 = 724987484 := by
  sorry

end NUMINAMATH_CALUDE_product_72516_9999_l2372_237239


namespace NUMINAMATH_CALUDE_integer_solutions_of_equation_l2372_237201

theorem integer_solutions_of_equation : 
  ∀ x y : ℤ, x^2 - x*y - 6*y^2 + 2*x + 19*y = 18 ↔ (x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2) := by
sorry

end NUMINAMATH_CALUDE_integer_solutions_of_equation_l2372_237201


namespace NUMINAMATH_CALUDE_mersenne_fermat_prime_composite_l2372_237252

theorem mersenne_fermat_prime_composite (n : ℕ) (h : n > 2) :
  (Nat.Prime (2^n - 1) → ¬Nat.Prime (2^n + 1)) ∧
  (Nat.Prime (2^n + 1) → ¬Nat.Prime (2^n - 1)) :=
sorry

end NUMINAMATH_CALUDE_mersenne_fermat_prime_composite_l2372_237252


namespace NUMINAMATH_CALUDE_number_difference_l2372_237289

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 23540)
  (b_div_16 : b % 16 = 0)
  (b_eq_100a : b = 100 * a) : 
  b - a = 23067 := by sorry

end NUMINAMATH_CALUDE_number_difference_l2372_237289


namespace NUMINAMATH_CALUDE_triangle_to_pentagon_area_ratio_l2372_237270

/-- The ratio of the area of an equilateral triangle to the area of a pentagon formed by
    placing the triangle atop a square (where the triangle's base equals the square's side) -/
theorem triangle_to_pentagon_area_ratio :
  let s : ℝ := 1  -- Assume unit length for simplicity
  let square_area := s^2
  let triangle_area := (s^2 * Real.sqrt 3) / 4
  let pentagon_area := square_area + triangle_area
  triangle_area / pentagon_area = (4 * Real.sqrt 3 - 3) / 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_to_pentagon_area_ratio_l2372_237270


namespace NUMINAMATH_CALUDE_faculty_reduction_proof_l2372_237251

/-- The original number of faculty members before reduction -/
def original_faculty : ℕ := 253

/-- The percentage of faculty remaining after reduction -/
def remaining_percentage : ℚ := 77 / 100

/-- The number of faculty members after reduction -/
def reduced_faculty : ℕ := 195

/-- Theorem stating that the original faculty count, when reduced by 23%, 
    results in approximately 195 professors -/
theorem faculty_reduction_proof : 
  ⌊(original_faculty : ℚ) * remaining_percentage⌋ = reduced_faculty :=
sorry

end NUMINAMATH_CALUDE_faculty_reduction_proof_l2372_237251


namespace NUMINAMATH_CALUDE_ratio_to_thirteen_l2372_237299

theorem ratio_to_thirteen : ∃ x : ℚ, (5 : ℚ) / 1 = x / 13 ∧ x = 65 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_thirteen_l2372_237299


namespace NUMINAMATH_CALUDE_circle_C_properties_l2372_237255

-- Define the circle C
def circle_C (x y : ℝ) := (x - 3)^2 + (y - 1)^2 = 1

-- Define the line l
def line_l (x y m : ℝ) := x + 2*y + m = 0

theorem circle_C_properties :
  -- Circle C passes through (2,1) and (3,2)
  circle_C 2 1 ∧ circle_C 3 2 ∧
  -- Circle C is symmetric with respect to x-3y=0
  (∀ x y, circle_C x y → circle_C (3*y) y) →
  -- The standard equation of C is (x-3)^2 + (y-1)^2 = 1
  (∀ x y, circle_C x y ↔ (x - 3)^2 + (y - 1)^2 = 1) ∧
  -- If C intersects line_l at A and B with |AB| = 4√5/5, then m = -4 or m = -6
  (∀ m : ℝ, (∃ A B : ℝ × ℝ, 
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    line_l A.1 A.2 m ∧ line_l B.1 B.2 m ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (4*Real.sqrt 5/5)^2) →
    m = -4 ∨ m = -6) :=
sorry

end NUMINAMATH_CALUDE_circle_C_properties_l2372_237255


namespace NUMINAMATH_CALUDE_all_options_satisfy_statement_l2372_237240

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem all_options_satisfy_statement : 
  ∀ n ∈ ({54, 81, 99, 108} : Set ℕ), 
    (sum_of_digits n) % 9 = 0 → n % 9 = 0 ∧ n % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_all_options_satisfy_statement_l2372_237240


namespace NUMINAMATH_CALUDE_expression_evaluation_l2372_237277

theorem expression_evaluation (x y : ℝ) (hx : x = 2) (hy : y = -1) :
  (x - 2*y)^2 - (x - 3*y)*(x + 3*y) - 4*y^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2372_237277


namespace NUMINAMATH_CALUDE_towel_loads_l2372_237272

theorem towel_loads (towels_per_load : ℕ) (total_towels : ℕ) (h1 : towels_per_load = 7) (h2 : total_towels = 42) :
  total_towels / towels_per_load = 6 := by
  sorry

end NUMINAMATH_CALUDE_towel_loads_l2372_237272


namespace NUMINAMATH_CALUDE_constant_dot_product_l2372_237229

/-- The ellipse E -/
def ellipse_E (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- The hyperbola C -/
def hyperbola_C (x y : ℝ) : Prop := x^2 - y^2/2 = 1

/-- The foci of ellipse E and hyperbola C coincide -/
axiom foci_coincide : ∀ x y : ℝ, ellipse_E x y → hyperbola_C x y → x^2 - y^2 = 3

/-- The minor axis endpoints and one focus of ellipse E form an equilateral triangle -/
axiom equilateral_triangle : ∀ x y : ℝ, ellipse_E x y → x^2 + y^2 = 1 → x^2 = 3/4

/-- The dot product MP · MQ is constant when m = 17/8 -/
theorem constant_dot_product :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
  ellipse_E x₁ y₁ → ellipse_E x₂ y₂ →
  ∃ k : ℝ, y₁ = k * (x₁ - 1) ∧ y₂ = k * (x₂ - 1) →
  (17/8 - x₁) * (17/8 - x₂) + y₁ * y₂ = 33/64 :=
sorry

end NUMINAMATH_CALUDE_constant_dot_product_l2372_237229


namespace NUMINAMATH_CALUDE_blackboard_numbers_l2372_237261

def can_be_written (n : ℕ) : Prop :=
  ∃ x y : ℕ, n + 1 = 2^x * 3^y

theorem blackboard_numbers (n : ℕ) :
  can_be_written n ↔ 
  (n = 13121 ∨ (∃ a b : ℕ, can_be_written a ∧ can_be_written b ∧ n = a * b + a + b)) :=
sorry

end NUMINAMATH_CALUDE_blackboard_numbers_l2372_237261


namespace NUMINAMATH_CALUDE_counterexample_exists_l2372_237233

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2372_237233


namespace NUMINAMATH_CALUDE_equation_holds_l2372_237213

theorem equation_holds (a b : ℝ) : a^2 - b^2 - (-2*b^2) = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l2372_237213


namespace NUMINAMATH_CALUDE_paint_mixture_intensity_l2372_237242

/-- Calculates the intensity of a paint mixture -/
def mixturePaintIntensity (originalIntensity : ℚ) (addedIntensity : ℚ) (replacedFraction : ℚ) : ℚ :=
  (1 - replacedFraction) * originalIntensity + replacedFraction * addedIntensity

/-- Theorem stating that mixing 50% intensity paint with 20% intensity paint in a 2:1 ratio results in 40% intensity -/
theorem paint_mixture_intensity :
  mixturePaintIntensity (1/2) (1/5) (1/3) = (2/5) := by
  sorry

#eval mixturePaintIntensity (1/2) (1/5) (1/3)

end NUMINAMATH_CALUDE_paint_mixture_intensity_l2372_237242


namespace NUMINAMATH_CALUDE_number_of_children_l2372_237294

theorem number_of_children (crayons_per_child : ℕ) (total_crayons : ℕ) 
  (h1 : crayons_per_child = 12) 
  (h2 : total_crayons = 216) : 
  total_crayons / crayons_per_child = 18 := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l2372_237294


namespace NUMINAMATH_CALUDE_perpendicular_planes_condition_l2372_237210

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and between lines
variable (perp_planes : Plane → Plane → Prop)
variable (perp_lines : Line → Line → Prop)

-- Define the relation for a line being in a plane
variable (in_plane : Line → Plane → Prop)

-- Define the relation for two lines being different
variable (different : Line → Line → Prop)

-- Define the relation for two lines intersecting
variable (intersect : Line → Line → Prop)

-- State the theorem
theorem perpendicular_planes_condition 
  (α β : Plane) (m n l₁ l₂ : Line)
  (h1 : in_plane m α)
  (h2 : in_plane n α)
  (h3 : different m n)
  (h4 : in_plane l₁ β)
  (h5 : in_plane l₂ β)
  (h6 : intersect l₁ l₂)
  (h7 : perp_lines m l₁)
  (h8 : perp_lines m l₂) :
  perp_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_condition_l2372_237210


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_squared_l2372_237264

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The sum of the first five terms of the sequence equals 27. -/
def SumFirstFiveIs27 (a : ℕ → ℝ) : Prop :=
  a 1 + a 2 + a 3 + a 4 + a 5 = 27

/-- The sum of the reciprocals of the first five terms of the sequence equals 3. -/
def SumReciprocalFirstFiveIs3 (a : ℕ → ℝ) : Prop :=
  1 / a 1 + 1 / a 2 + 1 / a 3 + 1 / a 4 + 1 / a 5 = 3

theorem geometric_sequence_third_term_squared
  (a : ℕ → ℝ)
  (h_geometric : IsGeometricSequence a)
  (h_sum : SumFirstFiveIs27 a)
  (h_sum_reciprocal : SumReciprocalFirstFiveIs3 a) :
  (a 3) ^ 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_squared_l2372_237264


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2372_237284

/-- Given a quadratic function f(x) = ax² + bx + 1 with two distinct points
    (m, 2023) and (n, 2023) on its graph, prove that f(m + n) = 1 -/
theorem quadratic_function_property
  (a b m n : ℝ)
  (hm : a * m^2 + b * m + 1 = 2023)
  (hn : a * n^2 + b * n + 1 = 2023)
  (hd : m ≠ n) :
  a * (m + n)^2 + b * (m + n) + 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2372_237284


namespace NUMINAMATH_CALUDE_mikes_pumpkins_l2372_237276

theorem mikes_pumpkins (sandy_pumpkins : ℕ) (total_pumpkins : ℕ) (mike_pumpkins : ℕ) : 
  sandy_pumpkins = 51 → total_pumpkins = 74 → mike_pumpkins = total_pumpkins - sandy_pumpkins → mike_pumpkins = 23 := by
  sorry

end NUMINAMATH_CALUDE_mikes_pumpkins_l2372_237276


namespace NUMINAMATH_CALUDE_square_difference_l2372_237245

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 12) : 
  (x - y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2372_237245


namespace NUMINAMATH_CALUDE_stocking_stuffers_l2372_237259

/-- Calculates the total cost of stocking stuffers for all kids and the number of unique combinations of books and toys for each kid's stocking. -/
theorem stocking_stuffers (num_kids : ℕ) (num_candy_canes : ℕ) (candy_cane_price : ℚ)
  (num_beanie_babies : ℕ) (beanie_baby_price : ℚ) (num_books : ℕ) (book_price : ℚ)
  (num_toys_per_stocking : ℕ) (num_toy_options : ℕ) (toy_price : ℚ) (gift_card_value : ℚ) :
  num_kids = 4 →
  num_candy_canes = 4 →
  candy_cane_price = 1/2 →
  num_beanie_babies = 2 →
  beanie_baby_price = 3 →
  num_books = 5 →
  book_price = 5 →
  num_toys_per_stocking = 3 →
  num_toy_options = 10 →
  toy_price = 1 →
  gift_card_value = 10 →
  (num_kids * (num_candy_canes * candy_cane_price +
               num_beanie_babies * beanie_baby_price +
               book_price +
               num_toys_per_stocking * toy_price +
               gift_card_value) = 104) ∧
  (num_books * (num_toy_options.choose num_toys_per_stocking) = 600) :=
by sorry

end NUMINAMATH_CALUDE_stocking_stuffers_l2372_237259


namespace NUMINAMATH_CALUDE_smallest_x_quadratic_l2372_237221

theorem smallest_x_quadratic : 
  let f : ℝ → ℝ := λ x => 5 * x^2 + 8 * x + 3
  ∃ x : ℝ, f x = 9 ∧ ∀ y : ℝ, f y = 9 → x ≤ y ∧ x = (-8 - 2 * Real.sqrt 46) / 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_quadratic_l2372_237221


namespace NUMINAMATH_CALUDE_quadratic_value_range_l2372_237275

-- Define the set of x that satisfies the inequality
def S : Set ℝ := {x : ℝ | x^2 - 7*x + 12 < 0}

-- State the theorem
theorem quadratic_value_range : 
  ∀ x ∈ S, 0 < x^2 - 5*x + 6 ∧ x^2 - 5*x + 6 < 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_value_range_l2372_237275


namespace NUMINAMATH_CALUDE_min_value_expression_l2372_237232

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 3) :
  (2 * a^2 + 1) / a + (b^2 - 2) / (b + 2) ≥ 13 / 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2372_237232


namespace NUMINAMATH_CALUDE_right_triangle_rotation_volume_l2372_237295

/-- Given a right-angled triangle with legs b and c, and hypotenuse a, 
    where b + c = 25 and angle α = 61°55'40", 
    the volume of the solid formed by rotating the triangle around its hypotenuse 
    is approximately 887. -/
theorem right_triangle_rotation_volume 
  (b c a : ℝ) (α : Real) 
  (h_right_angle : b^2 + c^2 = a^2)
  (h_sum : b + c = 25)
  (h_angle : α = Real.pi * (61 + 55/60 + 40/3600) / 180) :
  ∃ (V : ℝ), abs (V - 887) < 1 ∧ V = (1/3) * Real.pi * c * (a * b / Real.sqrt (a^2 + b^2))^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_rotation_volume_l2372_237295


namespace NUMINAMATH_CALUDE_f_properties_l2372_237297

open Real

noncomputable def f (x : ℝ) := exp x - (1/2) * x^2

theorem f_properties :
  (∃ (m b : ℝ), m = 1 ∧ b = -1 ∧ ∀ x y, y = f x → m * x + b * y + 1 = 0) ∧
  (3/2 < f (log 2) ∧ f (log 2) < 2) ∧
  (∃! x, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2372_237297


namespace NUMINAMATH_CALUDE_expression_evaluation_l2372_237247

theorem expression_evaluation : 
  11 - 10 / 2 + (8 * 3) - 7 / 1 + 9 - 6 * 2 + 4 - 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2372_237247


namespace NUMINAMATH_CALUDE_base_sum_problem_l2372_237298

theorem base_sum_problem (G₁ G₂ : ℚ) : ∃! (S₁ S₂ : ℕ+),
  (G₁ = (4 * S₁ + 8) / (S₁^2 - 1) ∧ G₁ = (3 * S₂ + 6) / (S₂^2 - 1)) ∧
  (G₂ = (8 * S₁ + 4) / (S₁^2 - 1) ∧ G₂ = (6 * S₂ + 3) / (S₂^2 - 1)) ∧
  S₁ + S₂ = 23 := by
  sorry

end NUMINAMATH_CALUDE_base_sum_problem_l2372_237298


namespace NUMINAMATH_CALUDE_podium_height_l2372_237207

/-- The height of the podium given two configurations of books -/
theorem podium_height (l w : ℝ) (h : ℝ) : 
  l + h - w = 40 → w + h - l = 34 → h = 37 := by sorry

end NUMINAMATH_CALUDE_podium_height_l2372_237207


namespace NUMINAMATH_CALUDE_pencil_remainder_l2372_237211

theorem pencil_remainder (a b : ℕ) 
  (ha : a % 8 = 5) 
  (hb : b % 8 = 6) : 
  (a + b) % 8 = 3 := by
sorry

end NUMINAMATH_CALUDE_pencil_remainder_l2372_237211


namespace NUMINAMATH_CALUDE_assignment_methods_count_l2372_237260

/-- The number of companies available for internship --/
def num_companies : ℕ := 4

/-- The number of interns to be assigned --/
def num_interns : ℕ := 5

/-- The number of ways to assign interns to companies --/
def assignment_count : ℕ := num_companies ^ num_interns

/-- Theorem stating that the number of assignment methods is 1024 --/
theorem assignment_methods_count : assignment_count = 1024 := by
  sorry

end NUMINAMATH_CALUDE_assignment_methods_count_l2372_237260


namespace NUMINAMATH_CALUDE_matrix_is_own_inverse_l2372_237222

def A (a b c : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![a, b, c; 2, -1, 0; 0, 0, 1]

theorem matrix_is_own_inverse (a b c : ℝ) :
  A a b c * A a b c = 1 → a = 1 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_is_own_inverse_l2372_237222


namespace NUMINAMATH_CALUDE_min_episodes_watched_l2372_237200

/-- Represents the number of episodes aired on each day of the week -/
def weekly_schedule : List Nat := [0, 1, 1, 1, 1, 2, 2]

/-- The total number of episodes in the TV series -/
def total_episodes : Nat := 60

/-- The duration of Xiaogao's trip in days -/
def trip_duration : Nat := 17

/-- Calculates the maximum number of episodes that can be aired during the trip -/
def max_episodes_during_trip (schedule : List Nat) (duration : Nat) : Nat :=
  sorry

/-- Theorem: The minimum number of episodes Xiaogao can watch is 39 -/
theorem min_episodes_watched : 
  total_episodes - max_episodes_during_trip weekly_schedule trip_duration = 39 := by
  sorry

end NUMINAMATH_CALUDE_min_episodes_watched_l2372_237200


namespace NUMINAMATH_CALUDE_max_a_for_quadratic_inequality_l2372_237254

theorem max_a_for_quadratic_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + y + 3 = x * y) :
  ∃ (a_max : ℝ), ∀ (a : ℝ), 
    (∀ (x y : ℝ), x > 0 → y > 0 → x + y + 3 = x * y → 
      (x + y)^2 - a*(x + y) + 1 ≥ 0) ↔ a ≤ a_max ∧ a_max = 37/6 := by
  sorry

end NUMINAMATH_CALUDE_max_a_for_quadratic_inequality_l2372_237254


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l2372_237286

theorem x_squared_plus_reciprocal (x : ℝ) (h : 35 = x^6 + 1/x^6) :
  x^2 + 1/x^2 = 37 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l2372_237286


namespace NUMINAMATH_CALUDE_jessies_cars_l2372_237219

theorem jessies_cars (tommy : ℕ) (total : ℕ) (brother_extra : ℕ) :
  tommy = 3 →
  brother_extra = 5 →
  total = 17 →
  ∃ (jessie : ℕ), jessie = 3 ∧ tommy + jessie + (tommy + jessie + brother_extra) = total :=
by sorry

end NUMINAMATH_CALUDE_jessies_cars_l2372_237219


namespace NUMINAMATH_CALUDE_journey_speed_l2372_237237

/-- Proves that given a journey of 200 km completed in 10 hours with constant speed throughout, the speed of travel is 20 km/hr. -/
theorem journey_speed (total_distance : ℝ) (total_time : ℝ) (speed : ℝ) 
  (h1 : total_distance = 200) 
  (h2 : total_time = 10) 
  (h3 : speed * total_time = total_distance) : 
  speed = 20 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_l2372_237237


namespace NUMINAMATH_CALUDE_money_distribution_l2372_237235

theorem money_distribution (a b c d : ℤ) : 
  a + b + c + d = 600 →
  a + c = 200 →
  b + c = 350 →
  a + d = 300 →
  a ≥ 2 * b →
  c = 150 :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l2372_237235


namespace NUMINAMATH_CALUDE_intersection_equals_B_intersection_with_complement_empty_l2372_237280

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | a - 2 ≤ x ∧ x ≤ 2*a + 3}
def B : Set ℝ := {x : ℝ | x^2 - 6*x + 5 ≤ 0}

-- Theorem for the first question
theorem intersection_equals_B (a : ℝ) :
  (A a) ∩ B = B ↔ a ∈ Set.Icc 1 3 := by sorry

-- Theorem for the second question
theorem intersection_with_complement_empty (a : ℝ) :
  (A a) ∩ (Bᶜ) = ∅ ↔ a < -5 := by sorry

end NUMINAMATH_CALUDE_intersection_equals_B_intersection_with_complement_empty_l2372_237280


namespace NUMINAMATH_CALUDE_strawberry_weight_theorem_l2372_237266

/-- The weight of Marco's strawberries in pounds -/
def marco_weight : ℕ := 19

/-- The difference in weight between Marco's dad's strawberries and Marco's strawberries in pounds -/
def weight_difference : ℕ := 34

/-- The weight of Marco's dad's strawberries in pounds -/
def dad_weight : ℕ := marco_weight + weight_difference

/-- The total weight of Marco's and his dad's strawberries in pounds -/
def total_weight : ℕ := marco_weight + dad_weight

theorem strawberry_weight_theorem :
  total_weight = 72 := by sorry

end NUMINAMATH_CALUDE_strawberry_weight_theorem_l2372_237266


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_three_l2372_237212

theorem gcd_of_powers_of_three : Nat.gcd (3^1001 - 1) (3^1010 - 1) = 19682 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_three_l2372_237212


namespace NUMINAMATH_CALUDE_soccer_tournament_arrangements_l2372_237291

/-- The number of teams in the tournament -/
def num_teams : ℕ := 6

/-- The number of matches each team plays -/
def matches_per_team : ℕ := 2

/-- The total number of possible arrangements of matches -/
def total_arrangements : ℕ := 70

/-- Theorem stating the number of possible arrangements for the given conditions -/
theorem soccer_tournament_arrangements :
  ∀ (n : ℕ) (m : ℕ),
    n = num_teams →
    m = matches_per_team →
    (∀ (i j : ℕ), i < n → j < n → i ≠ j → ∃! (k : ℕ), k ≤ 1) →
    (∀ (i : ℕ), i < n → ∃ (s : Finset ℕ), s.card = m ∧ ∀ (j : ℕ), j ∈ s → j < n ∧ j ≠ i) →
    total_arrangements = 70 :=
by sorry

end NUMINAMATH_CALUDE_soccer_tournament_arrangements_l2372_237291


namespace NUMINAMATH_CALUDE_inequality_preservation_l2372_237282

theorem inequality_preservation (x y : ℝ) (h : x > y) : x - 2 > y - 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l2372_237282


namespace NUMINAMATH_CALUDE_sum_abs_zero_implies_a_minus_abs_2a_l2372_237268

theorem sum_abs_zero_implies_a_minus_abs_2a (a : ℝ) : a + |a| = 0 → a - |2*a| = 3*a := by
  sorry

end NUMINAMATH_CALUDE_sum_abs_zero_implies_a_minus_abs_2a_l2372_237268


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2372_237206

/-- Given that 5y varies inversely as the square of x, and y = 4 when x = 2, 
    prove that y = 1 when x = 4 -/
theorem inverse_variation_problem (k : ℝ) :
  (∀ x y : ℝ, x ≠ 0 → 5 * y = k / (x ^ 2)) →
  (5 * 4 = k / (2 ^ 2)) →
  ∃ y : ℝ, 5 * y = k / (4 ^ 2) ∧ y = 1 :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2372_237206


namespace NUMINAMATH_CALUDE_sarahs_reading_capacity_l2372_237216

/-- Sarah's reading problem -/
theorem sarahs_reading_capacity 
  (pages_per_hour : ℕ) 
  (pages_per_book : ℕ) 
  (available_hours : ℕ) 
  (h1 : pages_per_hour = 120) 
  (h2 : pages_per_book = 360) 
  (h3 : available_hours = 8) :
  (available_hours * pages_per_hour) / pages_per_book = 2 :=
sorry

end NUMINAMATH_CALUDE_sarahs_reading_capacity_l2372_237216


namespace NUMINAMATH_CALUDE_croissant_fold_time_l2372_237278

/-- Represents the time taken for croissant making process -/
structure CroissantTime where
  total_time : ℕ           -- Total time in minutes
  fold_count : ℕ           -- Number of times dough is folded
  rest_time : ℕ            -- Rest time for each fold in minutes
  mix_time : ℕ             -- Time to mix ingredients in minutes
  bake_time : ℕ            -- Time to bake in minutes
  fold_time : ℕ            -- Time to fold dough each time in minutes

/-- Theorem stating the time to fold the dough each time -/
theorem croissant_fold_time (c : CroissantTime) 
  (h1 : c.total_time = 6 * 60)  -- 6 hours in minutes
  (h2 : c.fold_count = 4)
  (h3 : c.rest_time = 75)
  (h4 : c.mix_time = 10)
  (h5 : c.bake_time = 30)
  (h6 : c.total_time = c.mix_time + c.bake_time + c.fold_count * c.rest_time + c.fold_count * c.fold_time) :
  c.fold_time = 5 := by
  sorry


end NUMINAMATH_CALUDE_croissant_fold_time_l2372_237278


namespace NUMINAMATH_CALUDE_binary_93_l2372_237249

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Theorem: The binary representation of 93 is [true, false, true, true, true, false, true] -/
theorem binary_93 : toBinary 93 = [true, false, true, true, true, false, true] := by
  sorry

end NUMINAMATH_CALUDE_binary_93_l2372_237249


namespace NUMINAMATH_CALUDE_least_product_of_distinct_primes_greater_than_10_l2372_237214

theorem least_product_of_distinct_primes_greater_than_10 :
  ∃ p q : ℕ,
    p.Prime ∧ q.Prime ∧
    p > 10 ∧ q > 10 ∧
    p ≠ q ∧
    p * q = 143 ∧
    ∀ a b : ℕ, a.Prime → b.Prime → a > 10 → b > 10 → a ≠ b → a * b ≥ 143 :=
by sorry

end NUMINAMATH_CALUDE_least_product_of_distinct_primes_greater_than_10_l2372_237214


namespace NUMINAMATH_CALUDE_number_multiplication_l2372_237248

theorem number_multiplication (x : ℝ) : x - 7 = 9 → 5 * x = 80 := by
  sorry

end NUMINAMATH_CALUDE_number_multiplication_l2372_237248


namespace NUMINAMATH_CALUDE_abc_sum_bounds_l2372_237225

theorem abc_sum_bounds (a b c d : ℝ) (h : a + b + c = -d) (h_d : d ≠ 0) :
  0 ≤ a * b + a * c + b * c ∧ a * b + a * c + b * c ≤ d^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_bounds_l2372_237225


namespace NUMINAMATH_CALUDE_difference_of_squares_l2372_237244

theorem difference_of_squares (m : ℝ) : m^2 - 1 = (m - 1) * (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2372_237244


namespace NUMINAMATH_CALUDE_two_oak_trees_cut_down_l2372_237223

/-- The number of oak trees cut down in the park --/
def oak_trees_cut_down (initial : ℕ) (final : ℕ) : ℕ :=
  initial - final

/-- Theorem: Given the initial and final number of oak trees, prove that 2 trees were cut down --/
theorem two_oak_trees_cut_down :
  oak_trees_cut_down 9 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_oak_trees_cut_down_l2372_237223


namespace NUMINAMATH_CALUDE_day_365_is_tuesday_l2372_237231

/-- Given a year with 365 days, if the 15th day is a Tuesday, then the 365th day is also a Tuesday. -/
theorem day_365_is_tuesday (year : ℕ) (h1 : year % 7 = 2) : (365 % 7 = year % 7) := by
  sorry

#check day_365_is_tuesday

end NUMINAMATH_CALUDE_day_365_is_tuesday_l2372_237231


namespace NUMINAMATH_CALUDE_impossible_to_cover_modified_chessboard_l2372_237271

/-- Represents a chessboard square --/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Defines the color of a square on the chessboard --/
def squareColor (s : Square) : Bool :=
  (s.row + s.col) % 2 = 0

/-- Represents the modified chessboard after removing two squares --/
def ModifiedChessboard : Set Square :=
  { s : Square | s ≠ ⟨0, 0⟩ ∧ s ≠ ⟨7, 7⟩ }

/-- A domino covers two adjacent squares --/
def validDomino (s1 s2 : Square) : Prop :=
  (s1.row = s2.row ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.col = s2.col ∧ s1.row.val + 1 = s2.row.val)

/-- A valid domino placement on the modified chessboard --/
def validPlacement (placement : Set (Square × Square)) : Prop :=
  ∀ (s1 s2 : Square), (s1, s2) ∈ placement →
    s1 ∈ ModifiedChessboard ∧ s2 ∈ ModifiedChessboard ∧ validDomino s1 s2

/-- The main theorem stating that it's impossible to cover the modified chessboard with dominos --/
theorem impossible_to_cover_modified_chessboard :
  ¬∃ (placement : Set (Square × Square)),
    validPlacement placement ∧
    (∀ s ∈ ModifiedChessboard, ∃ s1 s2, (s1, s2) ∈ placement ∧ (s = s1 ∨ s = s2)) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_cover_modified_chessboard_l2372_237271


namespace NUMINAMATH_CALUDE_largest_n_with_unique_k_l2372_237209

theorem largest_n_with_unique_k : ∃ (k : ℤ), 
  (5 : ℚ)/12 < (7 : ℚ)/(7 + k) ∧ (7 : ℚ)/(7 + k) < 4/9 ∧ 
  (∀ (m : ℕ) (j : ℤ), m > 7 → 
    ((5 : ℚ)/12 < (m : ℚ)/(m + j) ∧ (m : ℚ)/(m + j) < 4/9) → 
    (∃ (l : ℤ), l ≠ j ∧ (5 : ℚ)/12 < (m : ℚ)/(m + l) ∧ (m : ℚ)/(m + l) < 4/9)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_unique_k_l2372_237209


namespace NUMINAMATH_CALUDE_arrangements_equal_24_l2372_237293

/-- Represents the number of traditional Chinese paintings -/
def traditional_paintings : Nat := 3

/-- Represents the number of oil paintings -/
def oil_paintings : Nat := 2

/-- Represents the number of ink paintings -/
def ink_paintings : Nat := 1

/-- Calculates the number of arrangements for the paintings -/
def calculate_arrangements : Nat :=
  -- The actual calculation is not provided here
  -- It should consider the constraints mentioned in the problem
  sorry

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrangements_equal_24 : calculate_arrangements = 24 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_equal_24_l2372_237293


namespace NUMINAMATH_CALUDE_characterize_valid_pairs_l2372_237279

def is_valid_pair (n p : ℕ+) : Prop :=
  Nat.Prime p.val ∧ n.val ≤ 2 * p.val ∧ (p.val - 1)^n.val + 1 ∣ n.val^2

theorem characterize_valid_pairs :
  ∀ (n p : ℕ+), is_valid_pair n p ↔
    (n = 2 ∧ p = 2) ∨
    (n = 3 ∧ p = 3) ∨
    (n = 1 ∧ Nat.Prime p.val) :=
sorry

end NUMINAMATH_CALUDE_characterize_valid_pairs_l2372_237279


namespace NUMINAMATH_CALUDE_symmetric_points_l2372_237234

/-- Given a point M with coordinates (x, y), this theorem proves the coordinates
    of points symmetric to M with respect to x-axis, y-axis, and origin. -/
theorem symmetric_points (x y : ℝ) :
  let M : ℝ × ℝ := (x, y)
  let M_x_sym : ℝ × ℝ := (x, -y)  -- Symmetric to x-axis
  let M_y_sym : ℝ × ℝ := (-x, y)  -- Symmetric to y-axis
  let M_origin_sym : ℝ × ℝ := (-x, -y)  -- Symmetric to origin
  (M_x_sym = (x, -y)) ∧
  (M_y_sym = (-x, y)) ∧
  (M_origin_sym = (-x, -y)) := by
sorry


end NUMINAMATH_CALUDE_symmetric_points_l2372_237234


namespace NUMINAMATH_CALUDE_sequence_sum_l2372_237202

theorem sequence_sum (A B C D E F G H I J : ℝ) : 
  D = 8 →
  A + B + C + D = 45 →
  B + C + D + E = 45 →
  C + D + E + F = 45 →
  D + E + F + G = 45 →
  E + F + G + H = 45 →
  F + G + H + I = 45 →
  G + H + I + J = 45 →
  A + J = 0 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l2372_237202


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2372_237285

theorem sqrt_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.sqrt (x^2 / y) + Real.sqrt (y^2 / x) ≥ Real.sqrt x + Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2372_237285


namespace NUMINAMATH_CALUDE_rectangle_square_diagonal_intersection_l2372_237228

/-- Given a square and a rectangle with the same perimeter and a common corner,
    prove that the intersection of the rectangle's diagonals lies on the square's diagonal. -/
theorem rectangle_square_diagonal_intersection
  (s a b : ℝ) 
  (h_perimeter : 4 * s = 2 * a + 2 * b) 
  (h_positive : s > 0 ∧ a > 0 ∧ b > 0) :
  a / 2 = b / 2 := by sorry

end NUMINAMATH_CALUDE_rectangle_square_diagonal_intersection_l2372_237228


namespace NUMINAMATH_CALUDE_money_sharing_problem_l2372_237258

theorem money_sharing_problem (john jose binoy : ℕ) 
  (h1 : john + jose + binoy > 0)  -- Ensure total is positive
  (h2 : jose = 2 * john)          -- Ratio condition for Jose
  (h3 : binoy = 3 * john)         -- Ratio condition for Binoy
  (h4 : john = 2200)              -- John's share
  : john + jose + binoy = 13200 := by
  sorry

end NUMINAMATH_CALUDE_money_sharing_problem_l2372_237258


namespace NUMINAMATH_CALUDE_complex_pairs_sum_l2372_237253

theorem complex_pairs_sum : ∃ (a₁ b₁ a₂ b₂ : ℕ+), 
  a₁ < b₁ ∧ 
  a₂ < b₂ ∧ 
  (a₁ + Complex.I * b₁) * (b₁ - Complex.I * a₁) = 2020 ∧
  (a₂ + Complex.I * b₂) * (b₂ - Complex.I * a₂) = 2020 ∧
  (a₁ : ℕ) + (b₁ : ℕ) + (a₂ : ℕ) + (b₂ : ℕ) = 714 ∧
  (a₁, b₁) ≠ (a₂, b₂) :=
by sorry

end NUMINAMATH_CALUDE_complex_pairs_sum_l2372_237253


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_cone_lateral_surface_area_proof_l2372_237227

/-- The lateral surface area of a cone with base radius 3 and lateral surface that unfolds into a semicircle -/
theorem cone_lateral_surface_area : Real :=
  let base_radius : Real := 3
  let lateral_surface_is_semicircle : Prop := True  -- This is a placeholder for the condition
  18 * Real.pi

/-- Proof of the lateral surface area of the cone -/
theorem cone_lateral_surface_area_proof :
  cone_lateral_surface_area = 18 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_cone_lateral_surface_area_proof_l2372_237227


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l2372_237204

/-- 
For a quadratic equation x^2 - 3x + c to have roots in the form x = (3 ± √(2c-3)) / 2, 
c must equal 2.
-/
theorem quadratic_roots_condition (c : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + c = 0 ↔ ∃ s : ℝ, s^2 = 2*c - 3 ∧ x = (3 + s) / 2 ∨ x = (3 - s) / 2) →
  c = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l2372_237204


namespace NUMINAMATH_CALUDE_no_solution_exists_l2372_237230

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Repeated application of S function n times -/
def repeated_S (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The sum of n and repeated applications of S up to n times -/
def sum_with_repeated_S (n : ℕ) : ℕ := 
  n + (Finset.range n).sum (λ k => repeated_S n k)

/-- Theorem stating that there is no n satisfying the equation -/
theorem no_solution_exists : ¬ ∃ n : ℕ, sum_with_repeated_S n = 2000000 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2372_237230


namespace NUMINAMATH_CALUDE_sum_reciprocals_eq_2823_div_7_l2372_237287

/-- The function f(n) that returns the integer closest to the fourth root of n -/
def f (n : ℕ) : ℕ := sorry

/-- The sum of 1/f(k) for k from 1 to 2018 -/
def sum_reciprocals : ℚ :=
  (Finset.range 2018).sum (fun k => 1 / (f (k + 1) : ℚ))

/-- The theorem stating that the sum of reciprocals equals 2823/7 -/
theorem sum_reciprocals_eq_2823_div_7 : sum_reciprocals = 2823 / 7 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_eq_2823_div_7_l2372_237287


namespace NUMINAMATH_CALUDE_no_equal_roots_for_quadratic_l2372_237218

/-- The quadratic equation x^2 - (p+1)x + (p-1) = 0 has no real values of p for which its roots are equal. -/
theorem no_equal_roots_for_quadratic :
  ¬ ∃ p : ℝ, ∃ x : ℝ, x^2 - (p + 1) * x + (p - 1) = 0 ∧
    ∀ y : ℝ, y^2 - (p + 1) * y + (p - 1) = 0 → y = x :=
by sorry

end NUMINAMATH_CALUDE_no_equal_roots_for_quadratic_l2372_237218


namespace NUMINAMATH_CALUDE_students_wearing_other_colors_l2372_237265

theorem students_wearing_other_colors 
  (total_students : ℕ) 
  (blue_percent : ℚ) 
  (red_percent : ℚ) 
  (green_percent : ℚ) 
  (h1 : total_students = 900) 
  (h2 : blue_percent = 44 / 100) 
  (h3 : red_percent = 28 / 100) 
  (h4 : green_percent = 10 / 100) : 
  ℕ := by
  
  sorry

#check students_wearing_other_colors

end NUMINAMATH_CALUDE_students_wearing_other_colors_l2372_237265


namespace NUMINAMATH_CALUDE_ski_prices_solution_l2372_237273

theorem ski_prices_solution (x y : ℝ) :
  (2 * x + y = 340) ∧ (3 * x + 2 * y = 570) ↔ x = 110 ∧ y = 120 := by
  sorry

end NUMINAMATH_CALUDE_ski_prices_solution_l2372_237273


namespace NUMINAMATH_CALUDE_max_intersections_theorem_l2372_237281

/-- The number of intersection points for k lines in a plane -/
def num_intersections (k : ℕ) : ℕ := sorry

/-- The maximum number of intersection points after adding one more line to k lines -/
def max_intersections_after_adding_line (k : ℕ) : ℕ := sorry

/-- Theorem: The maximum number of intersection points after adding one more line
    to k lines is equal to the number of intersection points for k lines plus k -/
theorem max_intersections_theorem (k : ℕ) :
  max_intersections_after_adding_line k = num_intersections k + k := by sorry

end NUMINAMATH_CALUDE_max_intersections_theorem_l2372_237281


namespace NUMINAMATH_CALUDE_f_properties_l2372_237220

noncomputable def f (x : ℝ) : ℝ := Real.log (x * (Real.exp x - Real.exp (-x)) / 2)

theorem f_properties :
  (∀ x, f (-x) = f x) ∧
  (∀ x y, 0 < x → x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2372_237220


namespace NUMINAMATH_CALUDE_operation_is_multiplication_l2372_237208

theorem operation_is_multiplication : 
  ((0.137 + 0.098)^2 - (0.137 - 0.098)^2) / (0.137 * 0.098) = 4 := by
  sorry

end NUMINAMATH_CALUDE_operation_is_multiplication_l2372_237208


namespace NUMINAMATH_CALUDE_infinitely_many_satisfying_points_l2372_237241

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance squared between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- A circle with center at the origin and radius 2 -/
def Circle := {p : Point | p.x^2 + p.y^2 ≤ 4}

/-- The endpoints of a diameter of the circle -/
def diameterEndpoints : (Point × Point) :=
  ({x := -2, y := 0}, {x := 2, y := 0})

/-- The condition for a point P to satisfy the sum of squares property -/
def satisfiesSumOfSquares (p : Point) : Prop :=
  let (a, b) := diameterEndpoints
  distanceSquared p a + distanceSquared p b = 8

/-- The set of points satisfying the condition -/
def SatisfyingPoints : Set Point :=
  {p ∈ Circle | satisfiesSumOfSquares p}

theorem infinitely_many_satisfying_points :
  Set.Infinite SatisfyingPoints :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_satisfying_points_l2372_237241


namespace NUMINAMATH_CALUDE_product_of_decimals_l2372_237283

theorem product_of_decimals : (0.5 : ℝ) * 0.8 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_l2372_237283


namespace NUMINAMATH_CALUDE_selection_theorem_l2372_237238

/-- Represents the number of boys in the group -/
def num_boys : ℕ := 5

/-- Represents the number of girls in the group -/
def num_girls : ℕ := 3

/-- Represents the number of course representatives to be selected -/
def num_representatives : ℕ := 5

/-- Calculates the number of ways to select representatives under condition I -/
def selection_ways_I : ℕ := sorry

/-- Calculates the number of ways to select representatives under condition II -/
def selection_ways_II : ℕ := sorry

/-- Calculates the number of ways to select representatives under condition III -/
def selection_ways_III : ℕ := sorry

/-- Calculates the number of ways to select representatives under condition IV -/
def selection_ways_IV : ℕ := sorry

theorem selection_theorem :
  selection_ways_I = 840 ∧
  selection_ways_II = 3360 ∧
  selection_ways_III = 5400 ∧
  selection_ways_IV = 1080 := by sorry

end NUMINAMATH_CALUDE_selection_theorem_l2372_237238


namespace NUMINAMATH_CALUDE_f_inequality_l2372_237262

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x + 1 else Real.exp (x * Real.log 2)

theorem f_inequality (x : ℝ) : 
  f x + f (x - 1/2) > 1 ↔ x > -1/4 := by sorry

end NUMINAMATH_CALUDE_f_inequality_l2372_237262


namespace NUMINAMATH_CALUDE_magnified_cell_size_l2372_237250

/-- The diameter of a certain type of cell in meters -/
def cell_diameter : ℝ := 1.56e-6

/-- The magnification factor -/
def magnification : ℝ := 1e6

/-- The magnified size of the cell -/
def magnified_size : ℝ := cell_diameter * magnification

theorem magnified_cell_size :
  magnified_size = 1.56 := by sorry

end NUMINAMATH_CALUDE_magnified_cell_size_l2372_237250


namespace NUMINAMATH_CALUDE_sandcastle_ratio_l2372_237246

theorem sandcastle_ratio : 
  ∀ (j : ℕ), 
    20 + 200 + j + 5 * j = 580 →
    j / 20 = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_sandcastle_ratio_l2372_237246


namespace NUMINAMATH_CALUDE_solution_approximation_l2372_237236

/-- The solution to the equation (0.0077 * 4.5) / (x * 0.1 * 0.007) = 990 is approximately 28571.42 -/
theorem solution_approximation : ∃ x : ℝ, 
  (0.0077 * 4.5) / (x * 0.1 * 0.007) = 990 ∧ 
  abs (x - 28571.42) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_solution_approximation_l2372_237236


namespace NUMINAMATH_CALUDE_smallest_a_for_composite_f_l2372_237267

/-- A function that represents x^4 + a^2 --/
def f (x a : ℤ) : ℤ := x^4 + a^2

/-- Definition of a composite number --/
def is_composite (n : ℤ) : Prop := ∃ (a b : ℤ), a ≠ 1 ∧ a ≠ -1 ∧ a ≠ n ∧ a ≠ -n ∧ n = a * b

/-- The main theorem --/
theorem smallest_a_for_composite_f :
  ∀ x : ℤ, is_composite (f x 8) ∧
  ∀ a : ℕ, a > 0 ∧ a < 8 → ∃ x : ℤ, ¬is_composite (f x a) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_composite_f_l2372_237267


namespace NUMINAMATH_CALUDE_quadratic_rotate_translate_l2372_237263

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- Rotation of a function by 180 degrees around the origin -/
def Rotate180 (f : ℝ → ℝ) : ℝ → ℝ := fun x ↦ -f x

/-- Translation of a function upwards by d units -/
def TranslateUp (f : ℝ → ℝ) (d : ℝ) : ℝ → ℝ := fun x ↦ f x + d

/-- The theorem stating the result of rotating a quadratic function 180 degrees
    around the origin and then translating it upwards -/
theorem quadratic_rotate_translate (a b c d : ℝ) :
  (TranslateUp (Rotate180 (QuadraticFunction a b c)) d) =
  QuadraticFunction (-a) (-b) (-c + d) :=
sorry

end NUMINAMATH_CALUDE_quadratic_rotate_translate_l2372_237263


namespace NUMINAMATH_CALUDE_planar_figures_l2372_237217

-- Define the types of figures
inductive Figure
  | TwoSegmentPolyline
  | ThreeSegmentPolyline
  | TriangleClosed
  | QuadrilateralEqualOppositeSides
  | Trapezoid

-- Define what it means for a figure to be planar
def isPlanar (f : Figure) : Prop :=
  match f with
  | Figure.TwoSegmentPolyline => true
  | Figure.ThreeSegmentPolyline => false
  | Figure.TriangleClosed => true
  | Figure.QuadrilateralEqualOppositeSides => false
  | Figure.Trapezoid => true

-- Theorem statement
theorem planar_figures :
  (∀ f : Figure, isPlanar f ↔ (f = Figure.TwoSegmentPolyline ∨ f = Figure.TriangleClosed ∨ f = Figure.Trapezoid)) :=
by sorry

end NUMINAMATH_CALUDE_planar_figures_l2372_237217


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2372_237226

/-- An arithmetic sequence with common difference 1 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 1

/-- The nth term of the sequence -/
def a (n : ℕ) : ℝ := sorry

theorem arithmetic_sequence_first_term :
  arithmetic_sequence a ∧ a 5 ^ 2 = a 3 * a 11 → a 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2372_237226


namespace NUMINAMATH_CALUDE_black_balls_count_l2372_237224

theorem black_balls_count (total_balls : ℕ) (white_balls : ℕ → ℕ) (black_balls : ℕ) :
  total_balls = 56 →
  white_balls black_balls = 6 * black_balls →
  total_balls = white_balls black_balls + black_balls →
  black_balls = 8 := by
sorry

end NUMINAMATH_CALUDE_black_balls_count_l2372_237224


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_100_l2372_237215

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 5) * circle_radius
  rectangle_length * rectangle_breadth

theorem rectangle_area_is_100 :
  rectangle_area 625 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_is_100_l2372_237215


namespace NUMINAMATH_CALUDE_triangle_area_decomposition_l2372_237257

/-- Given a triangle with area T and a point inside it, through which lines are drawn parallel to each side,
    dividing the triangle into smaller parallelograms and triangles, with the areas of the resulting
    smaller triangles being T₁, T₂, and T₃, prove that √T₁ + √T₂ + √T₃ = √T. -/
theorem triangle_area_decomposition (T T₁ T₂ T₃ : ℝ) 
  (h₁ : T > 0) (h₂ : T₁ > 0) (h₃ : T₂ > 0) (h₄ : T₃ > 0) :
  Real.sqrt T₁ + Real.sqrt T₂ + Real.sqrt T₃ = Real.sqrt T := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_decomposition_l2372_237257


namespace NUMINAMATH_CALUDE_triangle_perimeter_proof_l2372_237274

theorem triangle_perimeter_proof :
  ∀ a : ℕ,
  a % 2 = 0 →
  2 < a →
  a < 14 →
  6 + 8 + a = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_proof_l2372_237274
