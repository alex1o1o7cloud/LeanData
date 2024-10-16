import Mathlib

namespace NUMINAMATH_CALUDE_asymptote_sum_l3024_302403

/-- Given an equation y = x / (x^3 + Dx^2 + Ex + F) where D, E, F are integers,
    if the graph has vertical asymptotes at x = -3, 0, and 3,
    then D + E + F = -9 -/
theorem asymptote_sum (D E F : ℤ) : 
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 3 → 
    ∃ y : ℝ, y = x / (x^3 + D*x^2 + E*x + F)) →
  D + E + F = -9 := by
  sorry

end NUMINAMATH_CALUDE_asymptote_sum_l3024_302403


namespace NUMINAMATH_CALUDE_vector_operation_proof_l3024_302462

def vector_subtraction_and_scalar_multiplication : Prop :=
  let v1 : Fin 2 → ℝ := ![3, -4]
  let v2 : Fin 2 → ℝ := ![2, -6]
  let result : Fin 2 → ℝ := ![-7, 26]
  v1 - 5 • v2 = result

theorem vector_operation_proof : vector_subtraction_and_scalar_multiplication := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l3024_302462


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3024_302441

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, (a * x^2 + b * x + 2 > 0) ↔ (-1/2 < x ∧ x < 1/3)) →
  a - b = -10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3024_302441


namespace NUMINAMATH_CALUDE_product_of_x_values_l3024_302409

theorem product_of_x_values (x : ℝ) : 
  (|18 / x + 4| = 3) → 
  (∃ y : ℝ, y ≠ x ∧ |18 / y + 4| = 3 ∧ x * y = 324 / 7) :=
by sorry

end NUMINAMATH_CALUDE_product_of_x_values_l3024_302409


namespace NUMINAMATH_CALUDE_friend_has_five_balloons_l3024_302465

/-- The number of balloons you have -/
def your_balloons : ℕ := 7

/-- The difference between your balloons and your friend's balloons -/
def difference : ℕ := 2

/-- The number of balloons your friend has -/
def friend_balloons : ℕ := your_balloons - difference

theorem friend_has_five_balloons : friend_balloons = 5 := by
  sorry

end NUMINAMATH_CALUDE_friend_has_five_balloons_l3024_302465


namespace NUMINAMATH_CALUDE_box_sales_ratio_l3024_302407

theorem box_sales_ratio (thursday_sales : ℕ) 
  (h1 : thursday_sales = 1200)
  (h2 : ∃ wednesday_sales : ℕ, wednesday_sales = 2 * thursday_sales)
  (h3 : ∃ tuesday_sales : ℕ, tuesday_sales = 2 * wednesday_sales) :
  ∃ (tuesday_sales wednesday_sales : ℕ),
    tuesday_sales = 2 * wednesday_sales ∧
    wednesday_sales = 2 * thursday_sales :=
by
  sorry

end NUMINAMATH_CALUDE_box_sales_ratio_l3024_302407


namespace NUMINAMATH_CALUDE_ahn_max_number_l3024_302477

theorem ahn_max_number : ∃ (max : ℕ), max = 700 ∧ 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 2 * (500 - n - 50) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_ahn_max_number_l3024_302477


namespace NUMINAMATH_CALUDE_equation_solution_l3024_302429

theorem equation_solution :
  ∃ x : ℚ, x + 5/8 = 2 + 3/16 - 2/3 ∧ x = 43/48 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3024_302429


namespace NUMINAMATH_CALUDE_dave_spent_on_mom_lunch_l3024_302492

def derek_initial : ℕ := 40
def derek_lunch1 : ℕ := 14
def derek_dad_lunch : ℕ := 11
def derek_lunch2 : ℕ := 5
def dave_initial : ℕ := 50
def difference_left : ℕ := 33

theorem dave_spent_on_mom_lunch :
  dave_initial - (derek_initial - derek_lunch1 - derek_dad_lunch - derek_lunch2 + difference_left) = 7 := by
  sorry

end NUMINAMATH_CALUDE_dave_spent_on_mom_lunch_l3024_302492


namespace NUMINAMATH_CALUDE_parabola_axis_equation_l3024_302497

/-- Given a parabola with equation x = (1/4)y^2, its axis equation is x = -1 -/
theorem parabola_axis_equation (y : ℝ) :
  let x := (1/4) * y^2
  (∃ p : ℝ, p/2 = 1) → (x = -1) := by
  sorry

end NUMINAMATH_CALUDE_parabola_axis_equation_l3024_302497


namespace NUMINAMATH_CALUDE_kelsey_travel_time_l3024_302488

theorem kelsey_travel_time (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_distance = 400)
  (h2 : speed1 = 25)
  (h3 : speed2 = 40) : 
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_kelsey_travel_time_l3024_302488


namespace NUMINAMATH_CALUDE_cubic_factorization_l3024_302480

theorem cubic_factorization (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3024_302480


namespace NUMINAMATH_CALUDE_triangle_area_l3024_302443

variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides
variable (S : ℝ) -- Area

-- Define the triangle
axiom triangle_sides : b + c = 5 ∧ a = Real.sqrt 7

-- Define the area formula
axiom area_formula : S = (Real.sqrt 3 / 2) * (b * c * Real.cos A)

-- State the theorem
theorem triangle_area : S = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3024_302443


namespace NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l3024_302479

/-- The y-coordinate of the vertex of the parabola y = 3x^2 - 6x + 4 is 1 -/
theorem parabola_vertex_y_coordinate :
  let f (x : ℝ) := 3 * x^2 - 6 * x + 4
  ∃ x₀ : ℝ, ∀ x : ℝ, f x ≥ f x₀ ∧ f x₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_y_coordinate_l3024_302479


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3024_302402

theorem cubic_root_sum_cubes (r s t : ℂ) : 
  (8 * r^3 + 2010 * r + 4016 = 0) →
  (8 * s^3 + 2010 * s + 4016 = 0) →
  (8 * t^3 + 2010 * t + 4016 = 0) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 1506 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3024_302402


namespace NUMINAMATH_CALUDE_mod_eight_thirteen_fourth_l3024_302431

theorem mod_eight_thirteen_fourth (m : ℕ) : 
  13^4 ≡ m [ZMOD 8] → 0 ≤ m → m < 8 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_mod_eight_thirteen_fourth_l3024_302431


namespace NUMINAMATH_CALUDE_circle_Q_equation_no_perpendicular_bisector_l3024_302426

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 4 = 0

-- Define point P
def point_P : ℝ × ℝ := (2, 0)

-- Define line l₁ passing through P and intersecting circle C at M and N
def line_l₁ (x y : ℝ) : Prop := ∃ (t : ℝ), x = 2 + t ∧ y = t ∧ circle_C x y

-- Define the length of MN
def MN_length : ℝ := 4

-- Define line ax - y + 1 = 0
def line_AB (a x y : ℝ) : Prop := a*x - y + 1 = 0

-- Theorem 1: Equation of circle Q
theorem circle_Q_equation : 
  ∀ x y : ℝ, (∃ M N : ℝ × ℝ, line_l₁ M.1 M.2 ∧ line_l₁ N.1 N.2 ∧ 
    (M.1 - N.1)^2 + (M.2 - N.2)^2 = MN_length^2) →
  ((x - 2)^2 + y^2 = 4) := 
sorry

-- Theorem 2: Non-existence of a
theorem no_perpendicular_bisector :
  ¬ ∃ a : ℝ, ∀ A B : ℝ × ℝ, 
    (line_AB a A.1 A.2 ∧ circle_C A.1 A.2 ∧ 
     line_AB a B.1 B.2 ∧ circle_C B.1 B.2 ∧ A ≠ B) →
    (∃ l₂ : ℝ → ℝ → Prop, 
      l₂ point_P.1 point_P.2 ∧
      l₂ ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) ∧
      (B.2 - A.2) * (point_P.1 - A.1) = (point_P.2 - A.2) * (B.1 - A.1)) :=
sorry

end NUMINAMATH_CALUDE_circle_Q_equation_no_perpendicular_bisector_l3024_302426


namespace NUMINAMATH_CALUDE_parabola_proof_l3024_302467

def parabola (x : ℝ) (b c : ℝ) : ℝ := x^2 + b*x + c

theorem parabola_proof :
  ∃ (b c : ℝ),
    (parabola 3 b c = 0) ∧
    (parabola 0 b c = -3) ∧
    (∀ x, parabola x b c = x^2 - 2*x - 3) ∧
    (∀ x, -1 ≤ x ∧ x ≤ 4 → parabola x b c ≤ 5) ∧
    (∀ x, -1 ≤ x ∧ x ≤ 4 → parabola x b c ≥ -4) ∧
    (∃ x, -1 ≤ x ∧ x ≤ 4 ∧ parabola x b c = 5) ∧
    (∃ x, -1 ≤ x ∧ x ≤ 4 ∧ parabola x b c = -4) :=
by
  sorry


end NUMINAMATH_CALUDE_parabola_proof_l3024_302467


namespace NUMINAMATH_CALUDE_seven_hash_three_l3024_302433

/-- Custom operator # defined for real numbers -/
def hash (a b : ℝ) : ℝ := 4*a + 2*b - 6

/-- Theorem stating that 7 # 3 = 28 -/
theorem seven_hash_three : hash 7 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_seven_hash_three_l3024_302433


namespace NUMINAMATH_CALUDE_carnival_game_ratio_l3024_302495

/-- The ratio of winners to losers in a carnival game -/
def carnival_ratio (winners losers : ℕ) : ℚ :=
  winners / losers

/-- Simplify a ratio by dividing both numerator and denominator by their GCD -/
def simplify_ratio (n d : ℕ) : ℚ :=
  (n / Nat.gcd n d) / (d / Nat.gcd n d)

theorem carnival_game_ratio :
  simplify_ratio 28 7 = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_carnival_game_ratio_l3024_302495


namespace NUMINAMATH_CALUDE_sam_seashells_l3024_302496

/-- The number of seashells Sam has after giving some to Joan -/
def remaining_seashells (initial : ℕ) (given : ℕ) : ℕ :=
  initial - given

theorem sam_seashells : remaining_seashells 35 18 = 17 := by
  sorry

end NUMINAMATH_CALUDE_sam_seashells_l3024_302496


namespace NUMINAMATH_CALUDE_correct_transformation_l3024_302457

theorem correct_transformation (x : ℝ) : (x / 2 - x / 3 = 1) ↔ (3 * x - 2 * x = 6) := by
  sorry

end NUMINAMATH_CALUDE_correct_transformation_l3024_302457


namespace NUMINAMATH_CALUDE_joans_cake_eggs_l3024_302416

/-- The number of eggs needed for baking cakes -/
def total_eggs (vanilla_count chocolate_count carrot_count : ℕ) 
               (vanilla_eggs chocolate_eggs carrot_eggs : ℕ) : ℕ :=
  vanilla_count * vanilla_eggs + chocolate_count * chocolate_eggs + carrot_count * carrot_eggs

/-- Theorem stating the total number of eggs needed for Joan's cakes -/
theorem joans_cake_eggs : 
  total_eggs 5 4 3 8 6 10 = 94 := by
  sorry

end NUMINAMATH_CALUDE_joans_cake_eggs_l3024_302416


namespace NUMINAMATH_CALUDE_distinct_triangles_count_l3024_302460

/-- Given a triangle ABC with n1 points on side AB (excluding A and B),
    n2 points on side BC (excluding B and C), and n3 points on side AC (excluding A and C),
    the number of distinct triangles formed by choosing one point from each side
    is equal to n1 * n2 * n3. -/
theorem distinct_triangles_count (n1 n2 n3 : ℕ) : ℕ :=
  n1 * n2 * n3

#check distinct_triangles_count

end NUMINAMATH_CALUDE_distinct_triangles_count_l3024_302460


namespace NUMINAMATH_CALUDE_smiley_face_tulips_l3024_302442

/-- The number of red tulips needed for one eye -/
def red_tulips_per_eye : ℕ := 8

/-- The number of purple tulips needed for one eyebrow -/
def purple_tulips_per_eyebrow : ℕ := 5

/-- The number of red tulips needed for the nose -/
def red_tulips_for_nose : ℕ := 12

/-- The number of red tulips needed for the smile -/
def red_tulips_for_smile : ℕ := 18

/-- The factor for calculating yellow background tulips -/
def yellow_background_factor : ℕ := 9

/-- The total number of tulips needed for the smiley face -/
def total_tulips : ℕ := 218

theorem smiley_face_tulips :
  (2 * red_tulips_per_eye + 2 * purple_tulips_per_eyebrow + red_tulips_for_nose + red_tulips_for_smile +
   yellow_background_factor * red_tulips_for_smile) = total_tulips := by
  sorry

end NUMINAMATH_CALUDE_smiley_face_tulips_l3024_302442


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3024_302451

theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n : ℕ, ∃ q : ℝ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 5 = 1 →
  a 9 = 81 →
  a 7 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3024_302451


namespace NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_255_sin_165_l3024_302499

theorem cos_75_cos_15_minus_sin_255_sin_165 :
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) -
  Real.sin (255 * π / 180) * Real.sin (165 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_255_sin_165_l3024_302499


namespace NUMINAMATH_CALUDE_employee_reduction_percentage_l3024_302482

def original_employees : ℝ := 227
def reduced_employees : ℝ := 195

theorem employee_reduction_percentage : 
  let difference := original_employees - reduced_employees
  let percentage := (difference / original_employees) * 100
  abs (percentage - 14.1) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_employee_reduction_percentage_l3024_302482


namespace NUMINAMATH_CALUDE_projection_matrix_values_l3024_302494

/-- A projection matrix P satisfies P² = P -/
def IsProjectionMatrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

/-- The specific matrix form given in the problem -/
def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, 15/34],
    ![c, 25/34]]

theorem projection_matrix_values :
  ∀ a c : ℚ, IsProjectionMatrix (P a c) ↔ a = 9/34 ∧ c = 15/34 := by
  sorry

end NUMINAMATH_CALUDE_projection_matrix_values_l3024_302494


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l3024_302421

theorem fraction_product_simplification :
  (18 : ℚ) / 17 * 13 / 24 * 68 / 39 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l3024_302421


namespace NUMINAMATH_CALUDE_probability_at_least_one_history_or_geography_l3024_302400

def total_outcomes : ℕ := Nat.choose 5 2

def favorable_outcomes : ℕ := Nat.choose 2 1 * Nat.choose 3 1 + Nat.choose 2 2

theorem probability_at_least_one_history_or_geography :
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_history_or_geography_l3024_302400


namespace NUMINAMATH_CALUDE_particle_final_position_l3024_302473

/-- Represents the position of a particle -/
structure Position where
  x : Int
  y : Int

/-- Calculates the position of the particle after n steps -/
def particle_position (n : Nat) : Position :=
  sorry

/-- The number of complete rectangles after 2023 minutes -/
def complete_rectangles : Nat :=
  sorry

/-- The remaining time after completing the rectangles -/
def remaining_time : Nat :=
  sorry

theorem particle_final_position :
  particle_position (complete_rectangles + 1) = Position.mk 44 1 :=
sorry

end NUMINAMATH_CALUDE_particle_final_position_l3024_302473


namespace NUMINAMATH_CALUDE_cone_roll_ratio_sum_l3024_302489

/-- Represents a right circular cone -/
structure RightCircularCone where
  r : ℝ  -- base radius
  h : ℝ  -- height
  r_pos : r > 0
  h_pos : h > 0

/-- Checks if a number is not a multiple of any prime squared -/
def not_multiple_of_prime_squared (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p^2 ∣ n)

/-- Main theorem -/
theorem cone_roll_ratio_sum (cone : RightCircularCone) 
    (m n : ℕ) (m_pos : m > 0) (n_pos : n > 0)
    (h_ratio : cone.h / cone.r = m * Real.sqrt n)
    (h_rotations : (2 * Real.pi * Real.sqrt (cone.r^2 + cone.h^2)) = 40 * Real.pi * cone.r)
    (h_not_multiple : not_multiple_of_prime_squared n) :
    m + n = 136 := by
  sorry

end NUMINAMATH_CALUDE_cone_roll_ratio_sum_l3024_302489


namespace NUMINAMATH_CALUDE_S_is_specific_set_l3024_302456

/-- A set of complex numbers satisfying certain conditions -/
def S : Set ℂ :=
  {z : ℂ | ∃ (n : ℕ), 2 < n ∧ n < 6 ∧ Complex.abs z = 1}

/-- The condition that 1 is in S -/
axiom one_in_S : (1 : ℂ) ∈ S

/-- The closure property of S -/
axiom S_closure (z₁ z₂ : ℂ) (h₁ : z₁ ∈ S) (h₂ : z₂ ∈ S) :
  z₁ - 2 * z₂ * Complex.cos (Complex.arg (z₁ / z₂)) ∈ S

/-- The theorem to be proved -/
theorem S_is_specific_set : S = {-1, 1, -Complex.I, Complex.I} := by
  sorry

end NUMINAMATH_CALUDE_S_is_specific_set_l3024_302456


namespace NUMINAMATH_CALUDE_meeting_point_theorem_l3024_302450

/-- The distance between point A and point B in kilometers -/
def distance_AB : ℝ := 120

/-- Xiao Zhang's speed in km/h -/
def speed_Zhang : ℝ := 60

/-- Xiao Wang's speed in km/h -/
def speed_Wang : ℝ := 40

/-- Time difference between Xiao Zhang and Xiao Wang's departures in hours -/
def time_difference : ℝ := 1

/-- Total travel time for both Xiao Zhang and Xiao Wang in hours -/
def total_time : ℝ := 4

/-- The meeting point of Xiao Zhang and Xiao Wang in km from point A -/
def meeting_point : ℝ := 96

theorem meeting_point_theorem :
  speed_Zhang * time_difference + 
  (speed_Zhang * speed_Wang / (speed_Zhang + speed_Wang)) * 
  (distance_AB - speed_Zhang * time_difference) = meeting_point :=
sorry

end NUMINAMATH_CALUDE_meeting_point_theorem_l3024_302450


namespace NUMINAMATH_CALUDE_count_valid_removal_sequences_for_specific_arrangement_l3024_302452

/-- Represents the arrangement of bricks -/
inductive BrickArrangement
| Empty : BrickArrangement
| Add : BrickArrangement → Nat → BrickArrangement

/-- Checks if a removal sequence is valid for a given arrangement -/
def isValidRemovalSequence (arrangement : BrickArrangement) (sequence : List Nat) : Prop := sorry

/-- Counts the number of valid removal sequences for a given arrangement -/
def countValidRemovalSequences (arrangement : BrickArrangement) : Nat := sorry

/-- The specific arrangement of 6 bricks as described in the problem -/
def specificArrangement : BrickArrangement := sorry

theorem count_valid_removal_sequences_for_specific_arrangement :
  countValidRemovalSequences specificArrangement = 10 := by sorry

end NUMINAMATH_CALUDE_count_valid_removal_sequences_for_specific_arrangement_l3024_302452


namespace NUMINAMATH_CALUDE_subsets_and_sum_of_M_l3024_302430

def M : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

theorem subsets_and_sum_of_M :
  (Finset.powerset M).card = 2^10 ∧
  (Finset.powerset M).sum (fun s => s.sum id) = 55 * 2^9 := by
  sorry

end NUMINAMATH_CALUDE_subsets_and_sum_of_M_l3024_302430


namespace NUMINAMATH_CALUDE_min_yellow_surface_fraction_l3024_302422

/-- Represents a 4x4x4 cube constructed from smaller 1-inch cubes -/
structure LargeCube where
  small_cubes : Fin 64 → Color
  blue_count : Nat
  yellow_count : Nat
  h_blue_count : blue_count = 32
  h_yellow_count : yellow_count = 32
  h_total_count : blue_count + yellow_count = 64

inductive Color
  | Blue
  | Yellow

/-- Calculates the surface area of the large cube -/
def surface_area : Nat := 6 * 4 * 4

/-- Calculates the minimum yellow surface area possible -/
def min_yellow_surface_area (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating that the minimum fraction of yellow surface area is 1/4 -/
theorem min_yellow_surface_fraction (cube : LargeCube) :
  (min_yellow_surface_area cube : ℚ) / surface_area = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_yellow_surface_fraction_l3024_302422


namespace NUMINAMATH_CALUDE_pet_store_cages_l3024_302417

theorem pet_store_cages (initial_puppies : Nat) (sold_puppies : Nat) (puppies_per_cage : Nat) : 
  initial_puppies = 18 → sold_puppies = 3 → puppies_per_cage = 5 → 
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l3024_302417


namespace NUMINAMATH_CALUDE_triangle_properties_l3024_302461

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  a + b + c = 3 →
  a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C →
  (∃ (R : ℝ), R > 0 ∧ R * (a + b + c) = a * b * Real.sin C) →
  C = π / 3 ∧
  (∀ (S : ℝ), S = π * R^2 → S ≤ π / 12) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3024_302461


namespace NUMINAMATH_CALUDE_initial_kids_on_soccer_field_l3024_302476

theorem initial_kids_on_soccer_field (initial_kids final_kids joined_kids : ℕ) :
  final_kids = initial_kids + joined_kids →
  joined_kids = 22 →
  final_kids = 36 →
  initial_kids = 14 := by
sorry

end NUMINAMATH_CALUDE_initial_kids_on_soccer_field_l3024_302476


namespace NUMINAMATH_CALUDE_university_groups_l3024_302498

theorem university_groups (total_students : ℕ) (group_reduction : ℕ) 
  (h1 : total_students = 2808)
  (h2 : group_reduction = 4)
  (h3 : ∃ (n : ℕ), n > 0 ∧ total_students % n = 0 ∧ total_students % (n + group_reduction) = 0)
  (h4 : ∀ (n : ℕ), n > 0 → total_students % n = 0 → (total_students / n < 30)) :
  ∃ (new_groups : ℕ), new_groups = 104 ∧ 
    total_students % new_groups = 0 ∧
    total_students % (new_groups + group_reduction) = 0 ∧
    total_students / new_groups < 30 :=
by sorry

end NUMINAMATH_CALUDE_university_groups_l3024_302498


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_linear_l3024_302485

theorem unique_solution_quadratic_linear (m : ℝ) :
  (∃! x : ℝ, x^2 = 4*x + m) ↔ m = -4 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_linear_l3024_302485


namespace NUMINAMATH_CALUDE_sss_sufficient_for_angle_construction_l3024_302414

/-- A triangle in a plane -/
structure Triangle :=
  (A B C : Point)

/-- Congruence relation between triangles -/
def Congruent (t1 t2 : Triangle) : Prop := sorry

/-- Length of a side in a triangle -/
def SideLength (t : Triangle) (side : Fin 3) : ℝ := sorry

/-- Angle measure in a triangle -/
def AngleMeasure (t : Triangle) (angle : Fin 3) : ℝ := sorry

/-- SSS congruence criterion -/
axiom sss_congruence (t1 t2 : Triangle) :
  (∀ i : Fin 3, SideLength t1 i = SideLength t2 i) → Congruent t1 t2

/-- Compass and straightedge construction -/
def ConstructibleAngle (θ : ℝ) : Prop := sorry

/-- Theorem: SSS is sufficient for angle construction -/
theorem sss_sufficient_for_angle_construction (θ : ℝ) (t : Triangle) :
  (∃ i : Fin 3, AngleMeasure t i = θ) →
  ConstructibleAngle θ :=
sorry

end NUMINAMATH_CALUDE_sss_sufficient_for_angle_construction_l3024_302414


namespace NUMINAMATH_CALUDE_percentage_of_360_equals_115_2_l3024_302413

theorem percentage_of_360_equals_115_2 : 
  let whole : ℝ := 360
  let part : ℝ := 115.2
  let percentage : ℝ := (part / whole) * 100
  percentage = 32 := by sorry

end NUMINAMATH_CALUDE_percentage_of_360_equals_115_2_l3024_302413


namespace NUMINAMATH_CALUDE_remainder_7n_mod_4_l3024_302415

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_7n_mod_4_l3024_302415


namespace NUMINAMATH_CALUDE_fifth_power_sum_l3024_302483

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 24)
  (h4 : a * x^4 + b * y^4 = 58) :
  a * x^5 + b * y^5 = 3004 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_sum_l3024_302483


namespace NUMINAMATH_CALUDE_company_profits_l3024_302405

theorem company_profits (revenue_prev : ℝ) (profit_prev : ℝ) (revenue_2009 : ℝ) (profit_2009 : ℝ) :
  revenue_2009 = 0.8 * revenue_prev →
  profit_2009 = 0.16 * revenue_2009 →
  profit_2009 = 1.28 * profit_prev →
  profit_prev = 0.1 * revenue_prev :=
by sorry

end NUMINAMATH_CALUDE_company_profits_l3024_302405


namespace NUMINAMATH_CALUDE_factorial_division_l3024_302478

theorem factorial_division (h : Nat.factorial 7 = 5040) :
  Nat.factorial 7 / Nat.factorial 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l3024_302478


namespace NUMINAMATH_CALUDE_no_prime_5n_plus_3_l3024_302459

theorem no_prime_5n_plus_3 : ¬∃ (n : ℕ+), 
  (∃ k : ℕ, (2 : ℤ) * n + 1 = k^2) ∧ 
  (∃ l : ℕ, (3 : ℤ) * n + 1 = l^2) ∧ 
  Nat.Prime ((5 : ℤ) * n + 3).toNat :=
by sorry

end NUMINAMATH_CALUDE_no_prime_5n_plus_3_l3024_302459


namespace NUMINAMATH_CALUDE_inequality_proof_l3024_302425

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3024_302425


namespace NUMINAMATH_CALUDE_tom_apple_purchase_l3024_302439

/-- The problem of determining how many kg of apples Tom purchased -/
theorem tom_apple_purchase (apple_price mango_price : ℕ) (mango_amount total_paid : ℕ) 
  (h1 : apple_price = 70)
  (h2 : mango_price = 70)
  (h3 : mango_amount = 9)
  (h4 : total_paid = 1190) :
  ∃ (apple_amount : ℕ), apple_amount * apple_price + mango_amount * mango_price = total_paid ∧ apple_amount = 8 := by
  sorry

end NUMINAMATH_CALUDE_tom_apple_purchase_l3024_302439


namespace NUMINAMATH_CALUDE_aquafaba_for_angel_food_cakes_l3024_302437

/-- Proves that the number of tablespoons of aquafaba needed for two angel food cakes is 32 -/
theorem aquafaba_for_angel_food_cakes 
  (aquafaba_per_egg : ℕ) 
  (cakes : ℕ) 
  (egg_whites_per_cake : ℕ) 
  (h1 : aquafaba_per_egg = 2)
  (h2 : cakes = 2)
  (h3 : egg_whites_per_cake = 8) : 
  aquafaba_per_egg * cakes * egg_whites_per_cake = 32 :=
by sorry

end NUMINAMATH_CALUDE_aquafaba_for_angel_food_cakes_l3024_302437


namespace NUMINAMATH_CALUDE_dogs_not_doing_anything_l3024_302464

def total_dogs : ℕ := 500

def running_dogs : ℕ := (18 * total_dogs) / 100
def playing_dogs : ℕ := (3 * total_dogs) / 20
def barking_dogs : ℕ := (7 * total_dogs) / 100
def digging_dogs : ℕ := total_dogs / 10
def agility_dogs : ℕ := 12
def sleeping_dogs : ℕ := (2 * total_dogs) / 25
def eating_dogs : ℕ := total_dogs / 5

def dogs_doing_something : ℕ := 
  running_dogs + playing_dogs + barking_dogs + digging_dogs + 
  agility_dogs + sleeping_dogs + eating_dogs

theorem dogs_not_doing_anything : 
  total_dogs - dogs_doing_something = 98 := by sorry

end NUMINAMATH_CALUDE_dogs_not_doing_anything_l3024_302464


namespace NUMINAMATH_CALUDE_a_left_after_ten_days_l3024_302435

/-- The number of days it takes A to complete the work -/
def days_A : ℝ := 30

/-- The number of days it takes B to complete the work -/
def days_B : ℝ := 30

/-- The number of days B worked after A left -/
def days_B_worked : ℝ := 10

/-- The number of days C worked to finish the work -/
def days_C_worked : ℝ := 10

/-- The number of days it takes C to complete the whole work -/
def days_C : ℝ := 29.999999999999996

/-- The theorem stating that A left the work after 10 days -/
theorem a_left_after_ten_days :
  ∃ (x : ℝ),
    x > 0 ∧
    x / days_A + days_B_worked / days_B + days_C_worked / days_C = 1 ∧
    x = 10 := by
  sorry

end NUMINAMATH_CALUDE_a_left_after_ten_days_l3024_302435


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3024_302446

theorem complex_equation_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (a + 2 * i) * i = b + i → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3024_302446


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_value_l3024_302432

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * x + 1

-- State the theorem
theorem tangent_line_implies_a_value (a : ℝ) (h1 : a ≠ 0) :
  (∃ (m : ℝ), (∀ x : ℝ, x + f a x - 2 = m * (x - 1)) ∧ 
               (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 1| < δ → 
                 |(f a x - f a 1) - m * (x - 1)| < ε * |x - 1|)) →
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_value_l3024_302432


namespace NUMINAMATH_CALUDE_opponent_total_score_l3024_302469

def volleyball_problem (team_scores : List Nat) : Prop :=
  let n := team_scores.length
  n = 6 ∧
  team_scores = [2, 3, 5, 7, 11, 13] ∧
  (∃ lost_scores : List Nat,
    lost_scores.length = 3 ∧
    lost_scores ⊆ team_scores ∧
    (∀ score ∈ lost_scores, ∃ opp_score, opp_score = score + 2)) ∧
  (∃ won_scores : List Nat,
    won_scores.length = 3 ∧
    won_scores ⊆ team_scores ∧
    (∀ score ∈ won_scores, ∃ opp_score, 3 * opp_score = score))

theorem opponent_total_score (team_scores : List Nat) 
  (h : volleyball_problem team_scores) : 
  (List.sum (team_scores.map (λ score => 
    if score ∈ [2, 3, 5] then score + 2 
    else score / 3))) = 25 := by
  sorry

end NUMINAMATH_CALUDE_opponent_total_score_l3024_302469


namespace NUMINAMATH_CALUDE_expression_value_theorem_l3024_302408

theorem expression_value_theorem (x : ℝ) (h : x = Real.sqrt (19 - 8 * Real.sqrt 3)) :
  (x^4 - 6*x^3 - 2*x^2 + 18*x + 23) / (x^2 - 8*x + 15) = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_theorem_l3024_302408


namespace NUMINAMATH_CALUDE_probability_of_choosing_circle_l3024_302406

theorem probability_of_choosing_circle (total : ℕ) (circles : ℕ) 
  (h1 : total = 12) (h2 : circles = 5) : 
  (circles : ℚ) / total = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_choosing_circle_l3024_302406


namespace NUMINAMATH_CALUDE_path_area_theorem_l3024_302419

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Theorem: The area of a 2.5m wide path around a 60m x 55m field is 600 sq m -/
theorem path_area_theorem :
  path_area 60 55 2.5 = 600 := by sorry

end NUMINAMATH_CALUDE_path_area_theorem_l3024_302419


namespace NUMINAMATH_CALUDE_bowTie_equation_solution_l3024_302463

-- Define the operation ⊗
noncomputable def bowTie (n h : ℝ) : ℝ := n + Real.sqrt (h + Real.sqrt (h + Real.sqrt (h + Real.sqrt h)))

-- State the theorem
theorem bowTie_equation_solution :
  ∃ h : ℝ, bowTie 5 h = 8 ∧ h = 6 := by
  sorry

end NUMINAMATH_CALUDE_bowTie_equation_solution_l3024_302463


namespace NUMINAMATH_CALUDE_v_2003_equals_5_l3024_302434

-- Define the function g
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 2
| 4 => 1
| 5 => 4
| _ => 0  -- For completeness, though not used in the problem

-- Define the sequence v
def v : ℕ → ℕ
| 0 => 5
| n + 1 => g (v n)

-- Theorem to prove
theorem v_2003_equals_5 : v 2003 = 5 := by
  sorry


end NUMINAMATH_CALUDE_v_2003_equals_5_l3024_302434


namespace NUMINAMATH_CALUDE_gcd_2024_1728_l3024_302411

theorem gcd_2024_1728 : Nat.gcd 2024 1728 = 8 := by sorry

end NUMINAMATH_CALUDE_gcd_2024_1728_l3024_302411


namespace NUMINAMATH_CALUDE_patrol_results_l3024_302423

/-- Represents the patrol records of the police car --/
def patrol_records : List Int := [6, -8, 9, -5, 4, -3]

/-- Fuel consumption rate in liters per kilometer --/
def fuel_consumption_rate : ℚ := 0.2

/-- Initial fuel in the tank in liters --/
def initial_fuel : ℚ := 5

/-- Calculates the final position of the police car --/
def final_position (records : List Int) : Int :=
  records.sum

/-- Calculates the total distance traveled --/
def total_distance (records : List Int) : Int :=
  records.map (abs) |>.sum

/-- Calculates the total fuel consumed --/
def total_fuel_consumed (distance : Int) (rate : ℚ) : ℚ :=
  (distance : ℚ) * rate

/-- Calculates the additional fuel needed --/
def additional_fuel_needed (consumed : ℚ) (initial : ℚ) : ℚ :=
  max (consumed - initial) 0

theorem patrol_results :
  (final_position patrol_records = 3) ∧
  (total_fuel_consumed (total_distance patrol_records) fuel_consumption_rate = 7) ∧
  (additional_fuel_needed (total_fuel_consumed (total_distance patrol_records) fuel_consumption_rate) initial_fuel = 2) :=
by sorry

end NUMINAMATH_CALUDE_patrol_results_l3024_302423


namespace NUMINAMATH_CALUDE_number_of_newborns_l3024_302490

/-- Proves the number of newborns in a children's home --/
theorem number_of_newborns (total_children teenagers toddlers newborns : ℕ) : 
  total_children = 40 →
  teenagers = 5 * toddlers →
  toddlers = 6 →
  total_children = teenagers + toddlers + newborns →
  newborns = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_newborns_l3024_302490


namespace NUMINAMATH_CALUDE_employee_pay_calculation_l3024_302445

/-- Given two employees with a total pay of 550 rupees, where one employee is paid 120% of the other,
    prove that the employee with lower pay receives 250 rupees. -/
theorem employee_pay_calculation (total_pay : ℝ) (x y : ℝ) : 
  total_pay = 550 →
  x = 1.2 * y →
  x + y = total_pay →
  y = 250 := by
sorry

end NUMINAMATH_CALUDE_employee_pay_calculation_l3024_302445


namespace NUMINAMATH_CALUDE_copy_machines_total_output_l3024_302448

/-- 
Given two copy machines with constant rates:
- Machine 1 makes 35 copies per minute
- Machine 2 makes 65 copies per minute

Prove that they make 3000 copies together in 30 minutes.
-/
theorem copy_machines_total_output : 
  let machine1_rate : ℕ := 35
  let machine2_rate : ℕ := 65
  let time_in_minutes : ℕ := 30
  (machine1_rate * time_in_minutes) + (machine2_rate * time_in_minutes) = 3000 := by
  sorry

end NUMINAMATH_CALUDE_copy_machines_total_output_l3024_302448


namespace NUMINAMATH_CALUDE_greatest_integer_c_l3024_302458

-- Define the numerator and denominator of the expression
def numerator (x : ℝ) : ℝ := 16 * x^3 + 5 * x^2 + 28 * x + 12
def denominator (c x : ℝ) : ℝ := x^2 + c * x + 12

-- Define the condition for the expression to have a domain of all real numbers
def has_full_domain (c : ℝ) : Prop :=
  ∀ x : ℝ, denominator c x ≠ 0

-- State the theorem
theorem greatest_integer_c :
  (∃ c : ℤ, has_full_domain (c : ℝ) ∧ 
   ∀ d : ℤ, d > c → ¬has_full_domain (d : ℝ)) ∧
  (∃ c : ℤ, c = 6 ∧ has_full_domain (c : ℝ) ∧ 
   ∀ d : ℤ, d > c → ¬has_full_domain (d : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_c_l3024_302458


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_intersection_l3024_302428

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Finds the intersection point of two line segments -/
def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

theorem quadrilateral_diagonal_intersection 
  (ABCD : Quadrilateral) 
  (hConvex : isConvex ABCD) 
  (hAB : distance ABCD.A ABCD.B = 10)
  (hCD : distance ABCD.C ABCD.D = 15)
  (hAC : distance ABCD.A ABCD.C = 17)
  (E : Point)
  (hE : E = lineIntersection ABCD.A ABCD.C ABCD.B ABCD.D)
  (hAreas : triangleArea ABCD.A E ABCD.D = triangleArea ABCD.B E ABCD.C) :
  distance ABCD.A E = 17 / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_intersection_l3024_302428


namespace NUMINAMATH_CALUDE_trapezium_side_length_l3024_302455

theorem trapezium_side_length 
  (x : ℝ) 
  (h : x > 0) 
  (area : ℝ) 
  (height : ℝ) 
  (other_side : ℝ) 
  (h_area : area = 228) 
  (h_height : height = 12) 
  (h_other_side : other_side = 18) 
  (h_trapezium_area : area = (1/2) * (x + other_side) * height) : 
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_trapezium_side_length_l3024_302455


namespace NUMINAMATH_CALUDE_reading_time_calculation_l3024_302471

theorem reading_time_calculation (total_time math_time spelling_time : ℕ) 
  (h1 : total_time = 60)
  (h2 : math_time = 15)
  (h3 : spelling_time = 18) :
  total_time - (math_time + spelling_time) = 27 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l3024_302471


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l3024_302474

/-- The surface area of a sphere circumscribing a cube with edge length 1 is 3π. -/
theorem circumscribed_sphere_surface_area (cube_edge : ℝ) (h : cube_edge = 1) :
  let sphere_radius := (Real.sqrt 3 / 2) * cube_edge
  4 * Real.pi * sphere_radius^2 = 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l3024_302474


namespace NUMINAMATH_CALUDE_zeros_before_nonzero_digit_l3024_302449

theorem zeros_before_nonzero_digit (n : ℕ) (m : ℕ) : 
  (Nat.log 10 (2^n * 5^m)).pred = n.max m := by sorry

end NUMINAMATH_CALUDE_zeros_before_nonzero_digit_l3024_302449


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3024_302447

/-- Given two positive integers with a ratio of 4:5 and HCF of 4, their LCM is 80 -/
theorem lcm_of_ratio_and_hcf (a b : ℕ+) (h_ratio : a.val * 5 = b.val * 4) 
  (h_hcf : Nat.gcd a.val b.val = 4) : Nat.lcm a.val b.val = 80 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l3024_302447


namespace NUMINAMATH_CALUDE_pipe_fill_time_l3024_302491

/-- The time it takes for a pipe to fill a tank without a leak, given:
  1. With the leak, it takes 12 hours to fill the tank.
  2. The leak alone can empty the full tank in 12 hours. -/
def fill_time_without_leak : ℝ := 6

/-- The time it takes to fill the tank with both the pipe and leak working -/
def fill_time_with_leak : ℝ := 12

/-- The time it takes for the leak to empty a full tank -/
def leak_empty_time : ℝ := 12

theorem pipe_fill_time :
  fill_time_without_leak = 6 ∧
  (1 / fill_time_without_leak - 1 / leak_empty_time = 1 / fill_time_with_leak) :=
sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l3024_302491


namespace NUMINAMATH_CALUDE_marker_problem_l3024_302493

theorem marker_problem :
  ∃ (n : ℕ) (p : ℝ), 
    p > 0 ∧
    3.51 = p * n ∧
    4.25 = p * (n + 4) ∧
    n > 0 := by
  sorry

end NUMINAMATH_CALUDE_marker_problem_l3024_302493


namespace NUMINAMATH_CALUDE_min_diff_composite_sum_105_l3024_302484

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def sum_to_105 (a b : ℕ) : Prop := a + b = 105

theorem min_diff_composite_sum_105 :
  ∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ sum_to_105 a b ∧
  ∀ (c d : ℕ), is_composite c → is_composite d → sum_to_105 c d →
  (c : ℤ) - (d : ℤ) ≥ 3 ∨ (d : ℤ) - (c : ℤ) ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_diff_composite_sum_105_l3024_302484


namespace NUMINAMATH_CALUDE_x_coordinate_at_y_3_l3024_302420

-- Define the line
def line (x y : ℝ) : Prop :=
  y + 3 = (1/2) * (x + 2)

-- Define the point (-2, -3) on the line
axiom point_on_line : line (-2) (-3)

-- Define the x-intercept
axiom x_intercept : line 4 0

-- Theorem to prove
theorem x_coordinate_at_y_3 :
  ∃ (x : ℝ), line x 3 ∧ x = 10 :=
sorry

end NUMINAMATH_CALUDE_x_coordinate_at_y_3_l3024_302420


namespace NUMINAMATH_CALUDE_function_identity_l3024_302427

theorem function_identity (f : ℕ+ → ℕ+) 
  (h : ∀ n : ℕ+, f (n + 1) > f (f n)) : 
  ∀ n : ℕ+, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l3024_302427


namespace NUMINAMATH_CALUDE_equation_solutions_l3024_302444

theorem equation_solutions : ∃ (x₁ x₂ : ℝ), 
  (x₁ - 1) * (x₁ + 3) = 12 ∧ 
  (x₂ - 1) * (x₂ + 3) = 12 ∧ 
  x₁ = -5 ∧ 
  x₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3024_302444


namespace NUMINAMATH_CALUDE_weight_order_l3024_302486

/-- Conversion factor from kilograms to grams -/
def kg_to_g : ℕ → ℕ := (· * 1000)

/-- Conversion factor from tonnes to grams -/
def t_to_g : ℕ → ℕ := (· * 1000000)

/-- Weight in grams -/
def weight_908g : ℕ := 908

/-- Weight in grams (9kg80g) -/
def weight_9kg80g : ℕ := kg_to_g 9 + 80

/-- Weight in grams (900kg) -/
def weight_900kg : ℕ := kg_to_g 900

/-- Weight in grams (0.09t) -/
def weight_009t : ℕ := t_to_g 0 + 90000

theorem weight_order :
  weight_908g < weight_9kg80g ∧
  weight_9kg80g < weight_009t ∧
  weight_009t < weight_900kg := by
  sorry

end NUMINAMATH_CALUDE_weight_order_l3024_302486


namespace NUMINAMATH_CALUDE_log_equation_solution_l3024_302468

theorem log_equation_solution : 
  ∃! x : ℝ, (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 4)) ∧ 
  (x = 11 / 2) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3024_302468


namespace NUMINAMATH_CALUDE_geometric_series_cube_sum_l3024_302438

theorem geometric_series_cube_sum (a r : ℝ) (hr : -1 < r ∧ r < 1) :
  (a / (1 - r) = 2) →
  (a^2 / (1 - r^2) = 6) →
  (a^3 / (1 - r^3) = 96/7) :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_cube_sum_l3024_302438


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l3024_302470

theorem quadratic_solution_property (a b : ℝ) : 
  (a * 1^2 + b * 1 + 1 = 0) → (3 - a - b = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l3024_302470


namespace NUMINAMATH_CALUDE_flagpole_height_is_8_l3024_302472

/-- The height of the flagpole in meters. -/
def flagpole_height : ℝ := 8

/-- The length of the rope in meters. -/
def rope_length : ℝ := flagpole_height + 2

/-- The distance the rope is pulled away from the flagpole in meters. -/
def pull_distance : ℝ := 6

theorem flagpole_height_is_8 :
  flagpole_height = 8 ∧
  rope_length = flagpole_height + 2 ∧
  flagpole_height ^ 2 + pull_distance ^ 2 = rope_length ^ 2 :=
sorry

end NUMINAMATH_CALUDE_flagpole_height_is_8_l3024_302472


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3024_302440

theorem simplify_and_evaluate (m : ℝ) (h : m = 2) : 
  (2 * m - 6) / (m^2 - 9) / ((2 * m + 2) / (m + 3)) - m / (m + 1) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3024_302440


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3024_302466

theorem repeating_decimal_to_fraction :
  ∀ (x : ℚ), (∃ (n : ℕ), x = (10^n * 6 - 6) / (10^n - 1)) → x = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3024_302466


namespace NUMINAMATH_CALUDE_train_speed_l3024_302410

/-- Given a train of length 300 meters that crosses an electric pole in 20 seconds,
    prove that its speed is 15 meters per second. -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 300) (h2 : crossing_time = 20) :
  train_length / crossing_time = 15 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l3024_302410


namespace NUMINAMATH_CALUDE_tennis_ball_box_capacity_l3024_302436

theorem tennis_ball_box_capacity :
  ∀ (total_balls : ℕ) (box_capacity : ℕ),
  (4 * box_capacity - 8 = total_balls) →
  (3 * box_capacity + 4 = total_balls) →
  box_capacity = 12 := by
sorry

end NUMINAMATH_CALUDE_tennis_ball_box_capacity_l3024_302436


namespace NUMINAMATH_CALUDE_cubic_inequality_range_l3024_302487

theorem cubic_inequality_range (m : ℝ) : 
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, m * x^3 - x^2 + 4*x + 3 ≥ 0) ↔ m ∈ Set.Icc (-6 : ℝ) (-2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_range_l3024_302487


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_m_part2_l3024_302454

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x - 3| + |x + m|

-- Part 1: Solution set of f(x) ≥ 6 when m = 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -2 ∨ x ≥ 4} :=
sorry

-- Part 2: Range of m when solution set of f(x) ≤ 5 is not empty
theorem range_of_m_part2 :
  (∃ x : ℝ, f m x ≤ 5) → m ∈ Set.Icc (-8) (-2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_m_part2_l3024_302454


namespace NUMINAMATH_CALUDE_right_triangle_area_and_hypotenuse_l3024_302424

theorem right_triangle_area_and_hypotenuse 
  (leg1 leg2 : ℝ) 
  (h_leg1 : leg1 = 30) 
  (h_leg2 : leg2 = 45) : 
  (1/2 * leg1 * leg2 = 675) ∧ 
  (Real.sqrt (leg1^2 + leg2^2) = 54) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_and_hypotenuse_l3024_302424


namespace NUMINAMATH_CALUDE_equation_system_equivalent_quadratic_l3024_302481

theorem equation_system_equivalent_quadratic (x y : ℝ) :
  (3 * x^2 + 4 * x + 7 * y + 2 = 0) ∧ (3 * x + 2 * y + 4 = 0) →
  4 * y^2 + 29 * y + 6 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_system_equivalent_quadratic_l3024_302481


namespace NUMINAMATH_CALUDE_distance_to_square_center_l3024_302475

-- Define the right triangle ABC
structure RightTriangle where
  a : ℝ  -- length of BC
  b : ℝ  -- length of AC
  h : 0 < a ∧ 0 < b  -- positive lengths

-- Define the square ABDE on the hypotenuse
structure SquareOnHypotenuse (t : RightTriangle) where
  center : ℝ × ℝ  -- coordinates of the center of the square

-- Theorem statement
theorem distance_to_square_center (t : RightTriangle) (s : SquareOnHypotenuse t) :
  Real.sqrt ((s.center.1 ^ 2) + (s.center.2 ^ 2)) = (t.a + t.b) / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_square_center_l3024_302475


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l3024_302412

theorem parallelogram_side_length 
  (s : ℝ) 
  (side1 : ℝ) 
  (side2 : ℝ) 
  (angle : ℝ) 
  (area : ℝ) 
  (h : side1 = 3 * s) 
  (h' : side2 = s) 
  (h'' : angle = π / 3) 
  (h''' : area = 9 * Real.sqrt 3) 
  (h'''' : area = side2 * side1 * Real.sin angle) : 
  s = Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l3024_302412


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_17_l3024_302401

theorem smallest_four_digit_multiple_of_17 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n → 1003 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_17_l3024_302401


namespace NUMINAMATH_CALUDE_investment_interest_rate_l3024_302418

/-- Given an investment scenario, prove the interest rate for the second investment --/
theorem investment_interest_rate 
  (total_investment : ℝ) 
  (desired_interest : ℝ) 
  (first_investment : ℝ) 
  (first_rate : ℝ) 
  (h1 : total_investment = 10000)
  (h2 : desired_interest = 980)
  (h3 : first_investment = 6000)
  (h4 : first_rate = 0.09)
  : 
  let second_investment := total_investment - first_investment
  let first_interest := first_investment * first_rate
  let second_interest := desired_interest - first_interest
  let second_rate := second_interest / second_investment
  second_rate = 0.11 := by
sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l3024_302418


namespace NUMINAMATH_CALUDE_value_std_dev_from_mean_l3024_302404

/-- Proves that for a normal distribution with mean 16.5 and standard deviation 1.5,
    the value 13.5 is 2 standard deviations less than the mean. -/
theorem value_std_dev_from_mean :
  let μ : ℝ := 16.5  -- mean
  let σ : ℝ := 1.5   -- standard deviation
  let x : ℝ := 13.5  -- value in question
  (x - μ) / σ = -2
  := by sorry

end NUMINAMATH_CALUDE_value_std_dev_from_mean_l3024_302404


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_not_q_l3024_302453

-- Define the conditions p and q
def p (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 4
def q (x : ℝ) : Prop := |x - 2| > 1

-- Define the negation of q
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem stating that p is a necessary but not sufficient condition for ¬q
theorem p_necessary_not_sufficient_for_not_q :
  (∀ x, not_q x → p x) ∧ 
  (∃ x, p x ∧ ¬(not_q x)) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_not_q_l3024_302453
