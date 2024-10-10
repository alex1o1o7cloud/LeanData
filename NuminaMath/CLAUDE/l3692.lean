import Mathlib

namespace apples_pears_equivalence_l3692_369226

-- Define the relationship between apples and pears
def apples_to_pears (apples : ℚ) : ℚ :=
  (10 / 6) * apples

-- Theorem statement
theorem apples_pears_equivalence :
  apples_to_pears (3/4 * 6) = 7.5 := by
  sorry

end apples_pears_equivalence_l3692_369226


namespace ratio_in_interval_l3692_369291

theorem ratio_in_interval (a : Fin 10 → ℕ) (h : ∀ i, a i ≤ 91) :
  ∃ i j, i ≠ j ∧ 2/3 ≤ (a i : ℚ) / (a j : ℚ) ∧ (a i : ℚ) / (a j : ℚ) ≤ 3/2 :=
sorry

end ratio_in_interval_l3692_369291


namespace arc_length_45_degrees_l3692_369230

/-- Given a circle with circumference 100 meters, the length of an arc subtended by a central angle of 45° is 12.5 meters. -/
theorem arc_length_45_degrees (D : Real) (arc_EF : Real) :
  D = 100 → -- Circumference of circle D is 100 meters
  arc_EF = D * (45 / 360) → -- Arc length is proportional to the central angle
  arc_EF = 12.5 := by
sorry

end arc_length_45_degrees_l3692_369230


namespace nonagon_diagonals_l3692_369258

/-- A convex nonagon is a 9-sided polygon -/
def ConvexNonagon : Type := Unit

/-- The number of sides in a convex nonagon -/
def num_sides (n : ConvexNonagon) : ℕ := 9

/-- The number of distinct diagonals in a convex nonagon -/
def num_diagonals (n : ConvexNonagon) : ℕ := 27

theorem nonagon_diagonals (n : ConvexNonagon) : 
  num_diagonals n = (num_sides n * (num_sides n - 3)) / 2 := by
  sorry

end nonagon_diagonals_l3692_369258


namespace agricultural_product_prices_l3692_369237

/-- Given two linear equations representing the cost of agricultural products A and B,
    prove that the unique solution for the prices of A and B is (120, 150). -/
theorem agricultural_product_prices (x y : ℚ) : 
  (2 * x + 3 * y = 690) ∧ (x + 4 * y = 720) → x = 120 ∧ y = 150 := by
  sorry

end agricultural_product_prices_l3692_369237


namespace blue_face_probability_l3692_369298

theorem blue_face_probability (total_faces : ℕ) (blue_faces : ℕ)
  (h1 : total_faces = 12)
  (h2 : blue_faces = 4) :
  (blue_faces : ℚ) / total_faces = 1 / 3 :=
by sorry

end blue_face_probability_l3692_369298


namespace power_sum_difference_l3692_369279

theorem power_sum_difference : 2^(0+1+2) - (2^0 + 2^1 + 2^2) = 1 := by
  sorry

end power_sum_difference_l3692_369279


namespace water_added_to_tank_l3692_369287

theorem water_added_to_tank (tank_capacity : ℚ) (initial_fraction : ℚ) (final_fraction : ℚ) : 
  tank_capacity = 32 →
  initial_fraction = 3/4 →
  final_fraction = 7/8 →
  final_fraction * tank_capacity - initial_fraction * tank_capacity = 4 := by
  sorry

end water_added_to_tank_l3692_369287


namespace mark_spent_40_dollars_l3692_369277

/-- The total amount Mark spent on tomatoes and apples -/
def total_spent (tomato_price : ℝ) (tomato_weight : ℝ) (apple_price : ℝ) (apple_weight : ℝ) : ℝ :=
  tomato_price * tomato_weight + apple_price * apple_weight

/-- Proof that Mark spent $40 in total -/
theorem mark_spent_40_dollars : total_spent 5 2 6 5 = 40 := by
  sorry

end mark_spent_40_dollars_l3692_369277


namespace topsoil_discounted_cost_l3692_369212

-- Define constants
def price_per_cubic_foot : ℝ := 7
def purchase_volume_yards : ℝ := 10
def discount_threshold : ℝ := 200
def discount_rate : ℝ := 0.05

-- Define conversion factor
def cubic_yards_to_cubic_feet : ℝ := 27

-- Theorem statement
theorem topsoil_discounted_cost :
  let volume_feet := purchase_volume_yards * cubic_yards_to_cubic_feet
  let base_cost := volume_feet * price_per_cubic_foot
  let discounted_cost := if volume_feet > discount_threshold
                         then base_cost * (1 - discount_rate)
                         else base_cost
  discounted_cost = 1795.50 := by
sorry

end topsoil_discounted_cost_l3692_369212


namespace ariel_fish_count_l3692_369259

theorem ariel_fish_count (total : ℕ) (male_fraction : ℚ) (female_count : ℕ) : 
  total = 45 → 
  male_fraction = 2/3 → 
  female_count = total - (total * male_fraction).num → 
  female_count = 15 :=
by sorry

end ariel_fish_count_l3692_369259


namespace circle_center_coordinates_l3692_369213

/-- The center coordinates of a circle with equation (x-h)^2 + (y-k)^2 = r^2 are (h, k) -/
theorem circle_center_coordinates (h k r : ℝ) :
  let circle_equation := fun (x y : ℝ) ↦ (x - h)^2 + (y - k)^2 = r^2
  circle_equation = fun (x y : ℝ) ↦ (x - 2)^2 + (y + 3)^2 = 1 →
  (h, k) = (2, -3) := by
  sorry

#check circle_center_coordinates

end circle_center_coordinates_l3692_369213


namespace desktop_revenue_is_12000_l3692_369285

/-- The revenue generated from the sale of desktop computers in Mr. Lu's store --/
def desktop_revenue (total_computers : ℕ) (laptop_price netbook_price desktop_price : ℕ) : ℕ :=
  let laptop_count := total_computers / 2
  let netbook_count := total_computers / 3
  let desktop_count := total_computers - laptop_count - netbook_count
  desktop_count * desktop_price

/-- Theorem stating the revenue from desktop computers --/
theorem desktop_revenue_is_12000 :
  desktop_revenue 72 750 500 1000 = 12000 := by
  sorry

end desktop_revenue_is_12000_l3692_369285


namespace sphere_circular_cross_section_l3692_369201

-- Define the types of solids
inductive Solid
| Cylinder
| Cone
| Sphere
| Frustum

-- Define the types of cross-section shapes
inductive CrossSectionShape
| Rectangular
| Triangular
| Circular
| IsoscelesTrapezoid

-- Function to get the cross-section shape of a solid through its axis of rotation
def crossSectionThroughAxis (s : Solid) : CrossSectionShape :=
  match s with
  | Solid.Cylinder => CrossSectionShape.Rectangular
  | Solid.Cone => CrossSectionShape.Triangular
  | Solid.Sphere => CrossSectionShape.Circular
  | Solid.Frustum => CrossSectionShape.IsoscelesTrapezoid

-- Theorem stating that only the Sphere has a circular cross-section through its axis of rotation
theorem sphere_circular_cross_section :
  ∀ s : Solid, crossSectionThroughAxis s = CrossSectionShape.Circular ↔ s = Solid.Sphere :=
by
  sorry


end sphere_circular_cross_section_l3692_369201


namespace pages_with_text_l3692_369243

/-- Given a book with the following properties:
  * It has 98 pages in total
  * Half of the pages are filled with images
  * 11 pages are for introduction
  * The remaining pages are equally split between blank and text
  Prove that the number of pages with text is 19 -/
theorem pages_with_text (total_pages : ℕ) (image_pages : ℕ) (intro_pages : ℕ) : 
  total_pages = 98 →
  image_pages = total_pages / 2 →
  intro_pages = 11 →
  (total_pages - image_pages - intro_pages) % 2 = 0 →
  (total_pages - image_pages - intro_pages) / 2 = 19 :=
by sorry

end pages_with_text_l3692_369243


namespace bus_cost_proof_l3692_369257

/-- The cost of a bus ride from town P to town Q -/
def bus_cost : ℝ := 1.50

/-- The cost of a train ride from town P to town Q -/
def train_cost : ℝ := bus_cost + 6.85

/-- The total cost of one train ride and one bus ride -/
def total_cost : ℝ := 9.85

theorem bus_cost_proof : bus_cost = 1.50 := by sorry

end bus_cost_proof_l3692_369257


namespace power_tower_mod_500_l3692_369289

theorem power_tower_mod_500 : 5^(5^(5^5)) ≡ 125 [ZMOD 500] := by sorry

end power_tower_mod_500_l3692_369289


namespace m_less_than_two_necessary_not_sufficient_l3692_369267

/-- The condition for the quadratic inequality x^2 + mx + 1 > 0 to have ℝ as its solution set -/
def has_real_solution_set (m : ℝ) : Prop :=
  ∀ x, x^2 + m*x + 1 > 0

/-- The statement that m < 2 is a necessary but not sufficient condition -/
theorem m_less_than_two_necessary_not_sufficient :
  (∀ m, has_real_solution_set m → m < 2) ∧
  ¬(∀ m, m < 2 → has_real_solution_set m) := by sorry

end m_less_than_two_necessary_not_sufficient_l3692_369267


namespace ABCDE_binary_digits_l3692_369227

-- Define the base-16 number ABCDE₁₆
def ABCDE : ℕ := 10 * 16^4 + 11 * 16^3 + 12 * 16^2 + 13 * 16^1 + 14

-- Theorem stating that ABCDE₁₆ has 20 binary digits
theorem ABCDE_binary_digits : 
  2^19 ≤ ABCDE ∧ ABCDE < 2^20 :=
by sorry

end ABCDE_binary_digits_l3692_369227


namespace sequence_general_term_l3692_369254

theorem sequence_general_term (a : ℕ+ → ℝ) (h1 : a 1 = 2) 
  (h2 : ∀ n : ℕ+, n * a (n + 1) = (n + 1) * a n + 2) : 
  ∀ n : ℕ+, a n = 4 * n - 2 := by
sorry

end sequence_general_term_l3692_369254


namespace point_line_distance_l3692_369271

/-- A type representing points on a line -/
structure Point where
  x : ℝ

/-- Distance between two points -/
def dist (p q : Point) : ℝ := |p.x - q.x|

theorem point_line_distance (A : Fin 11 → Point) :
  (dist (A 0) (A 10) = 56) →
  (∀ i, i < 9 → dist (A i) (A (i + 2)) ≤ 12) →
  (∀ j, j < 8 → dist (A j) (A (j + 3)) ≥ 17) →
  dist (A 1) (A 6) = 29 := by
sorry

end point_line_distance_l3692_369271


namespace tim_payment_proof_l3692_369222

/-- Represents the available bills Tim has -/
structure AvailableBills where
  tens : Nat
  fives : Nat
  ones : Nat

/-- Calculates the minimum number of bills needed to pay a given amount -/
def minBillsNeeded (bills : AvailableBills) (amount : Nat) : Nat :=
  sorry

theorem tim_payment_proof (bills : AvailableBills) (amount : Nat) :
  bills.tens = 13 ∧ bills.fives = 11 ∧ bills.ones = 17 ∧ amount = 128 →
  minBillsNeeded bills amount = 16 := by
  sorry

end tim_payment_proof_l3692_369222


namespace intersection_A_complement_B_l3692_369270

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- Define set B
def B : Set ℝ := { y | ∃ x ∈ A, y = x + 1 }

-- Statement to prove
theorem intersection_A_complement_B : A ∩ (U \ B) = Icc 0 1 := by sorry

end intersection_A_complement_B_l3692_369270


namespace smallest_number_l3692_369274

def base_6_to_decimal (x : ℕ) : ℕ := x

def base_4_to_decimal (x : ℕ) : ℕ := x

def base_2_to_decimal (x : ℕ) : ℕ := x

theorem smallest_number 
  (h1 : base_6_to_decimal 210 = 78)
  (h2 : base_4_to_decimal 100 = 16)
  (h3 : base_2_to_decimal 111111 = 63) :
  base_4_to_decimal 100 < base_6_to_decimal 210 ∧ 
  base_4_to_decimal 100 < base_2_to_decimal 111111 :=
sorry

end smallest_number_l3692_369274


namespace room_width_calculation_l3692_369299

/-- Proves that given a rectangular room with length 5.5 m and a total paving cost of $16,500 at $800 per square meter, the width of the room is 3.75 m. -/
theorem room_width_calculation (length : Real) (cost_per_sqm : Real) (total_cost : Real) :
  length = 5.5 →
  cost_per_sqm = 800 →
  total_cost = 16500 →
  (total_cost / cost_per_sqm) / length = 3.75 := by
  sorry

#check room_width_calculation

end room_width_calculation_l3692_369299


namespace product_of_sums_and_differences_l3692_369245

theorem product_of_sums_and_differences (W X Y Z : ℝ) : 
  W = (Real.sqrt 2025 + Real.sqrt 2024) →
  X = (-Real.sqrt 2025 - Real.sqrt 2024) →
  Y = (Real.sqrt 2025 - Real.sqrt 2024) →
  Z = (Real.sqrt 2024 - Real.sqrt 2025) →
  W * X * Y * Z = 1 := by
  sorry

end product_of_sums_and_differences_l3692_369245


namespace triangle_max_area_l3692_369219

theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a * Real.cos C + c * Real.cos A = 3 →
  a^2 + c^2 = 9 + a*c →
  (∃ (S : ℝ), S = (1/2) * a * c * Real.sin B ∧
    ∀ (S' : ℝ), S' = (1/2) * a * c * Real.sin B → S' ≤ S) →
  S = (9 * Real.sqrt 3) / 4 :=
by sorry

end triangle_max_area_l3692_369219


namespace zu_chongzhi_pi_calculation_l3692_369238

/-- Represents a historical mathematician -/
structure Mathematician where
  name : String
  calculating_pi : Bool
  decimal_places : ℕ
  father_of_pi : Bool

/-- The mathematician who calculated π to the 9th decimal place in ancient China -/
def ancient_chinese_pi_calculator : Mathematician :=
  { name := "Zu Chongzhi",
    calculating_pi := true,
    decimal_places := 9,
    father_of_pi := true }

/-- Theorem stating that Zu Chongzhi calculated π to the 9th decimal place and is known as the "Father of π" -/
theorem zu_chongzhi_pi_calculation :
  ∃ (m : Mathematician), m.calculating_pi ∧ m.decimal_places = 9 ∧ m.father_of_pi ∧ m.name = "Zu Chongzhi" :=
by
  sorry

end zu_chongzhi_pi_calculation_l3692_369238


namespace sin_thirty_degrees_l3692_369235

theorem sin_thirty_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end sin_thirty_degrees_l3692_369235


namespace AC_length_approx_l3692_369246

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the conditions
def satisfies_conditions (q : Quadrilateral) : Prop :=
  let dist := λ p1 p2 : ℝ × ℝ => Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  dist q.A q.B = 15 ∧ 
  dist q.D q.C = 24 ∧ 
  dist q.A q.D = 9

-- Theorem statement
theorem AC_length_approx (q : Quadrilateral) 
  (h : satisfies_conditions q) : 
  ∃ ε > 0, |dist q.A q.C - 30.7| < ε :=
sorry

#check AC_length_approx

end AC_length_approx_l3692_369246


namespace no_rational_solution_and_unique_perfect_square_l3692_369297

theorem no_rational_solution_and_unique_perfect_square :
  (∀ a : ℕ, ¬∃ x y z : ℚ, x^2 + y^2 + z^2 = 8 * a + 7) ∧
  (∀ n : ℕ, (∃ k : ℤ, 7^n + 8 = k^2) ↔ n = 0) :=
by sorry

end no_rational_solution_and_unique_perfect_square_l3692_369297


namespace original_statement_converse_is_false_inverse_is_false_neither_converse_nor_inverse_true_l3692_369204

-- Define the properties of triangles
def is_equilateral (t : Triangle) : Prop := sorry
def is_isosceles (t : Triangle) : Prop := sorry

-- The original statement
theorem original_statement (t : Triangle) : is_equilateral t → is_isosceles t := sorry

-- The converse is false
theorem converse_is_false : ¬(∀ t : Triangle, is_isosceles t → is_equilateral t) := sorry

-- The inverse is false
theorem inverse_is_false : ¬(∀ t : Triangle, ¬is_equilateral t → ¬is_isosceles t) := sorry

-- Main theorem: Neither the converse nor the inverse is true
theorem neither_converse_nor_inverse_true : 
  (¬(∀ t : Triangle, is_isosceles t → is_equilateral t)) ∧ 
  (¬(∀ t : Triangle, ¬is_equilateral t → ¬is_isosceles t)) := sorry

end original_statement_converse_is_false_inverse_is_false_neither_converse_nor_inverse_true_l3692_369204


namespace circle_line_intersection_range_l3692_369263

theorem circle_line_intersection_range (k : ℝ) : 
  (∃ (a b : ℝ), (b = k * a - 2) ∧ 
    (∃ (x y : ℝ), (x^2 + y^2 + 8*x + 15 = 0) ∧ 
      ((x - a)^2 + (y - b)^2 = 1))) →
  -4/3 ≤ k ∧ k ≤ 0 :=
by sorry

end circle_line_intersection_range_l3692_369263


namespace two_prime_pairs_sum_to_100_l3692_369231

def isPrime (n : ℕ) : Prop := sorry

theorem two_prime_pairs_sum_to_100 : 
  ∃! (count : ℕ), ∃ (S : Finset (ℕ × ℕ)), 
    (∀ (p q : ℕ), (p, q) ∈ S ↔ isPrime p ∧ isPrime q ∧ p + q = 100 ∧ p ≤ q) ∧
    S.card = count ∧
    count = 2 :=
sorry

end two_prime_pairs_sum_to_100_l3692_369231


namespace salary_decrease_theorem_l3692_369224

/-- Represents the decrease in average salary of all employees per day -/
def salary_decrease (illiterate_count : ℕ) (literate_count : ℕ) (old_wage : ℕ) (new_wage : ℕ) : ℚ :=
  let total_employees := illiterate_count + literate_count
  let wage_decrease := old_wage - new_wage
  let total_decrease := illiterate_count * wage_decrease
  (total_decrease : ℚ) / total_employees

/-- Theorem stating the decrease in average salary of all employees per day -/
theorem salary_decrease_theorem :
  salary_decrease 20 10 25 10 = 10 := by sorry

end salary_decrease_theorem_l3692_369224


namespace angle_bisector_sum_l3692_369233

-- Define the triangle vertices
def P : ℝ × ℝ := (-8, 5)
def Q : ℝ × ℝ := (-15, -19)
def R : ℝ × ℝ := (1, -7)

-- Define the equation of the angle bisector
def angle_bisector_equation (a c : ℝ) (x y : ℝ) : Prop :=
  a * x + 2 * y + c = 0

-- Theorem statement
theorem angle_bisector_sum (a c : ℝ) :
  (∃ x y, angle_bisector_equation a c x y ∧
          (x, y) ≠ P ∧
          (∃ t : ℝ, (1 - t) • P + t • Q = (x, y) ∨ (1 - t) • P + t • R = (x, y))) →
  a + c = 89 :=
sorry

end angle_bisector_sum_l3692_369233


namespace smallest_initial_value_l3692_369260

theorem smallest_initial_value : 
  ∃ (x : ℕ), x + 42 = 456 ∧ ∀ (y : ℕ), y + 42 = 456 → x ≤ y :=
by sorry

end smallest_initial_value_l3692_369260


namespace factorial_ratio_l3692_369272

theorem factorial_ratio : Nat.factorial 10 / (Nat.factorial 7 * Nat.factorial 2 * Nat.factorial 1) = 360 := by
  sorry

end factorial_ratio_l3692_369272


namespace binomial_9_6_l3692_369200

theorem binomial_9_6 : Nat.choose 9 6 = 84 := by
  sorry

end binomial_9_6_l3692_369200


namespace exists_bisecting_line_l3692_369234

/-- A convex figure in the plane -/
structure ConvexFigure where
  -- We don't define the internal structure of ConvexFigure,
  -- as it's not necessary for the statement of the theorem

/-- A line in the plane -/
structure Line where
  -- We don't define the internal structure of Line,
  -- as it's not necessary for the statement of the theorem

/-- The perimeter of a convex figure -/
noncomputable def perimeter (F : ConvexFigure) : ℝ :=
  sorry

/-- The area of a convex figure -/
noncomputable def area (F : ConvexFigure) : ℝ :=
  sorry

/-- Predicate to check if a line bisects the perimeter of a convex figure -/
def bisects_perimeter (l : Line) (F : ConvexFigure) : Prop :=
  sorry

/-- Predicate to check if a line bisects the area of a convex figure -/
def bisects_area (l : Line) (F : ConvexFigure) : Prop :=
  sorry

/-- Theorem: For any convex figure in the plane, there exists a line that
    simultaneously bisects both its perimeter and area -/
theorem exists_bisecting_line (F : ConvexFigure) :
  ∃ l : Line, bisects_perimeter l F ∧ bisects_area l F :=
sorry

end exists_bisecting_line_l3692_369234


namespace inequality_system_solutions_l3692_369276

theorem inequality_system_solutions : 
  {x : ℤ | x ≥ 0 ∧ 4 * (x + 1) ≤ 7 * x + 10 ∧ x - 5 < (x - 8) / 3} = {0, 1, 2, 3} := by
  sorry

end inequality_system_solutions_l3692_369276


namespace g_is_even_l3692_369220

-- Define the function f on ℝ
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + f (-x)

-- Theorem stating that g is an even function
theorem g_is_even : ∀ x : ℝ, g f x = g f (-x) := by
  sorry

end g_is_even_l3692_369220


namespace altered_prism_edges_l3692_369248

/-- Represents a rectangular prism that has been altered by truncating vertices and cutting faces diagonally -/
structure AlteredPrism where
  initialEdges : Nat
  initialVertices : Nat
  initialFaces : Nat
  newEdgesPerTruncatedVertex : Nat
  newEdgesPerCutFace : Nat

/-- Calculates the total number of edges in the altered prism -/
def totalEdges (p : AlteredPrism) : Nat :=
  p.initialEdges + 
  (p.initialVertices * p.newEdgesPerTruncatedVertex) + 
  (p.initialFaces * p.newEdgesPerCutFace)

/-- Theorem stating that the altered rectangular prism has 42 edges -/
theorem altered_prism_edges :
  ∀ (p : AlteredPrism),
    p.initialEdges = 12 →
    p.initialVertices = 8 →
    p.initialFaces = 6 →
    p.newEdgesPerTruncatedVertex = 3 →
    p.newEdgesPerCutFace = 1 →
    totalEdges p = 42 := by
  sorry

#check altered_prism_edges

end altered_prism_edges_l3692_369248


namespace world_population_scientific_notation_l3692_369215

/-- The world's population in billions -/
def world_population : ℝ := 8

/-- Scientific notation representation of a number -/
def scientific_notation (n : ℝ) (base : ℝ) (exponent : ℤ) : Prop :=
  n = base * (10 : ℝ) ^ exponent ∧ 1 ≤ base ∧ base < 10

/-- Theorem: The world population of 8 billion in scientific notation is 8 × 10^9 -/
theorem world_population_scientific_notation :
  scientific_notation (world_population * 1000000000) 8 9 := by
  sorry

end world_population_scientific_notation_l3692_369215


namespace sarahs_age_l3692_369228

theorem sarahs_age (game_formula : ℕ → ℕ) (name_letters : ℕ) (marriage_age : ℕ) :
  game_formula name_letters = marriage_age →
  name_letters = 5 →
  marriage_age = 23 →
  ∃ current_age : ℕ, game_formula 5 = 23 ∧ current_age = 9 :=
by sorry

end sarahs_age_l3692_369228


namespace angle_sum_in_polygon_l3692_369278

theorem angle_sum_in_polygon (D E F p q : ℝ) : 
  D = 38 → E = 58 → F = 36 → 
  D + E + (360 - p) + 90 + (126 - q) = 540 → 
  p + q = 132 :=
by
  sorry

end angle_sum_in_polygon_l3692_369278


namespace geometric_sequence_property_l3692_369273

def is_geometric_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ+, a (n + 1) = q * a n

def satisfies_condition (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a n * a (n + 3) = a (n + 1) * a (n + 2)

theorem geometric_sequence_property :
  (∀ a : ℕ+ → ℝ, is_geometric_sequence a → satisfies_condition a) ∧
  (∃ a : ℕ+ → ℝ, satisfies_condition a ∧ ¬is_geometric_sequence a) :=
sorry

end geometric_sequence_property_l3692_369273


namespace prime_pairs_square_sum_l3692_369216

theorem prime_pairs_square_sum (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ n : ℕ, p^2 + 5*p*q + 4*q^2 = n^2) ↔ ((p = 13 ∧ q = 3) ∨ (p = 7 ∧ q = 5) ∨ (p = 5 ∧ q = 11)) :=
sorry

end prime_pairs_square_sum_l3692_369216


namespace age_sum_is_37_l3692_369244

/-- Given the ages of A, B, and C, prove their sum is 37 -/
theorem age_sum_is_37 (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 14) :
  a + b + c = 37 := by
  sorry

end age_sum_is_37_l3692_369244


namespace circle_common_chord_l3692_369202

/-- Given two circles with equations x^2 + y^2 = a^2 and x^2 + y^2 + ay - 6 = 0,
    where the common chord length is 2√3, prove that a = ±2 -/
theorem circle_common_chord (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = a^2) ∧ 
  (∃ (x y : ℝ), x^2 + y^2 + a*y - 6 = 0) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 = a^2 ∧ 
    x₁^2 + y₁^2 + a*y₁ - 6 = 0 ∧
    x₂^2 + y₂^2 = a^2 ∧ 
    x₂^2 + y₂^2 + a*y₂ - 6 = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 12) →
  a = 2 ∨ a = -2 :=
by sorry

end circle_common_chord_l3692_369202


namespace gcd_180_270_l3692_369284

theorem gcd_180_270 : Nat.gcd 180 270 = 90 := by
  sorry

end gcd_180_270_l3692_369284


namespace jellyfish_cost_l3692_369268

theorem jellyfish_cost (jellyfish_cost eel_cost : ℝ) : 
  eel_cost = 9 * jellyfish_cost →
  jellyfish_cost + eel_cost = 200 →
  jellyfish_cost = 20 := by
sorry

end jellyfish_cost_l3692_369268


namespace intern_distribution_theorem_l3692_369232

def distribute_interns (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem intern_distribution_theorem :
  distribute_interns 4 3 = 36 :=
sorry

end intern_distribution_theorem_l3692_369232


namespace f_symmetric_l3692_369207

noncomputable def f (x : ℝ) : ℝ :=
  (6 * Real.cos (Real.pi + x) + 5 * (Real.sin (Real.pi - x))^2 - 4) / Real.cos (2 * Real.pi - x)

theorem f_symmetric (m : ℝ) (h : f m = 2) : f (-m) = 2 := by
  sorry

end f_symmetric_l3692_369207


namespace peanut_butter_price_increase_peanut_butter_problem_l3692_369252

/-- Calculates the new average price of returned peanut butter cans after a price increase -/
theorem peanut_butter_price_increase (initial_avg_price : ℚ) (num_cans : ℕ) 
  (price_increase : ℚ) (num_returned : ℕ) (remaining_avg_price : ℚ) : ℚ :=
  let total_initial_cost := initial_avg_price * num_cans
  let new_price_per_can := initial_avg_price * (1 + price_increase)
  let total_new_cost := new_price_per_can * num_cans
  let num_remaining := num_cans - num_returned
  let total_remaining_cost := remaining_avg_price * num_remaining
  let total_returned_cost := total_new_cost - total_remaining_cost
  let new_avg_returned_price := total_returned_cost / num_returned
  new_avg_returned_price

/-- The new average price of the two returned peanut butter cans is 65.925 cents -/
theorem peanut_butter_problem : 
  peanut_butter_price_increase (36.5 / 100) 6 (15 / 100) 2 (30 / 100) = 65925 / 100000 := by
  sorry

end peanut_butter_price_increase_peanut_butter_problem_l3692_369252


namespace equation_simplification_l3692_369247

theorem equation_simplification (x : ℝ) (h : x ≠ 1) :
  (1 / (x - 1) + 3) * (x - 1) = 1 + 3 * (x - 1) ∧
  3 * x / (1 - x) * (x - 1) = -3 * x ∧
  1 + 3 * (x - 1) = -3 * x :=
sorry

end equation_simplification_l3692_369247


namespace subtracted_value_problem_solution_l3692_369211

theorem subtracted_value (chosen_number : ℕ) (final_answer : ℕ) : ℕ :=
  let divided_result := chosen_number / 8
  divided_result - final_answer

theorem problem_solution :
  subtracted_value 1376 12 = 160 := by
  sorry

end subtracted_value_problem_solution_l3692_369211


namespace work_completion_proof_l3692_369225

/-- The number of days it takes for person B to complete the work alone -/
def b_days : ℝ := 20

/-- The number of days A and B work together -/
def work_together_days : ℝ := 7

/-- The fraction of work left after A and B work together for 7 days -/
def work_left : ℝ := 0.18333333333333335

/-- The number of days it takes for person A to complete the work alone -/
def a_days : ℝ := 15

theorem work_completion_proof :
  (work_together_days * (1 / a_days + 1 / b_days) = 1 - work_left) :=
by sorry

end work_completion_proof_l3692_369225


namespace x_equals_two_l3692_369286

theorem x_equals_two (some_number : ℝ) (h : x + some_number = 3) : x = 2 := by
  sorry

end x_equals_two_l3692_369286


namespace binary_to_decimal_111_l3692_369293

theorem binary_to_decimal_111 : 
  (1 : ℕ) * 2^0 + (1 : ℕ) * 2^1 + (1 : ℕ) * 2^2 = 7 := by
  sorry

end binary_to_decimal_111_l3692_369293


namespace ellipse_parameter_sum_l3692_369290

/-- An ellipse with foci F₁ and F₂, and constant sum of distances from any point to the foci. -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  sum_distances : ℝ

/-- The center, semi-major axis, and semi-minor axis of an ellipse. -/
structure EllipseParameters where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Theorem: For the given ellipse, h + k + a + b = 9 + √7 -/
theorem ellipse_parameter_sum (E : Ellipse) (P : EllipseParameters) :
  E.F₁ = (0, 2) →
  E.F₂ = (6, 2) →
  E.sum_distances = 8 →
  P.h + P.k + P.a + P.b = 9 + Real.sqrt 7 :=
by sorry

end ellipse_parameter_sum_l3692_369290


namespace range_of_a_given_max_value_l3692_369239

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := |x^2 - 2*x - a| + a

/-- The theorem stating the range of a given the maximum value of f(x) -/
theorem range_of_a_given_max_value :
  (∃ (a : ℝ), (∀ x ∈ Set.Icc (-1) 3, f a x ≤ 3) ∧ 
   (∃ x ∈ Set.Icc (-1) 3, f a x = 3)) ↔ 
  (∀ a : ℝ, a ≤ -1) := by sorry

end range_of_a_given_max_value_l3692_369239


namespace rectangle_measurement_error_l3692_369221

theorem rectangle_measurement_error (x : ℝ) : 
  (((1 + x / 100) * 0.9) = 1.08) → x = 20 := by
  sorry

end rectangle_measurement_error_l3692_369221


namespace circle_op_twelve_seven_l3692_369294

def circle_op (a b : ℝ) : ℝ := (a + b) * (a - b)

theorem circle_op_twelve_seven :
  circle_op 12 7 = 95 := by sorry

end circle_op_twelve_seven_l3692_369294


namespace exists_simultaneous_j_half_no_universal_j_half_l3692_369242

/-- A number is a j-half if it leaves a remainder of j when divided by 2j+1 -/
def is_j_half (n j : ℕ) : Prop := n % (2 * j + 1) = j

/-- For any positive integer k, there exists a number that is simultaneously a j-half for j = 1, 2, ..., k -/
theorem exists_simultaneous_j_half (k : ℕ) : ∃ n : ℕ, ∀ j ∈ Finset.range k, is_j_half n j := by sorry

/-- There is no number which is a j-half for all positive integers j -/
theorem no_universal_j_half : ¬∃ n : ℕ, ∀ j : ℕ, j > 0 → is_j_half n j := by sorry

end exists_simultaneous_j_half_no_universal_j_half_l3692_369242


namespace books_loaned_out_l3692_369280

theorem books_loaned_out (initial_books : ℕ) (final_books : ℕ) (return_rate : ℚ) :
  initial_books = 150 →
  final_books = 100 →
  return_rate = 3/5 →
  (initial_books - final_books : ℚ) / (1 - return_rate) = 125 :=
by sorry

end books_loaned_out_l3692_369280


namespace spider_sock_shoe_arrangements_l3692_369275

/-- The number of legs the spider has -/
def num_legs : ℕ := 10

/-- The total number of items (socks and shoes) -/
def total_items : ℕ := 2 * num_legs

/-- The number of valid arrangements for putting on socks and shoes -/
def valid_arrangements : ℕ := (Nat.factorial total_items) / (2^num_legs)

/-- Theorem stating the number of valid arrangements for the spider to put on socks and shoes -/
theorem spider_sock_shoe_arrangements :
  valid_arrangements = (Nat.factorial total_items) / (2^num_legs) :=
sorry

end spider_sock_shoe_arrangements_l3692_369275


namespace region_area_l3692_369261

/-- The area of a region bounded by three circular arcs -/
theorem region_area (r : ℝ) (θ : ℝ) : 
  r > 0 → 
  θ = π / 4 → 
  let sector_area := θ / (2 * π) * π * r^2
  let triangle_area := 1 / 2 * r^2 * Real.sin θ
  3 * (sector_area - triangle_area) = 24 * π - 48 * Real.sqrt 2 := by
  sorry

end region_area_l3692_369261


namespace number_composition_l3692_369240

/-- A number composed of hundreds, tens, ones, and hundredths -/
def compose_number (hundreds tens ones hundredths : ℕ) : ℚ :=
  (hundreds * 100 + tens * 10 + ones : ℚ) + (hundredths : ℚ) / 100

theorem number_composition :
  compose_number 3 4 6 8 = 346.08 := by
  sorry

end number_composition_l3692_369240


namespace division_remainder_l3692_369205

theorem division_remainder (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = divisor * quotient + remainder →
  dividend = 15 →
  divisor = 3 →
  quotient = 4 →
  remainder = 3 := by
  sorry

end division_remainder_l3692_369205


namespace arithmetic_sequence_sum_l3692_369208

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : isArithmeticSequence a) 
  (h_sum : a 2 + a 3 + a 7 = 6) : 
  a 1 + a 7 = 4 := by
  sorry

end arithmetic_sequence_sum_l3692_369208


namespace base_eight_sum_theorem_l3692_369255

/-- Converts a three-digit number in base 8 to its decimal representation -/
def baseEightToDecimal (a b c : ℕ) : ℕ := 64 * a + 8 * b + c

/-- Checks if a number is a valid non-zero digit in base 8 -/
def isValidBaseEightDigit (n : ℕ) : Prop := 0 < n ∧ n < 8

theorem base_eight_sum_theorem (A B C : ℕ) 
  (hA : isValidBaseEightDigit A) 
  (hB : isValidBaseEightDigit B) 
  (hC : isValidBaseEightDigit C) 
  (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (hSum : baseEightToDecimal A B C + baseEightToDecimal B C A + baseEightToDecimal C A B = baseEightToDecimal A A A) : 
  A + B + C = 8 := by
sorry

end base_eight_sum_theorem_l3692_369255


namespace square_area_with_four_circles_l3692_369256

/-- The area of a square containing four circles, each with a radius of 10 inches
    and touching two sides of the square and two other circles. -/
theorem square_area_with_four_circles (r : ℝ) (h : r = 10) : 
  let side_length := 4 * r
  (side_length ^ 2 : ℝ) = 1600 := by sorry

end square_area_with_four_circles_l3692_369256


namespace ral_to_suri_age_ratio_l3692_369264

def suri_future_age : ℕ := 16
def years_to_future : ℕ := 3
def ral_current_age : ℕ := 26

def suri_current_age : ℕ := suri_future_age - years_to_future

theorem ral_to_suri_age_ratio :
  ral_current_age / suri_current_age = 2 := by
  sorry

end ral_to_suri_age_ratio_l3692_369264


namespace opposite_face_of_B_is_H_l3692_369265

-- Define a cube type
structure Cube where
  faces : Fin 6 → Char

-- Define the set of valid labels
def ValidLabels : Set Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'}

-- Define a property that all faces have valid labels
def has_valid_labels (c : Cube) : Prop :=
  ∀ i : Fin 6, c.faces i ∈ ValidLabels

-- Define a property that all faces are unique
def has_unique_faces (c : Cube) : Prop :=
  ∀ i j : Fin 6, i ≠ j → c.faces i ≠ c.faces j

-- Define the theorem
theorem opposite_face_of_B_is_H (c : Cube) 
  (h1 : has_valid_labels c) 
  (h2 : has_unique_faces c) 
  (h3 : ∃ i : Fin 6, c.faces i = 'B') : 
  ∃ j : Fin 6, c.faces j = 'H' ∧ 
  (∀ k : Fin 6, c.faces k = 'B' → k.val + j.val = 5) :=
sorry

end opposite_face_of_B_is_H_l3692_369265


namespace sin_90_degrees_l3692_369206

theorem sin_90_degrees : Real.sin (π / 2) = 1 := by
  sorry

end sin_90_degrees_l3692_369206


namespace smallest_positive_integer_with_given_remainders_l3692_369253

theorem smallest_positive_integer_with_given_remainders : ∃! n : ℕ+, 
  (n : ℤ) % 5 = 4 ∧
  (n : ℤ) % 7 = 6 ∧
  (n : ℤ) % 9 = 8 ∧
  (n : ℤ) % 11 = 10 ∧
  ∀ m : ℕ+, 
    (m : ℤ) % 5 = 4 ∧
    (m : ℤ) % 7 = 6 ∧
    (m : ℤ) % 9 = 8 ∧
    (m : ℤ) % 11 = 10 →
    n ≤ m :=
by sorry

end smallest_positive_integer_with_given_remainders_l3692_369253


namespace min_value_function_l3692_369281

theorem min_value_function (a b : ℝ) (h : a + b = 1) :
  ∃ (min : ℝ), min = 5 * Real.sqrt 11 ∧
  ∀ (x y : ℝ), x + y = 1 →
  3 * Real.sqrt (1 + 2 * x^2) + 2 * Real.sqrt (40 + 9 * y^2) ≥ min :=
sorry

end min_value_function_l3692_369281


namespace job_completion_time_l3692_369295

theorem job_completion_time 
  (T : ℝ) -- Time for P to complete the job alone
  (h1 : T > 0) -- Ensure T is positive
  (h2 : 3 * (1/T + 1/20) + 0.4 * (1/T) = 1) -- Equation from working together and P finishing
  : T = 4 := by
  sorry

end job_completion_time_l3692_369295


namespace min_abs_z_plus_two_l3692_369218

open Complex

theorem min_abs_z_plus_two (z : ℂ) (h : (z * (1 + I)).im = 0) :
  ∃ (min : ℝ), min = Real.sqrt 2 ∧ ∀ (w : ℂ), (w * (1 + I)).im = 0 → Complex.abs (w + 2) ≥ min :=
sorry

end min_abs_z_plus_two_l3692_369218


namespace leading_coefficient_is_negative_eleven_l3692_369288

def polynomial (x : ℝ) : ℝ := 2 * (x^5 - 3*x^4 + 2*x^2) + 5 * (x^5 + x^4) - 6 * (3*x^5 + x^3 - x + 1)

theorem leading_coefficient_is_negative_eleven :
  ∃ (f : ℝ → ℝ), (∀ x, polynomial x = f x) ∧ 
  (∃ (a : ℝ) (g : ℝ → ℝ), a ≠ 0 ∧ (∀ x, f x = a * x^5 + g x) ∧ a = -11) :=
by sorry

end leading_coefficient_is_negative_eleven_l3692_369288


namespace square_difference_l3692_369249

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) :
  x^2 - y^2 = 80 := by sorry

end square_difference_l3692_369249


namespace power_division_rule_l3692_369251

theorem power_division_rule (a : ℝ) (ha : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end power_division_rule_l3692_369251


namespace unique_prime_perfect_power_l3692_369203

def is_perfect_power (x : ℕ) : Prop :=
  ∃ m n, m > 1 ∧ n ≥ 2 ∧ x = m^n

theorem unique_prime_perfect_power :
  ∀ p : ℕ, p ≤ 1000 → Prime p → is_perfect_power (2*p + 1) → p = 13 :=
by sorry

end unique_prime_perfect_power_l3692_369203


namespace f_properties_l3692_369262

def f (x : ℝ) : ℝ := x^3 - x

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x, x < -Real.sqrt 3 / 3 → (deriv f) x > 0) ∧
  (∀ x, x > Real.sqrt 3 / 3 → (deriv f) x > 0) :=
by sorry

end f_properties_l3692_369262


namespace mailman_theorem_l3692_369229

def mailman_problem (total_mail : ℕ) (mail_per_block : ℕ) : ℕ :=
  total_mail / mail_per_block

theorem mailman_theorem : 
  mailman_problem 192 48 = 4 := by
  sorry

end mailman_theorem_l3692_369229


namespace train_length_l3692_369236

/-- The length of a train given its speed, bridge length, and crossing time -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 * 1000 / 3600 →
  bridge_length = 255.03 →
  crossing_time = 30 →
  train_speed * crossing_time - bridge_length = 119.97 := by
  sorry

#check train_length

end train_length_l3692_369236


namespace negative_four_cubed_inequality_l3692_369269

theorem negative_four_cubed_inequality : (-4)^3 ≠ -4^3 := by
  -- Define the left-hand side
  have h1 : (-4)^3 = (-4) * (-4) * (-4) := by sorry
  -- Define the right-hand side
  have h2 : -4^3 = -(4 * 4 * 4) := by sorry
  -- Prove the inequality
  sorry

end negative_four_cubed_inequality_l3692_369269


namespace triangle_area_inequality_l3692_369241

-- Define a triangle as three points in a 2D plane
def Triangle (A B C : ℝ × ℝ) : Prop := True

-- Define a point on a line segment
def PointOnSegment (P A B : ℝ × ℝ) : Prop := 
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

-- Define the area of a triangle
noncomputable def TriangleArea (A B C : ℝ × ℝ) : ℝ := 
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_inequality (A B C P Q R : ℝ × ℝ) :
  Triangle A B C →
  PointOnSegment P B C →
  PointOnSegment Q C A →
  PointOnSegment R A B →
  min (TriangleArea A Q R) (min (TriangleArea B R P) (TriangleArea C P Q)) ≤ (1/4) * TriangleArea A B C :=
by sorry

end triangle_area_inequality_l3692_369241


namespace planes_parallel_to_line_not_necessarily_parallel_l3692_369283

-- Define a 3D space
variable (V : Type) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Fact (finrank ℝ V = 3)]

-- Define planes and lines in the space
variable (Plane : Type) (Line : Type)

-- Define a relation for a plane being parallel to a line
variable (plane_parallel_to_line : Plane → Line → Prop)

-- Define a relation for two planes being parallel
variable (planes_parallel : Plane → Plane → Prop)

-- Theorem: Two planes parallel to the same line are not necessarily parallel
theorem planes_parallel_to_line_not_necessarily_parallel 
  (P1 P2 : Plane) (L : Line) 
  (h1 : plane_parallel_to_line P1 L) 
  (h2 : plane_parallel_to_line P2 L) :
  ¬ (∀ P1 P2 : Plane, ∀ L : Line, 
    plane_parallel_to_line P1 L → plane_parallel_to_line P2 L → planes_parallel P1 P2) :=
sorry

end planes_parallel_to_line_not_necessarily_parallel_l3692_369283


namespace tanker_fill_time_l3692_369209

/-- Proves that two pipes with filling rates of 1/30 and 1/15 of a tanker per hour
    will fill the tanker in 10 hours when used simultaneously. -/
theorem tanker_fill_time (fill_time_A fill_time_B : ℝ) 
  (h_A : fill_time_A = 30) 
  (h_B : fill_time_B = 15) : 
  1 / (1 / fill_time_A + 1 / fill_time_B) = 10 := by
  sorry

end tanker_fill_time_l3692_369209


namespace algebraic_expression_value_l3692_369250

theorem algebraic_expression_value (x y : ℝ) : 
  x - 2*y + 1 = 3 → 2*x - 4*y + 1 = 5 := by
  sorry

end algebraic_expression_value_l3692_369250


namespace vasyas_number_l3692_369210

theorem vasyas_number (n : ℕ) 
  (h1 : 100 ≤ n ∧ n < 1000)
  (h2 : (n / 100) + (n % 10) = 1)
  (h3 : (n / 100) * ((n / 10) % 10) = 4) :
  n = 140 := by sorry

end vasyas_number_l3692_369210


namespace problem_statement_l3692_369292

theorem problem_statement (x y : ℝ) :
  |x + y - 6| + (x - y + 3)^2 = 0 → 3*x - y = 0 := by
sorry

end problem_statement_l3692_369292


namespace power_multiplication_zero_power_distribute_and_simplify_negative_power_and_division_l3692_369266

-- Define variables
variable (a m : ℝ)
variable (π : ℝ)

-- Theorem statements
theorem power_multiplication : a^2 * a^3 = a^5 := by sorry

theorem zero_power : (3.142 - π)^0 = 1 := by sorry

theorem distribute_and_simplify : 2*a*(a^2 - 1) = 2*a^3 - 2*a := by sorry

theorem negative_power_and_division : (-m^3)^2 / m^4 = m^2 := by sorry

end power_multiplication_zero_power_distribute_and_simplify_negative_power_and_division_l3692_369266


namespace tangent_slope_at_zero_l3692_369223

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem tangent_slope_at_zero : 
  (deriv f) 0 = 1 := by sorry

end tangent_slope_at_zero_l3692_369223


namespace rock_collecting_contest_l3692_369214

/-- The rock collecting contest between Sydney and Conner --/
theorem rock_collecting_contest 
  (sydney_start : ℕ) 
  (conner_start : ℕ) 
  (sydney_day1 : ℕ) 
  (conner_day1_multiplier : ℕ) 
  (sydney_day2 : ℕ) 
  (conner_day2 : ℕ) 
  (sydney_day3_multiplier : ℕ)
  (h1 : sydney_start = 837)
  (h2 : conner_start = 723)
  (h3 : sydney_day1 = 4)
  (h4 : conner_day1_multiplier = 8)
  (h5 : sydney_day2 = 0)
  (h6 : conner_day2 = 123)
  (h7 : sydney_day3_multiplier = 2) :
  ∃ conner_day3 : ℕ, 
    conner_day3 ≥ 27 ∧ 
    conner_start + conner_day1_multiplier * sydney_day1 + conner_day2 + conner_day3 ≥ 
    sydney_start + sydney_day1 + sydney_day2 + sydney_day3_multiplier * (conner_day1_multiplier * sydney_day1) :=
by sorry

end rock_collecting_contest_l3692_369214


namespace polynomial_expansion_l3692_369282

theorem polynomial_expansion (x : ℝ) : 
  (x^2 - 3*x + 3) * (x^2 + 3*x + 1) = x^4 - 5*x^2 + 6*x + 3 := by
  sorry

end polynomial_expansion_l3692_369282


namespace d_equals_four_l3692_369296

/-- A nine-digit number with specific properties -/
structure NineDigitNumber where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ
  E : ℕ
  F : ℕ
  G : ℕ
  first_three_sum : 6 + A + B = 13
  second_three_sum : A + B + C = 13
  third_three_sum : B + C + D = 13
  fourth_three_sum : C + D + E = 13
  fifth_three_sum : D + E + F = 13
  sixth_three_sum : E + F + G = 13
  last_three_sum : F + G + 3 = 13

/-- The digit D in the number must be 4 -/
theorem d_equals_four (n : NineDigitNumber) : n.D = 4 := by
  sorry

end d_equals_four_l3692_369296


namespace quadratic_factorization_problem_l3692_369217

theorem quadratic_factorization_problem :
  ∀ (a b : ℕ), 
    (∀ x : ℝ, x^2 - 20*x + 96 = (x - a)*(x - b)) →
    a > b →
    2*b - a = 4 := by
  sorry

end quadratic_factorization_problem_l3692_369217
