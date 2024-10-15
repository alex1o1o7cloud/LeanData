import Mathlib

namespace NUMINAMATH_CALUDE_min_distance_parallel_lines_l2487_248781

/-- The minimum distance between two parallel lines -/
theorem min_distance_parallel_lines :
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ x + 3 * y - 9 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y ↦ x + 3 * y + 1 = 0
  ∀ (P₁ : ℝ × ℝ) (P₂ : ℝ × ℝ),
  l₁ P₁.1 P₁.2 → l₂ P₂.1 P₂.2 →
  ∃ (P₁' : ℝ × ℝ) (P₂' : ℝ × ℝ),
  l₁ P₁'.1 P₁'.2 ∧ l₂ P₂'.1 P₂'.2 ∧
  Real.sqrt 10 = ‖(P₁'.1 - P₂'.1, P₁'.2 - P₂'.2)‖ ∧
  ∀ (Q₁ : ℝ × ℝ) (Q₂ : ℝ × ℝ),
  l₁ Q₁.1 Q₁.2 → l₂ Q₂.1 Q₂.2 →
  Real.sqrt 10 ≤ ‖(Q₁.1 - Q₂.1, Q₁.2 - Q₂.2)‖ :=
by
  sorry


end NUMINAMATH_CALUDE_min_distance_parallel_lines_l2487_248781


namespace NUMINAMATH_CALUDE_initial_sum_calculation_l2487_248709

/-- Proves that given a total amount of Rs. 15,500 after 4 years with a simple interest rate of 6% per annum, the initial sum of money (principal) is Rs. 12,500. -/
theorem initial_sum_calculation (total_amount : ℝ) (time : ℝ) (rate : ℝ) (principal : ℝ)
  (h1 : total_amount = 15500)
  (h2 : time = 4)
  (h3 : rate = 6)
  (h4 : total_amount = principal + (principal * rate * time / 100)) :
  principal = 12500 := by
sorry

end NUMINAMATH_CALUDE_initial_sum_calculation_l2487_248709


namespace NUMINAMATH_CALUDE_megan_zoo_pictures_l2487_248750

/-- The number of pictures Megan took at the zoo -/
def zoo_pictures : ℕ := sorry

/-- The number of pictures Megan took at the museum -/
def museum_pictures : ℕ := 18

/-- The number of pictures Megan deleted -/
def deleted_pictures : ℕ := 31

/-- The number of pictures Megan has left after deleting -/
def remaining_pictures : ℕ := 2

/-- Theorem stating that Megan took 15 pictures at the zoo -/
theorem megan_zoo_pictures : 
  zoo_pictures = 15 :=
by
  have h1 : zoo_pictures + museum_pictures - deleted_pictures = remaining_pictures := sorry
  sorry

end NUMINAMATH_CALUDE_megan_zoo_pictures_l2487_248750


namespace NUMINAMATH_CALUDE_probability_same_color_half_l2487_248798

/-- Represents a bag of colored balls -/
structure Bag where
  white : ℕ
  red : ℕ

/-- Calculates the probability of drawing balls of the same color from two bags -/
def probability_same_color (bag_a bag_b : Bag) : ℚ :=
  let total_a := bag_a.white + bag_a.red
  let total_b := bag_b.white + bag_b.red
  (bag_a.white * bag_b.white + bag_a.red * bag_b.red) / (total_a * total_b)

/-- The main theorem stating that the probability of drawing balls of the same color
    from the given bags is 1/2 -/
theorem probability_same_color_half :
  let bag_a : Bag := ⟨8, 4⟩
  let bag_b : Bag := ⟨6, 6⟩
  probability_same_color bag_a bag_b = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_probability_same_color_half_l2487_248798


namespace NUMINAMATH_CALUDE_distribution_schemes_count_l2487_248748

/-- The number of ways to distribute students from classes to districts -/
def distribute_students (num_classes : ℕ) (students_per_class : ℕ) (num_districts : ℕ) (students_per_district : ℕ) : ℕ :=
  -- Number of ways to choose 2 classes out of 4
  (num_classes.choose 2) *
  -- Number of ways to choose 2 districts out of 4
  (num_districts.choose 2) *
  -- Number of ways to choose 1 student from each of the remaining 2 classes
  (students_per_class.choose 1) * (students_per_class.choose 1) *
  -- Number of ways to assign these 2 students to the remaining 2 districts
  2

/-- Theorem stating that the number of distribution schemes is 288 -/
theorem distribution_schemes_count :
  distribute_students 4 2 4 2 = 288 := by
  sorry

end NUMINAMATH_CALUDE_distribution_schemes_count_l2487_248748


namespace NUMINAMATH_CALUDE_determinant_trigonometric_matrix_l2487_248770

theorem determinant_trigonometric_matrix (α β : Real) :
  let M : Matrix (Fin 3) (Fin 3) Real := ![
    ![Real.sin α * Real.sin β, Real.sin α * Real.cos β, Real.cos α],
    ![Real.cos β, -Real.sin β, 0],
    ![Real.cos α * Real.sin β, Real.cos α * Real.cos β, Real.sin α]
  ]
  Matrix.det M = Real.cos (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_determinant_trigonometric_matrix_l2487_248770


namespace NUMINAMATH_CALUDE_seventh_root_of_unity_product_l2487_248725

theorem seventh_root_of_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_of_unity_product_l2487_248725


namespace NUMINAMATH_CALUDE_fraction_zero_solution_l2487_248743

theorem fraction_zero_solution (x : ℝ) : 
  (x^2 - 4*x + 4) / (3*x - 9) = 0 ↔ x = 2 :=
by
  sorry

#check fraction_zero_solution

end NUMINAMATH_CALUDE_fraction_zero_solution_l2487_248743


namespace NUMINAMATH_CALUDE_evaluate_expression_l2487_248751

theorem evaluate_expression (a b : ℤ) (h1 : a = 5) (h2 : b = 3) :
  (a^2 + b)^2 - (a^2 - b)^2 = 300 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2487_248751


namespace NUMINAMATH_CALUDE_hedge_trimming_purpose_l2487_248722

/-- Represents the possible purposes of trimming hedges -/
inductive HedgeTrimPurpose
  | InhibitLateralBuds
  | PromoteLateralBuds
  | InhibitPhototropism
  | InhibitFloweringAndFruiting

/-- Represents the action of trimming hedges -/
structure HedgeTrimming where
  frequency : Nat  -- Represents how often the trimming occurs
  purpose : HedgeTrimPurpose

/-- Represents garden workers -/
structure GardenWorker where
  trims_hedges : Bool

/-- The theorem stating the purpose of hedge trimming -/
theorem hedge_trimming_purpose 
  (workers : List GardenWorker) 
  (trimming : HedgeTrimming) : 
  (∀ w ∈ workers, w.trims_hedges = true) → 
  (trimming.frequency > 0) → 
  (trimming.purpose = HedgeTrimPurpose.PromoteLateralBuds) :=
sorry

end NUMINAMATH_CALUDE_hedge_trimming_purpose_l2487_248722


namespace NUMINAMATH_CALUDE_prism_24_edges_has_10_faces_l2487_248767

/-- A prism is a polyhedron with two congruent parallel faces (bases) and rectangular lateral faces. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism given its number of edges -/
def num_faces (p : Prism) : ℕ :=
  let base_edges := p.edges / 3
  base_edges + 2

theorem prism_24_edges_has_10_faces (p : Prism) (h : p.edges = 24) : num_faces p = 10 := by
  sorry

end NUMINAMATH_CALUDE_prism_24_edges_has_10_faces_l2487_248767


namespace NUMINAMATH_CALUDE_solve_group_size_l2487_248793

def group_size_problem (n : ℕ) : Prop :=
  let weight_increase_per_person : ℚ := 5/2
  let weight_difference : ℕ := 20
  (weight_difference : ℚ) = n * weight_increase_per_person

theorem solve_group_size : ∃ n : ℕ, group_size_problem n ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_group_size_l2487_248793


namespace NUMINAMATH_CALUDE_second_number_is_255_l2487_248705

def first_set (x : ℝ) : List ℝ := [28, x, 42, 78, 104]
def second_set (x y : ℝ) : List ℝ := [128, y, 511, 1023, x]

theorem second_number_is_255 
  (x : ℝ)
  (h1 : (first_set x).sum / (first_set x).length = 90)
  (h2 : ∃ y, (second_set x y).sum / (second_set x y).length = 423) :
  ∃ y, (second_set x y).sum / (second_set x y).length = 423 ∧ y = 255 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_255_l2487_248705


namespace NUMINAMATH_CALUDE_ken_change_l2487_248702

/-- Represents the grocery purchase and payment scenario --/
def grocery_purchase (steak_price : ℕ) (steak_quantity : ℕ) (eggs_price : ℕ) 
  (milk_price : ℕ) (bagels_price : ℕ) (bill_20 : ℕ) (bill_10 : ℕ) 
  (bill_5 : ℕ) (coin_1 : ℕ) : Prop :=
  let total_cost := steak_price * steak_quantity + eggs_price + milk_price + bagels_price
  let total_paid := 20 * bill_20 + 10 * bill_10 + 5 * bill_5 + coin_1
  total_paid - total_cost = 16

/-- Theorem stating that Ken will receive $16 in change --/
theorem ken_change : grocery_purchase 7 2 3 4 6 1 1 2 3 := by
  sorry

end NUMINAMATH_CALUDE_ken_change_l2487_248702


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2487_248783

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, (2 * x^2 - x * (x - 4) = 5) ↔ (x^2 + 4*x - 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2487_248783


namespace NUMINAMATH_CALUDE_quadratic_properties_l2487_248732

/-- A quadratic function passing through specific points -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ) (h : a ≠ 0) :
  let f := QuadraticFunction a b c
  (f (-3) = 15) ∧ (f (-1) = 3) ∧ (f 0 = 0) ∧ (f 1 = -1) ∧ (f 2 = 0) ∧ (f 4 = 8) →
  (∀ x, f (1 + x) = f (1 - x)) ∧  -- Axis of symmetry at x = 1
  (f (-2) = 8) ∧ (f 3 = 3) ∧      -- Values at x = -2 and x = 3
  (f 0 = 0) ∧ (f 2 = 0)           -- Roots at x = 0 and x = 2
  := by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2487_248732


namespace NUMINAMATH_CALUDE_ann_found_blocks_l2487_248738

/-- Given that Ann initially had 9 blocks and ended up with 53 blocks,
    prove that she found 44 blocks. -/
theorem ann_found_blocks (initial_blocks : ℕ) (final_blocks : ℕ) :
  initial_blocks = 9 →
  final_blocks = 53 →
  final_blocks - initial_blocks = 44 := by
  sorry

end NUMINAMATH_CALUDE_ann_found_blocks_l2487_248738


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocal_sin_squared_l2487_248712

theorem min_value_sum_reciprocal_sin_squared (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C → -- Angles are positive
  A + B + C = π → -- Sum of angles in a triangle
  C = π / 2 → -- Right angle condition
  (∀ x y : ℝ, 0 < x → 0 < y → x + y + π/2 = π → 4 / (Real.sin x)^2 + 9 / (Real.sin y)^2 ≥ 25) ∧ 
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y + π/2 = π ∧ 4 / (Real.sin x)^2 + 9 / (Real.sin y)^2 = 25) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocal_sin_squared_l2487_248712


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2487_248707

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a d →
  d > 0 →
  a 1 + a 2 + a 3 = 15 →
  a 1 * a 2 * a 3 = 80 →
  a 11 + a 12 + a 13 = 105 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2487_248707


namespace NUMINAMATH_CALUDE_decagon_area_decagon_area_specific_l2487_248710

/-- The area of a decagon inscribed in a rectangle with specific properties. -/
theorem decagon_area (perimeter : ℝ) (length_ratio width_ratio : ℕ) : ℝ :=
  let length := (3 * perimeter) / (10 : ℝ)
  let width := (2 * perimeter) / (10 : ℝ)
  let rectangle_area := length * width
  let triangle_area_long := (1 / 2 : ℝ) * (length / 5) * (length / 5)
  let triangle_area_short := (1 / 2 : ℝ) * (width / 5) * (width / 5)
  let total_removed_area := 4 * triangle_area_long + 4 * triangle_area_short
  rectangle_area - total_removed_area

/-- 
  The area of a decagon inscribed in a rectangle is 1984 square centimeters, given:
  - The vertices of the decagon divide the sides of the rectangle into five equal parts
  - The perimeter of the rectangle is 200 centimeters
  - The ratio of length to width of the rectangle is 3:2
-/
theorem decagon_area_specific : decagon_area 200 3 2 = 1984 := by
  sorry

end NUMINAMATH_CALUDE_decagon_area_decagon_area_specific_l2487_248710


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2487_248715

theorem negation_of_proposition (f : ℝ → ℝ) :
  (¬ ∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0) ↔
  (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2487_248715


namespace NUMINAMATH_CALUDE_finite_nonempty_set_is_good_l2487_248733

/-- An expression using real numbers, ±, +, ×, and parentheses -/
inductive Expression : Type
| Const : ℝ → Expression
| PlusMinus : Expression → Expression → Expression
| Plus : Expression → Expression → Expression
| Times : Expression → Expression → Expression

/-- The range of an expression -/
def range (e : Expression) : Set ℝ :=
  sorry

/-- A set is good if it's the range of some expression -/
def is_good (S : Set ℝ) : Prop :=
  ∃ e : Expression, range e = S

theorem finite_nonempty_set_is_good (S : Set ℝ) (h₁ : S.Finite) (h₂ : S.Nonempty) :
  is_good S :=
sorry

end NUMINAMATH_CALUDE_finite_nonempty_set_is_good_l2487_248733


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_13_l2487_248740

theorem consecutive_integers_around_sqrt_13 (a b : ℤ) :
  (b = a + 1) → (a < Real.sqrt 13) → (Real.sqrt 13 < b) → (a + b = 7) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_13_l2487_248740


namespace NUMINAMATH_CALUDE_unoccupied_business_seats_count_l2487_248788

/-- Represents the seating configuration and occupancy of an airplane. -/
structure AirplaneSeating where
  firstClassSeats : ℕ
  businessClassSeats : ℕ
  economyClassSeats : ℕ
  firstClassOccupied : ℕ
  economyClassOccupied : ℕ
  businessAndFirstOccupied : ℕ

/-- Calculates the number of unoccupied seats in business class. -/
def unoccupiedBusinessSeats (a : AirplaneSeating) : ℕ :=
  a.businessClassSeats - (a.businessAndFirstOccupied - a.firstClassOccupied)

/-- Theorem stating the number of unoccupied seats in business class. -/
theorem unoccupied_business_seats_count
  (a : AirplaneSeating)
  (h1 : a.firstClassSeats = 10)
  (h2 : a.businessClassSeats = 30)
  (h3 : a.economyClassSeats = 50)
  (h4 : a.economyClassOccupied = a.economyClassSeats / 2)
  (h5 : a.businessAndFirstOccupied = a.economyClassOccupied)
  (h6 : a.firstClassOccupied = 3) :
  unoccupiedBusinessSeats a = 8 := by
  sorry

#eval unoccupiedBusinessSeats {
  firstClassSeats := 10,
  businessClassSeats := 30,
  economyClassSeats := 50,
  firstClassOccupied := 3,
  economyClassOccupied := 25,
  businessAndFirstOccupied := 25
}

end NUMINAMATH_CALUDE_unoccupied_business_seats_count_l2487_248788


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l2487_248746

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a^3 + b^3 = 100) : 
  a * b = -3 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l2487_248746


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2487_248799

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (x^2 + 16) = 12 ↔ x = 8 * Real.sqrt 2 ∨ x = -8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2487_248799


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l2487_248734

/-- Proves that the amount of fuel A added to the tank is 122 gallons -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ)
  (h1 : tank_capacity = 218)
  (h2 : ethanol_a = 0.12)
  (h3 : ethanol_b = 0.16)
  (h4 : total_ethanol = 30) :
  ∃ (fuel_a : ℝ), fuel_a = 122 ∧ 
    ethanol_a * fuel_a + ethanol_b * (tank_capacity - fuel_a) = total_ethanol :=
by sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l2487_248734


namespace NUMINAMATH_CALUDE_garden_length_proof_l2487_248776

theorem garden_length_proof (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = 2 + 3 * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 100 →
  length = 38 := by
sorry

end NUMINAMATH_CALUDE_garden_length_proof_l2487_248776


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l2487_248775

theorem parallelogram_side_length 
  (s : ℝ) 
  (area : ℝ) 
  (angle : ℝ) :
  s > 0 → 
  angle = π / 6 → 
  area = 27 * Real.sqrt 3 → 
  3 * s * s * Real.sqrt 3 = area → 
  s = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l2487_248775


namespace NUMINAMATH_CALUDE_B_equals_one_four_l2487_248706

def A : Set ℝ := {0, 1, 2, 3}

def B (m : ℝ) : Set ℝ := {x | x^2 - 5*x + m = 0}

theorem B_equals_one_four (m : ℝ) : 
  (A ∩ B m = {1}) → B m = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_B_equals_one_four_l2487_248706


namespace NUMINAMATH_CALUDE_polynomial_identity_l2487_248730

theorem polynomial_identity : 
  ∀ (a₀ a₁ a₂ a₃ a₄ : ℝ),
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2487_248730


namespace NUMINAMATH_CALUDE_symmetric_points_range_l2487_248719

noncomputable section

open Real

def e : ℝ := Real.exp 1

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x

def g (x : ℝ) : ℝ := exp x

def h (x : ℝ) : ℝ := log x

theorem symmetric_points_range (a : ℝ) :
  (∃ x y : ℝ, 1/e ≤ x ∧ x ≤ e ∧ 1/e ≤ y ∧ y ≤ e ∧
    f a x = g y ∧ f a y = g x) →
  1 ≤ a ∧ a ≤ e + 1/e :=
by sorry

end

end NUMINAMATH_CALUDE_symmetric_points_range_l2487_248719


namespace NUMINAMATH_CALUDE_perp_line_plane_condition_l2487_248777

/-- A straight line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Defines the perpendicular relationship between a line and another line -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Defines the perpendicular relationship between a line and a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines when a line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- The main theorem stating that "m ⊥ n" is a necessary but not sufficient condition for "m ⊥ α" -/
theorem perp_line_plane_condition (m n : Line3D) (α : Plane3D) 
  (h : line_in_plane n α) :
  (perpendicular_line_plane m α → perpendicular_lines m n) ∧
  ¬(perpendicular_lines m n → perpendicular_line_plane m α) :=
sorry

end NUMINAMATH_CALUDE_perp_line_plane_condition_l2487_248777


namespace NUMINAMATH_CALUDE_nine_hundred_in_column_B_l2487_248739

/-- The column type representing the six columns A, B, C, D, E, F -/
inductive Column
| A | B | C | D | E | F

/-- The function that determines the column for a given positive integer -/
def column_for_number (n : ℕ) : Column :=
  match (n - 3) % 12 with
  | 0 => Column.A
  | 1 => Column.B
  | 2 => Column.C
  | 3 => Column.D
  | 4 => Column.A
  | 5 => Column.F
  | 6 => Column.E
  | 7 => Column.F
  | 8 => Column.D
  | 9 => Column.C
  | 10 => Column.B
  | 11 => Column.A
  | _ => Column.A  -- This case should never occur

theorem nine_hundred_in_column_B :
  column_for_number 900 = Column.B :=
by sorry

end NUMINAMATH_CALUDE_nine_hundred_in_column_B_l2487_248739


namespace NUMINAMATH_CALUDE_sterling_candy_proof_l2487_248700

/-- The number of candy pieces earned for a correct answer -/
def correct_reward : ℕ := 3

/-- The total number of questions answered -/
def total_questions : ℕ := 7

/-- The number of questions answered correctly -/
def correct_answers : ℕ := 7

/-- The number of additional correct answers -/
def additional_correct : ℕ := 2

/-- The total number of candy pieces earned if Sterling answered 2 more questions correctly -/
def total_candy : ℕ := correct_reward * (correct_answers + additional_correct)

theorem sterling_candy_proof :
  total_candy = 27 :=
sorry

end NUMINAMATH_CALUDE_sterling_candy_proof_l2487_248700


namespace NUMINAMATH_CALUDE_joan_seashells_l2487_248794

theorem joan_seashells (seashells_left seashells_given_to_sam : ℕ) 
  (h1 : seashells_left = 27) 
  (h2 : seashells_given_to_sam = 43) : 
  seashells_left + seashells_given_to_sam = 70 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_l2487_248794


namespace NUMINAMATH_CALUDE_group_size_l2487_248735

theorem group_size (total_paise : ℕ) (h : total_paise = 7744) : 
  ∃ n : ℕ, n * n = total_paise ∧ n = 88 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l2487_248735


namespace NUMINAMATH_CALUDE_i_minus_one_squared_l2487_248728

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem i_minus_one_squared : (i - 1)^2 = -2*i := by
  sorry

end NUMINAMATH_CALUDE_i_minus_one_squared_l2487_248728


namespace NUMINAMATH_CALUDE_smallest_bottom_right_corner_l2487_248755

/-- Represents a 3x3 grid of natural numbers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Checks if all numbers in the grid are different -/
def all_different (g : Grid) : Prop :=
  ∀ i j k l, g i j = g k l → (i = k ∧ j = l)

/-- Checks if the sum condition is satisfied for rows -/
def row_sum_condition (g : Grid) : Prop :=
  ∀ i, g i 0 + g i 1 = g i 2

/-- Checks if the sum condition is satisfied for columns -/
def col_sum_condition (g : Grid) : Prop :=
  ∀ j, g 0 j + g 1 j = g 2 j

/-- The main theorem stating the smallest possible value for the bottom right corner -/
theorem smallest_bottom_right_corner (g : Grid) 
  (h1 : all_different g) 
  (h2 : row_sum_condition g) 
  (h3 : col_sum_condition g) : 
  g 2 2 ≥ 12 := by
  sorry


end NUMINAMATH_CALUDE_smallest_bottom_right_corner_l2487_248755


namespace NUMINAMATH_CALUDE_parrot_days_theorem_l2487_248764

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of phrases the parrot currently knows -/
def current_phrases : ℕ := 17

/-- The number of phrases Georgina teaches the parrot per week -/
def phrases_per_week : ℕ := 2

/-- The number of phrases the parrot knew when Georgina bought it -/
def initial_phrases : ℕ := 3

/-- The number of days Georgina has had the parrot -/
def days_with_parrot : ℕ := 49

theorem parrot_days_theorem :
  (current_phrases - initial_phrases) / phrases_per_week * days_per_week = days_with_parrot := by
  sorry

end NUMINAMATH_CALUDE_parrot_days_theorem_l2487_248764


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_m_range_l2487_248736

theorem complex_in_second_quadrant_m_range (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 - 2) (m - 1)
  (z.re < 0 ∧ z.im > 0) → (1 < m ∧ m < Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_m_range_l2487_248736


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l2487_248708

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 4 * x + 2 * y = 20) 
  (eq2 : 2 * x + 4 * y = 16) : 
  4 * x^2 + 12 * x * y + 12 * y^2 = 292 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l2487_248708


namespace NUMINAMATH_CALUDE_min_value_of_function_l2487_248741

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  9 * x + 1 / (x^3) ≥ 10 ∧ ∃ y > 0, 9 * y + 1 / (y^3) = 10 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2487_248741


namespace NUMINAMATH_CALUDE_square_difference_theorem_l2487_248726

theorem square_difference_theorem (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (diff_eq : x - y = 4) : 
  x^2 - y^2 = 32 := by
sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l2487_248726


namespace NUMINAMATH_CALUDE_exists_pentagon_with_similar_subpentagon_l2487_248754

/-- A convex pentagon with specific angles and side lengths -/
structure ConvexPentagon where
  -- Sides of the pentagon
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  -- Two angles of the pentagon (in radians)
  angle1 : ℝ
  angle2 : ℝ
  -- Convexity condition
  convex : angle1 > 0 ∧ angle2 > 0 ∧ angle1 < π ∧ angle2 < π

/-- Similarity between two pentagons -/
def isSimilar (p1 p2 : ConvexPentagon) : Prop :=
  ∃ k : ℝ, k > 0 ∧
    p2.side1 = k * p1.side1 ∧
    p2.side2 = k * p1.side2 ∧
    p2.side3 = k * p1.side3 ∧
    p2.side4 = k * p1.side4 ∧
    p2.side5 = k * p1.side5 ∧
    p2.angle1 = p1.angle1 ∧
    p2.angle2 = p1.angle2

/-- Theorem stating the existence of a specific convex pentagon with a similar sub-pentagon -/
theorem exists_pentagon_with_similar_subpentagon :
  ∃ (p : ConvexPentagon) (q : ConvexPentagon),
    p.side1 = 2 ∧ p.side2 = 4 ∧ p.side3 = 8 ∧ p.side4 = 6 ∧ p.side5 = 12 ∧
    p.angle1 = π / 3 ∧ p.angle2 = 2 * π / 3 ∧
    isSimilar p q :=
sorry

end NUMINAMATH_CALUDE_exists_pentagon_with_similar_subpentagon_l2487_248754


namespace NUMINAMATH_CALUDE_T_is_perfect_square_T_equals_fib_squared_l2487_248779

/-- A tetromino tile is formed by gluing together four unit square tiles, edge to edge. -/
def TetrominoTile : Type := Unit

/-- Tₙ is the number of ways to tile a 2×2n rectangular bathroom floor with tetromino tiles. -/
def T (n : ℕ) : ℕ := sorry

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- The main theorem: Tₙ is always a perfect square, specifically Fₙ₊₁² -/
theorem T_is_perfect_square (n : ℕ) : ∃ k : ℕ, T n = k ^ 2 :=
  sorry

/-- The specific form of Tₙ in terms of Fibonacci numbers -/
theorem T_equals_fib_squared (n : ℕ) : T n = (fib (n + 1)) ^ 2 :=
  sorry

end NUMINAMATH_CALUDE_T_is_perfect_square_T_equals_fib_squared_l2487_248779


namespace NUMINAMATH_CALUDE_mans_speed_in_still_water_l2487_248763

/-- The speed of a man rowing in still water, given his downstream performance and the current speed. -/
theorem mans_speed_in_still_water (current_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  current_speed = 3 →
  distance = 15 / 1000 →
  time = 2.9997600191984644 / 3600 →
  (distance / time) - current_speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_mans_speed_in_still_water_l2487_248763


namespace NUMINAMATH_CALUDE_train_crossing_time_l2487_248782

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (pole_crossing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 300)
  (h2 : pole_crossing_time = 18)
  (h3 : platform_length = 200) :
  (train_length + platform_length) / (train_length / pole_crossing_time) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2487_248782


namespace NUMINAMATH_CALUDE_cauliflower_area_l2487_248766

theorem cauliflower_area (this_year_side : ℕ) (last_year_side : ℕ) 
  (h1 : this_year_side ^ 2 = 12544)
  (h2 : this_year_side ^ 2 = last_year_side ^ 2 + 223) :
  1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cauliflower_area_l2487_248766


namespace NUMINAMATH_CALUDE_jessica_total_cost_l2487_248787

def cat_toy_cost : ℚ := 10.22
def cage_cost : ℚ := 11.73
def cat_food_cost : ℚ := 7.50
def leash_cost : ℚ := 5.15
def cat_treats_cost : ℚ := 3.98

theorem jessica_total_cost :
  cat_toy_cost + cage_cost + cat_food_cost + leash_cost + cat_treats_cost = 38.58 := by
  sorry

end NUMINAMATH_CALUDE_jessica_total_cost_l2487_248787


namespace NUMINAMATH_CALUDE_largest_fraction_l2487_248744

theorem largest_fraction : 
  let fractions : List ℚ := [2/5, 1/3, 5/15, 4/10, 7/21]
  ∀ x ∈ fractions, x ≤ 2/5 ∧ x ≤ 4/10 :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_l2487_248744


namespace NUMINAMATH_CALUDE_l_shaped_floor_paving_cost_l2487_248701

/-- Calculates the total cost of paving an L-shaped floor with two types of slabs -/
def total_paving_cost (large_length large_width small_length small_width type_a_cost type_b_cost : ℝ) : ℝ :=
  let large_area := large_length * large_width
  let small_area := small_length * small_width
  let large_cost := large_area * type_a_cost
  let small_cost := small_area * type_b_cost
  large_cost + small_cost

/-- Theorem stating that the total cost of paving the L-shaped floor is Rs. 13,781.25 -/
theorem l_shaped_floor_paving_cost :
  total_paving_cost 5.5 3.75 2.5 1.25 600 450 = 13781.25 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_floor_paving_cost_l2487_248701


namespace NUMINAMATH_CALUDE_little_twelve_conference_games_l2487_248795

/-- Calculates the number of games in a football conference with specified rules -/
def conference_games (num_divisions : ℕ) (teams_per_division : ℕ) (intra_div_games : ℕ) (inter_div_games : ℕ) : ℕ :=
  let intra_division_games := num_divisions * (teams_per_division * (teams_per_division - 1) / 2) * intra_div_games
  let inter_division_games := (num_divisions * teams_per_division) * (teams_per_division * (num_divisions - 1)) * inter_div_games / 2
  intra_division_games + inter_division_games

/-- The Little Twelve Football Conference scheduling theorem -/
theorem little_twelve_conference_games :
  conference_games 2 6 3 2 = 162 := by
  sorry

end NUMINAMATH_CALUDE_little_twelve_conference_games_l2487_248795


namespace NUMINAMATH_CALUDE_mutual_fund_investment_l2487_248713

theorem mutual_fund_investment
  (total_investment : ℝ)
  (mutual_fund_ratio : ℝ)
  (h1 : total_investment = 250000)
  (h2 : mutual_fund_ratio = 3) :
  let commodity_investment := total_investment / (1 + mutual_fund_ratio)
  let mutual_fund_investment := mutual_fund_ratio * commodity_investment
  mutual_fund_investment = 187500 := by
sorry

end NUMINAMATH_CALUDE_mutual_fund_investment_l2487_248713


namespace NUMINAMATH_CALUDE_water_needed_for_lemonade_l2487_248785

/-- Given a ratio of water to lemon juice and a total amount of lemonade to make,
    calculate the amount of water needed in quarts. -/
theorem water_needed_for_lemonade 
  (water_ratio : ℚ)
  (lemon_juice_ratio : ℚ)
  (total_gallons : ℚ)
  (quarts_per_gallon : ℚ) :
  water_ratio = 8 →
  lemon_juice_ratio = 1 →
  total_gallons = 3/2 →
  quarts_per_gallon = 4 →
  (water_ratio * total_gallons * quarts_per_gallon) / (water_ratio + lemon_juice_ratio) = 16/3 :=
by
  sorry

end NUMINAMATH_CALUDE_water_needed_for_lemonade_l2487_248785


namespace NUMINAMATH_CALUDE_arc_length_of_sector_l2487_248745

/-- The arc length of a sector with radius π cm and central angle 120° is 2π²/3 cm. -/
theorem arc_length_of_sector (r : Real) (θ : Real) : 
  r = π → θ = 120 * π / 180 → r * θ = 2 * π^2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_of_sector_l2487_248745


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l2487_248757

def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_implies_a_equals_one (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_one_l2487_248757


namespace NUMINAMATH_CALUDE_sin_two_alpha_value_l2487_248773

theorem sin_two_alpha_value (α : Real) (h : Real.sin α + Real.cos α = 2/3) : 
  Real.sin (2 * α) = -5/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_alpha_value_l2487_248773


namespace NUMINAMATH_CALUDE_no_interior_points_with_sum_20_l2487_248789

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance squared between two points -/
def distSquared (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- A circle with center at the origin and radius 2 -/
def insideCircle (p : Point) : Prop :=
  p.x^2 + p.y^2 < 4

theorem no_interior_points_with_sum_20 :
  ¬ ∃ (p : Point), insideCircle p ∧
    ∃ (a b : Point), 
      a.x^2 + a.y^2 = 4 ∧ 
      b.x^2 + b.y^2 = 4 ∧ 
      a.x = -b.x ∧ 
      a.y = -b.y ∧
      distSquared p a + distSquared p b = 20 :=
by sorry

end NUMINAMATH_CALUDE_no_interior_points_with_sum_20_l2487_248789


namespace NUMINAMATH_CALUDE_geometric_with_arithmetic_subsequence_l2487_248760

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

/-- An arithmetic subsequence of a sequence -/
def arithmetic_subsequence (a : ℕ → ℝ) (sub : ℕ → ℕ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (sub (n + 1)) - a (sub n) = d

/-- The main theorem: if a geometric sequence has an infinite arithmetic subsequence,
    then its common ratio is -1 -/
theorem geometric_with_arithmetic_subsequence
  (a : ℕ → ℝ) (q : ℝ) (sub : ℕ → ℕ) (d : ℝ) (h_ne_one : q ≠ 1) :
  geometric_sequence a q →
  (∃ d, arithmetic_subsequence a sub d) →
  q = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_with_arithmetic_subsequence_l2487_248760


namespace NUMINAMATH_CALUDE_odd_number_of_odd_sided_faces_l2487_248737

-- Define a convex polyhedron
structure ConvexPolyhedron where
  vertices : ℕ
  convex : Bool

-- Define a closed broken line on the polyhedron
structure ClosedBrokenLine where
  polyhedron : ConvexPolyhedron
  passes_all_vertices_once : Bool

-- Define a part of the polyhedron surface
structure SurfacePart where
  polyhedron : ConvexPolyhedron
  broken_line : ClosedBrokenLine
  faces : Finset (Finset ℕ)  -- Each face is represented as a set of its vertices

-- Function to count odd-sided faces in a surface part
def count_odd_sided_faces (part : SurfacePart) : ℕ :=
  (part.faces.filter (λ face => face.card % 2 = 1)).card

-- The main theorem
theorem odd_number_of_odd_sided_faces 
  (poly : ConvexPolyhedron) 
  (line : ClosedBrokenLine) 
  (part : SurfacePart) : 
  poly.vertices = 2003 → 
  poly.convex = true → 
  line.polyhedron = poly → 
  line.passes_all_vertices_once = true → 
  part.polyhedron = poly → 
  part.broken_line = line → 
  count_odd_sided_faces part % 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_odd_number_of_odd_sided_faces_l2487_248737


namespace NUMINAMATH_CALUDE_solution_when_a_neg3_m_0_range_of_a_for_real_roots_range_of_m_when_a_0_l2487_248720

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + a + 3
def g (m : ℝ) (x : ℝ) : ℝ := m*x + 5 - 2*m

-- Question 1
theorem solution_when_a_neg3_m_0 :
  {x : ℝ | f (-3) x - g 0 x = 0} = {-1, 5} := by sorry

-- Question 2
theorem range_of_a_for_real_roots :
  {a : ℝ | ∃ x ∈ Set.Icc (-1) 1, f a x = 0} = Set.Icc (-8) 0 := by sorry

-- Question 3
theorem range_of_m_when_a_0 :
  {m : ℝ | ∀ x₁ ∈ Set.Icc 1 4, ∃ x₂ ∈ Set.Icc 1 4, f 0 x₁ = g m x₂} =
  Set.Iic (-3) ∪ Set.Ici 6 := by sorry

end NUMINAMATH_CALUDE_solution_when_a_neg3_m_0_range_of_a_for_real_roots_range_of_m_when_a_0_l2487_248720


namespace NUMINAMATH_CALUDE_consecutive_squares_sum_diff_odd_l2487_248772

theorem consecutive_squares_sum_diff_odd (n : ℕ) : 
  Odd (n^2 + (n+1)^2) ∧ Odd ((n+1)^2 - n^2) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_sum_diff_odd_l2487_248772


namespace NUMINAMATH_CALUDE_baseball_audience_percentage_l2487_248723

theorem baseball_audience_percentage (total : ℕ) (second_team_percentage : ℚ) (non_supporters : ℕ) :
  total = 50 →
  second_team_percentage = 34 / 100 →
  non_supporters = 3 →
  (total - (total * second_team_percentage).floor - non_supporters : ℚ) / total = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_baseball_audience_percentage_l2487_248723


namespace NUMINAMATH_CALUDE_percentage_boys_playing_soccer_l2487_248761

/-- Proves that the percentage of boys among students playing soccer is 86% -/
theorem percentage_boys_playing_soccer
  (total_students : ℕ)
  (num_boys : ℕ)
  (num_playing_soccer : ℕ)
  (num_girls_not_playing : ℕ)
  (h1 : total_students = 450)
  (h2 : num_boys = 320)
  (h3 : num_playing_soccer = 250)
  (h4 : num_girls_not_playing = 95)
  : (((num_playing_soccer - (total_students - num_boys - num_girls_not_playing)) / num_playing_soccer) : ℚ) = 86 / 100 := by
  sorry


end NUMINAMATH_CALUDE_percentage_boys_playing_soccer_l2487_248761


namespace NUMINAMATH_CALUDE_abc_product_l2487_248796

theorem abc_product (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 24 * Real.rpow 3 (1/3))
  (hac : a * c = 40 * Real.rpow 3 (1/3))
  (hbc : b * c = 18 * Real.rpow 3 (1/3)) :
  a * b * c = 432 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_abc_product_l2487_248796


namespace NUMINAMATH_CALUDE_seventy_million_scientific_notation_l2487_248780

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem seventy_million_scientific_notation :
  toScientificNotation 70000000 = ScientificNotation.mk 7.0 7 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_seventy_million_scientific_notation_l2487_248780


namespace NUMINAMATH_CALUDE_ninth_term_is_twelve_l2487_248714

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_condition : a 5 + a 7 = 16
  third_term : a 3 = 4

/-- The 9th term of the arithmetic sequence is 12 -/
theorem ninth_term_is_twelve (seq : ArithmeticSequence) : seq.a 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_twelve_l2487_248714


namespace NUMINAMATH_CALUDE_books_to_read_l2487_248797

theorem books_to_read (total : ℕ) (mcgregor : ℕ) (floyd : ℕ) : 
  total = 89 → mcgregor = 34 → floyd = 32 → total - (mcgregor + floyd) = 23 := by
  sorry

end NUMINAMATH_CALUDE_books_to_read_l2487_248797


namespace NUMINAMATH_CALUDE_difference_repetition_l2487_248778

theorem difference_repetition (a : Fin 20 → ℕ) 
  (h_order : ∀ i j, i < j → a i < a j) 
  (h_bound : a 19 ≤ 70) : 
  ∃ (j₁ k₁ j₂ k₂ j₃ k₃ j₄ k₄ : Fin 20), 
    k₁ < j₁ ∧ k₂ < j₂ ∧ k₃ < j₃ ∧ k₄ < j₄ ∧
    (a j₁ - a k₁ : ℤ) = (a j₂ - a k₂) ∧
    (a j₁ - a k₁ : ℤ) = (a j₃ - a k₃) ∧
    (a j₁ - a k₁ : ℤ) = (a j₄ - a k₄) :=
by sorry

end NUMINAMATH_CALUDE_difference_repetition_l2487_248778


namespace NUMINAMATH_CALUDE_subset_of_any_set_implies_zero_l2487_248716

theorem subset_of_any_set_implies_zero (a : ℝ) : 
  (∀ S : Set ℝ, {x : ℝ | a * x = 1} ⊆ S) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_subset_of_any_set_implies_zero_l2487_248716


namespace NUMINAMATH_CALUDE_abs_neg_two_eq_two_l2487_248711

theorem abs_neg_two_eq_two : |(-2 : ℤ)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_eq_two_l2487_248711


namespace NUMINAMATH_CALUDE_f_g_minus_g_f_l2487_248769

def f (x : ℝ) : ℝ := 4 * x + 8

def g (x : ℝ) : ℝ := 2 * x - 3

theorem f_g_minus_g_f : ∀ x : ℝ, f (g x) - g (f x) = -17 := by
  sorry

end NUMINAMATH_CALUDE_f_g_minus_g_f_l2487_248769


namespace NUMINAMATH_CALUDE_inverse_function_solution_l2487_248765

/-- Given a function f(x) = 1 / (ax^2 + bx + c), where a, b, and c are nonzero real constants,
    the solutions to f^(-1)(x) = 1 are x = (-b ± √(b^2 - 4a(c-1))) / (2a) -/
theorem inverse_function_solution (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let f := fun x => 1 / (a * x^2 + b * x + c)
  let sol₁ := (-b + Real.sqrt (b^2 - 4*a*(c-1))) / (2*a)
  let sol₂ := (-b - Real.sqrt (b^2 - 4*a*(c-1))) / (2*a)
  (∀ x, f x = 1 ↔ x = sol₁ ∨ x = sol₂) :=
by sorry

end NUMINAMATH_CALUDE_inverse_function_solution_l2487_248765


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2487_248752

/-- 
For a quadratic equation qx^2 - 20x + 9 = 0 to have exactly one solution,
q must equal 100/9.
-/
theorem unique_solution_quadratic : 
  ∃! q : ℚ, q ≠ 0 ∧ (∃! x : ℝ, q * x^2 - 20 * x + 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2487_248752


namespace NUMINAMATH_CALUDE_f_has_three_distinct_roots_l2487_248771

/-- The polynomial function whose roots we want to count -/
def f (x : ℝ) : ℝ := (x + 5) * (x^2 + 5*x - 6)

/-- The statement that f has exactly 3 distinct real roots -/
theorem f_has_three_distinct_roots : ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) :=
sorry

end NUMINAMATH_CALUDE_f_has_three_distinct_roots_l2487_248771


namespace NUMINAMATH_CALUDE_lattice_point_decomposition_l2487_248768

/-- Represents a point in a 2D lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Represents a parallelogram OABC where O is the origin -/
structure Parallelogram where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Checks if a point is in or on a triangle -/
def inTriangle (P Q R S : LatticePoint) : Prop := sorry

/-- Vector addition -/
def vecAdd (P Q : LatticePoint) : LatticePoint := sorry

theorem lattice_point_decomposition 
  (OABC : Parallelogram) 
  (P : LatticePoint) 
  (h : inTriangle P OABC.A OABC.B OABC.C) :
  ∃ (Q R : LatticePoint), 
    inTriangle Q (LatticePoint.mk 0 0) OABC.A OABC.C ∧ 
    inTriangle R (LatticePoint.mk 0 0) OABC.A OABC.C ∧
    P = vecAdd Q R := by sorry

end NUMINAMATH_CALUDE_lattice_point_decomposition_l2487_248768


namespace NUMINAMATH_CALUDE_principal_arg_range_l2487_248747

open Complex

theorem principal_arg_range (z ω : ℂ) 
  (h1 : abs (z - I) = 1)
  (h2 : z ≠ 0)
  (h3 : z ≠ 2 * I)
  (h4 : ∃ (r : ℝ), (ω - 2 * I) / ω * z / (z - 2 * I) = r) :
  ∃ (θ : ℝ), θ ∈ (Set.Ioo 0 π ∪ Set.Ioo π (2 * π)) ∧ arg (ω - 2) = θ :=
sorry

end NUMINAMATH_CALUDE_principal_arg_range_l2487_248747


namespace NUMINAMATH_CALUDE_symmetric_point_correct_l2487_248756

/-- Given a point A and a line l, find the point B symmetric to A about l -/
def symmetricPoint (A : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ × ℝ := sorry

/-- The line x - y - 1 = 0 -/
def line (p : ℝ × ℝ) : Prop :=
  p.1 - p.2 - 1 = 0

theorem symmetric_point_correct :
  let A : ℝ × ℝ := (-1, 1)
  let B : ℝ × ℝ := (2, -2)
  symmetricPoint A line = B := by sorry

end NUMINAMATH_CALUDE_symmetric_point_correct_l2487_248756


namespace NUMINAMATH_CALUDE_parabola_equation_correct_l2487_248762

/-- A parabola with vertex (h, k) and passing through point (x₀, y₀) -/
structure Parabola where
  h : ℝ  -- x-coordinate of vertex
  k : ℝ  -- y-coordinate of vertex
  x₀ : ℝ  -- x-coordinate of point on parabola
  y₀ : ℝ  -- y-coordinate of point on parabola

/-- The equation of a parabola in the form ax^2 + bx + c -/
structure ParabolaEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: The given parabola equation represents the specified parabola -/
theorem parabola_equation_correct (p : Parabola) (eq : ParabolaEquation) : 
  p.h = 3 ∧ p.k = 5 ∧ p.x₀ = 2 ∧ p.y₀ = 2 ∧
  eq.a = -3 ∧ eq.b = 18 ∧ eq.c = -22 →
  ∀ x y : ℝ, y = eq.a * x^2 + eq.b * x + eq.c ↔ 
    (x = p.h ∧ y = p.k) ∨ 
    (y = eq.a * (x - p.h)^2 + p.k) ∨
    (x = p.x₀ ∧ y = p.y₀) := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_correct_l2487_248762


namespace NUMINAMATH_CALUDE_winning_percentage_l2487_248792

/-- Given an election with 6000 total votes and a winning margin of 1200 votes,
    prove that the winning candidate received 60% of the votes. -/
theorem winning_percentage (total_votes : ℕ) (winning_margin : ℕ) (winning_percentage : ℚ) :
  total_votes = 6000 →
  winning_margin = 1200 →
  winning_percentage = 60 / 100 →
  winning_percentage * total_votes = (total_votes + winning_margin) / 2 :=
by sorry

end NUMINAMATH_CALUDE_winning_percentage_l2487_248792


namespace NUMINAMATH_CALUDE_total_cost_calculation_l2487_248704

/-- The cost of items in dollars -/
structure ItemCost where
  mango : ℝ
  rice : ℝ
  flour : ℝ

/-- Given conditions and the theorem to prove -/
theorem total_cost_calculation (c : ItemCost) 
  (h1 : 10 * c.mango = 24 * c.rice)
  (h2 : 6 * c.flour = 2 * c.rice)
  (h3 : c.flour = 23) :
  4 * c.mango + 3 * c.rice + 5 * c.flour = 984.4 := by
  sorry


end NUMINAMATH_CALUDE_total_cost_calculation_l2487_248704


namespace NUMINAMATH_CALUDE_circle_radius_proof_l2487_248790

theorem circle_radius_proof (r : ℝ) (h : r > 0) :
  3 * (2 * Real.pi * r) = 2 * (Real.pi * r^2) → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l2487_248790


namespace NUMINAMATH_CALUDE_no_solution_inequalities_l2487_248724

theorem no_solution_inequalities (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ¬∃ x : ℝ, x > a ∧ x < -b := by
sorry

end NUMINAMATH_CALUDE_no_solution_inequalities_l2487_248724


namespace NUMINAMATH_CALUDE_triangle_side_length_l2487_248774

theorem triangle_side_length (a b c : ℝ) (B : ℝ) : 
  a * c = 8 → a + c = 7 → B = π / 3 → b = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2487_248774


namespace NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l2487_248717

-- Define the function f(x) = |x-a| + x
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + x

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ x + 2} = {x : ℝ | x ≥ 3 ∨ x ≤ -1} := by sorry

-- Part 2
theorem solution_set_part2 (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 3*x} = {x : ℝ | x ≥ 2}) → a = 6 := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_solution_set_part2_l2487_248717


namespace NUMINAMATH_CALUDE_power_sum_prime_l2487_248786

theorem power_sum_prime (p a n : ℕ) : 
  Prime p → a > 0 → n > 0 → (2 ^ p + 3 ^ p = a ^ n) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_prime_l2487_248786


namespace NUMINAMATH_CALUDE_tetrahedron_cross_section_l2487_248703

noncomputable def cross_section_area (V : ℝ) (d : ℝ) : ℝ :=
  3 * V / (5 * d)

theorem tetrahedron_cross_section 
  (V : ℝ) 
  (d : ℝ) 
  (h_V : V = 5) 
  (h_d : d = 1) :
  cross_section_area V d = 3 := by
sorry

end NUMINAMATH_CALUDE_tetrahedron_cross_section_l2487_248703


namespace NUMINAMATH_CALUDE_triangle_area_in_square_pyramid_l2487_248784

/-- Square pyramid with given dimensions and points -/
structure SquarePyramid where
  -- Base side length
  base_side : ℝ
  -- Altitude
  altitude : ℝ
  -- Points P, Q, R are located 1/4 of the way from B, D, C to E respectively
  point_ratio : ℝ

/-- The area of triangle PQR in the square pyramid -/
def triangle_area (pyramid : SquarePyramid) : ℝ := sorry

/-- Theorem statement -/
theorem triangle_area_in_square_pyramid :
  ∀ (pyramid : SquarePyramid),
  pyramid.base_side = 4 ∧ 
  pyramid.altitude = 8 ∧ 
  pyramid.point_ratio = 1/4 →
  triangle_area pyramid = (45 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_in_square_pyramid_l2487_248784


namespace NUMINAMATH_CALUDE_set_A_elements_l2487_248742

theorem set_A_elements (A B : Finset ℤ) (h1 : A.card = 4) (h2 : B = {-1, 3, 5, 8}) 
  (h3 : ∀ S : Finset ℤ, S ⊆ A → S.card = 3 → (S.sum id) ∈ B) 
  (h4 : ∀ b ∈ B, ∃ S : Finset ℤ, S ⊆ A ∧ S.card = 3 ∧ S.sum id = b) : 
  A = {-3, 0, 2, 6} := by
sorry

end NUMINAMATH_CALUDE_set_A_elements_l2487_248742


namespace NUMINAMATH_CALUDE_cost_for_3150_pencils_l2487_248718

/-- Calculates the total cost of pencils with a bulk discount --/
def total_cost_with_discount (pencils_per_box : ℕ) (regular_price : ℚ) 
  (discount_price : ℚ) (discount_threshold : ℕ) (total_pencils : ℕ) : ℚ :=
  let boxes := (total_pencils + pencils_per_box - 1) / pencils_per_box
  let price_per_box := if total_pencils > discount_threshold then discount_price else regular_price
  boxes * price_per_box

/-- Theorem stating the total cost for 3150 pencils --/
theorem cost_for_3150_pencils : 
  total_cost_with_discount 150 40 35 2000 3150 = 735 := by
  sorry

end NUMINAMATH_CALUDE_cost_for_3150_pencils_l2487_248718


namespace NUMINAMATH_CALUDE_range_of_g_l2487_248729

/-- The function g(x) = ⌊2x⌋ - 2x has a range of [-1, 0] -/
theorem range_of_g : 
  let g : ℝ → ℝ := λ x => ⌊2 * x⌋ - 2 * x
  ∀ y : ℝ, (∃ x : ℝ, g x = y) ↔ -1 ≤ y ∧ y ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l2487_248729


namespace NUMINAMATH_CALUDE_map_length_l2487_248753

theorem map_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 10 → area = 20 → area = width * length → length = 2 := by
sorry

end NUMINAMATH_CALUDE_map_length_l2487_248753


namespace NUMINAMATH_CALUDE_simplify_power_l2487_248791

theorem simplify_power (y : ℝ) : (3 * y^4)^5 = 243 * y^20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_l2487_248791


namespace NUMINAMATH_CALUDE_sum_remainder_mod_11_l2487_248758

theorem sum_remainder_mod_11 : (8735 + 8736 + 8737 + 8738) % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_11_l2487_248758


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2487_248749

/-- Given a quadratic equation 3x^2 + mx - 7 = 0 where -1 is one root, 
    prove that the other root is 7/3 -/
theorem other_root_of_quadratic (m : ℝ) : 
  (∃ x, 3 * x^2 + m * x - 7 = 0 ∧ x = -1) → 
  (∃ y, 3 * y^2 + m * y - 7 = 0 ∧ y = 7/3) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2487_248749


namespace NUMINAMATH_CALUDE_like_terms_value_l2487_248721

theorem like_terms_value (m n : ℕ) (a b c : ℝ) : 
  (∃ k : ℝ, 3 * a^m * b * c^2 = k * (-2 * a^3 * b^n * c^2)) → 
  3^2 * n - (2 * m * n^2 - 2 * (m^2 * n + 2 * m * n^2)) = 51 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_value_l2487_248721


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l2487_248727

theorem line_parabola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * p.1 + 2 ∧ p.2^2 = 8 * p.1) ↔ (k = 0 ∨ k = 1) :=
sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l2487_248727


namespace NUMINAMATH_CALUDE_solution_set_implies_sum_l2487_248731

-- Define the inequality solution set
def SolutionSet (a b : ℝ) : Set ℝ :=
  {x | 1 < x ∧ x < 2}

-- State the theorem
theorem solution_set_implies_sum (a b : ℝ) :
  SolutionSet a b = {x | (x - a) * (x - b) < 0} →
  a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_sum_l2487_248731


namespace NUMINAMATH_CALUDE_andy_wrappers_l2487_248759

theorem andy_wrappers (total : ℕ) (max_wrappers : ℕ) (andy_wrappers : ℕ) :
  total = 49 →
  max_wrappers = 15 →
  total = andy_wrappers + max_wrappers →
  andy_wrappers = 34 := by
sorry

end NUMINAMATH_CALUDE_andy_wrappers_l2487_248759
