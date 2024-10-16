import Mathlib

namespace NUMINAMATH_CALUDE_sum_32_45_base5_l2961_296143

/-- Converts a base 10 number to base 5 --/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number --/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_32_45_base5 :
  toBase5 (32 + 45) = [3, 0, 2] :=
sorry

end NUMINAMATH_CALUDE_sum_32_45_base5_l2961_296143


namespace NUMINAMATH_CALUDE_shirts_bought_l2961_296102

/-- Given John's initial and final shirt counts, prove the number of shirts bought. -/
theorem shirts_bought (initial_shirts final_shirts : ℕ) 
  (h1 : initial_shirts = 12)
  (h2 : final_shirts = 16)
  : final_shirts - initial_shirts = 4 := by
  sorry

end NUMINAMATH_CALUDE_shirts_bought_l2961_296102


namespace NUMINAMATH_CALUDE_train_length_l2961_296103

/-- The length of a train given its speed, time to pass a platform, and platform length -/
theorem train_length (train_speed : ℝ) (pass_time : ℝ) (platform_length : ℝ) : 
  train_speed = 45 * (1000 / 3600) →
  pass_time = 40 →
  platform_length = 140 →
  train_speed * pass_time - platform_length = 360 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2961_296103


namespace NUMINAMATH_CALUDE_product_purchase_savings_l2961_296136

/-- Proves that under given conditions, the product could have been purchased for 10% less -/
theorem product_purchase_savings (original_selling_price : ℝ) 
  (h1 : original_selling_price = 989.9999999999992)
  (h2 : original_selling_price = 1.1 * original_purchase_price)
  (h3 : 1.3 * reduced_purchase_price = original_selling_price + 63) :
  (original_purchase_price - reduced_purchase_price) / original_purchase_price = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_product_purchase_savings_l2961_296136


namespace NUMINAMATH_CALUDE_winner_is_C_l2961_296178

structure Singer :=
  (name : String)

def Singers : List Singer := [⟨"A"⟩, ⟨"B"⟩, ⟨"C"⟩, ⟨"D"⟩]

def Statement : Singer → Prop
| ⟨"A"⟩ => ∃ s : Singer, (s.name = "B" ∨ s.name = "C") ∧ s ∈ Singers
| ⟨"B"⟩ => ∀ s : Singer, (s.name = "A" ∨ s.name = "C") → s ∉ Singers
| ⟨"C"⟩ => ⟨"C"⟩ ∈ Singers
| ⟨"D"⟩ => ⟨"B"⟩ ∈ Singers
| _ => False

def Winner (s : Singer) : Prop :=
  s ∈ Singers ∧
  (∀ t : Singer, t ∈ Singers ∧ t ≠ s → t ∉ Singers) ∧
  (∃ (s1 s2 : Singer), s1 ≠ s2 ∧ Statement s1 ∧ Statement s2 ∧
    (∀ s3 : Singer, s3 ≠ s1 ∧ s3 ≠ s2 → ¬Statement s3))

theorem winner_is_C :
  Winner ⟨"C"⟩ ∧ (∀ s : Singer, s ≠ ⟨"C"⟩ → ¬Winner s) :=
sorry

end NUMINAMATH_CALUDE_winner_is_C_l2961_296178


namespace NUMINAMATH_CALUDE_students_per_normal_class_l2961_296119

theorem students_per_normal_class
  (total_students : ℕ)
  (moving_percentage : ℚ)
  (grade_levels : ℕ)
  (advanced_class_size : ℕ)
  (normal_classes_per_grade : ℕ)
  (h1 : total_students = 1590)
  (h2 : moving_percentage = 40 / 100)
  (h3 : grade_levels = 3)
  (h4 : advanced_class_size = 20)
  (h5 : normal_classes_per_grade = 6)
  : ℕ :=
by
  -- Proof goes here
  sorry

#check @students_per_normal_class

end NUMINAMATH_CALUDE_students_per_normal_class_l2961_296119


namespace NUMINAMATH_CALUDE_bounded_area_calculation_l2961_296146

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the area of the region bounded by two circles and two vertical lines -/
def boundedArea (c1 c2 : Circle) (x1 x2 : ℝ) : ℝ :=
  sorry

theorem bounded_area_calculation :
  let c1 : Circle := { center := (4, 4), radius := 4 }
  let c2 : Circle := { center := (12, 12), radius := 4 }
  let x1 : ℝ := 4
  let x2 : ℝ := 12
  boundedArea c1 c2 x1 x2 = 64 - 8 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_bounded_area_calculation_l2961_296146


namespace NUMINAMATH_CALUDE_no_integer_solutions_l2961_296137

theorem no_integer_solutions : ¬∃ (x y : ℤ), 3 * x^2 = 16 * y^2 + 8 * y + 5 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l2961_296137


namespace NUMINAMATH_CALUDE_function_derivative_problem_l2961_296139

theorem function_derivative_problem (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = (2*x + a)^2)
  (h2 : deriv f 2 = 20) : 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_function_derivative_problem_l2961_296139


namespace NUMINAMATH_CALUDE_ac_range_l2961_296132

-- Define the triangle
def Triangle (A B C : ℝ × ℝ) : Prop := sorry

-- Define the angle at a vertex
def Angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the length of a side
def SideLength (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ac_range (A B C : ℝ × ℝ) : 
  Triangle A B C → 
  Angle A B C < π / 2 → 
  Angle B C A < π / 2 → 
  Angle C A B < π / 2 → 
  SideLength B C = 1 → 
  Angle B A C = 2 * Angle A B C → 
  Real.sqrt 2 < SideLength A C ∧ SideLength A C < Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ac_range_l2961_296132


namespace NUMINAMATH_CALUDE_rectangle_tiles_l2961_296115

theorem rectangle_tiles (length width : ℕ) : 
  width = 2 * length →
  (length * length + width * width : ℚ).sqrt = 45 →
  length * width = 810 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_tiles_l2961_296115


namespace NUMINAMATH_CALUDE_least_acute_triangle_side_l2961_296141

/-- A function that checks if three side lengths form an acute triangle -/
def is_acute_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2

/-- The least positive integer A such that an acute triangle with side lengths 5, A, and 8 exists -/
theorem least_acute_triangle_side : ∃ (A : ℕ), 
  (∀ (k : ℕ), k < A → ¬is_acute_triangle 5 (k : ℝ) 8) ∧ 
  is_acute_triangle 5 A 8 ∧
  A = 7 := by
  sorry

end NUMINAMATH_CALUDE_least_acute_triangle_side_l2961_296141


namespace NUMINAMATH_CALUDE_square_sequence_50th_term_l2961_296109

/-- Represents the number of squares in the nth figure of the sequence -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- The theorem states that the 50th term of the sequence is 7651 -/
theorem square_sequence_50th_term :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 → f 50 = 7651 := by
  sorry

end NUMINAMATH_CALUDE_square_sequence_50th_term_l2961_296109


namespace NUMINAMATH_CALUDE_new_person_weight_l2961_296124

/-- The weight of the new person given the conditions of the problem -/
theorem new_person_weight (n : ℕ) (initial_weight : ℝ) (weight_increase : ℝ) : 
  n = 9 → 
  initial_weight = 86 → 
  weight_increase = 5.5 → 
  (n : ℝ) * weight_increase + initial_weight = 135.5 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l2961_296124


namespace NUMINAMATH_CALUDE_fake_to_total_handbags_ratio_l2961_296194

theorem fake_to_total_handbags_ratio
  (total_purses : ℕ)
  (total_handbags : ℕ)
  (authentic_items : ℕ)
  (h1 : total_purses = 26)
  (h2 : total_handbags = 24)
  (h3 : authentic_items = 31)
  (h4 : total_purses / 2 = total_purses - authentic_items + total_handbags - authentic_items) :
  (total_handbags - (authentic_items - total_purses / 2)) / total_handbags = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_fake_to_total_handbags_ratio_l2961_296194


namespace NUMINAMATH_CALUDE_unique_perfect_square_sum_l2961_296186

theorem unique_perfect_square_sum (p : Nat) (hp : p.Prime ∧ p > 2) :
  ∃! n : Nat, n > 0 ∧ ∃ k : Nat, n^2 + n*p = k^2 :=
by
  use ((p - 1)^2) / 4
  sorry

end NUMINAMATH_CALUDE_unique_perfect_square_sum_l2961_296186


namespace NUMINAMATH_CALUDE_machine_doesnt_require_repair_l2961_296163

/-- Represents a weighing machine for food portions -/
structure WeighingMachine where
  max_deviation : ℝ
  nominal_mass : ℝ
  unreadable_deviation_bound : ℝ

/-- Determines if a weighing machine requires repair based on its measurements -/
def requires_repair (m : WeighingMachine) : Prop :=
  m.max_deviation > 0.1 * m.nominal_mass ∨ 
  m.unreadable_deviation_bound ≥ m.max_deviation

theorem machine_doesnt_require_repair (m : WeighingMachine) 
  (h1 : m.max_deviation = 37)
  (h2 : m.max_deviation ≤ 0.1 * m.nominal_mass)
  (h3 : m.unreadable_deviation_bound < m.max_deviation) :
  ¬(requires_repair m) :=
sorry

#check machine_doesnt_require_repair

end NUMINAMATH_CALUDE_machine_doesnt_require_repair_l2961_296163


namespace NUMINAMATH_CALUDE_sqrt_10_irrational_l2961_296165

theorem sqrt_10_irrational : Irrational (Real.sqrt 10) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_10_irrational_l2961_296165


namespace NUMINAMATH_CALUDE_fraction_subtraction_fraction_division_l2961_296118

-- Problem 1
theorem fraction_subtraction (x y : ℝ) (h : x + y ≠ 0) :
  (2 * x + 3 * y) / (x + y) - (x + 2 * y) / (x + y) = 1 := by
sorry

-- Problem 2
theorem fraction_division (a : ℝ) (h : a ≠ 2) :
  (a^2 - 1) / (a^2 - 4*a + 4) / ((a - 1) / (a - 2)) = (a + 1) / (a - 2) := by
sorry

end NUMINAMATH_CALUDE_fraction_subtraction_fraction_division_l2961_296118


namespace NUMINAMATH_CALUDE_cylinder_volume_l2961_296153

/-- The volume of a cylinder with diameter 4 cm and height 5 cm is equal to π * 20 cm³ -/
theorem cylinder_volume (π : ℝ) (h : π = Real.pi) : 
  let d : ℝ := 4 -- diameter in cm
  let h : ℝ := 5 -- height in cm
  let r : ℝ := d / 2 -- radius in cm
  let v : ℝ := π * r^2 * h -- volume formula
  v = π * 20 := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l2961_296153


namespace NUMINAMATH_CALUDE_min_crooks_proof_l2961_296191

/-- Represents the total number of ministers -/
def total_ministers : ℕ := 100

/-- Represents the size of any subgroup of ministers that must contain at least one crook -/
def subgroup_size : ℕ := 10

/-- Represents the property that any subgroup of ministers contains at least one crook -/
def at_least_one_crook (num_crooks : ℕ) : Prop :=
  ∀ (subgroup : Finset ℕ), subgroup.card = subgroup_size → 
    (total_ministers - num_crooks < subgroup.card)

/-- The minimum number of crooks in the cabinet -/
def min_crooks : ℕ := total_ministers - (subgroup_size - 1)

theorem min_crooks_proof :
  (at_least_one_crook min_crooks) ∧ 
  (∀ k < min_crooks, ¬(at_least_one_crook k)) :=
sorry

end NUMINAMATH_CALUDE_min_crooks_proof_l2961_296191


namespace NUMINAMATH_CALUDE_crackers_distribution_l2961_296195

theorem crackers_distribution (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_friend : ℕ) :
  total_crackers = 81 →
  num_friends = 27 →
  total_crackers = num_friends * crackers_per_friend →
  crackers_per_friend = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_crackers_distribution_l2961_296195


namespace NUMINAMATH_CALUDE_square_plot_side_length_l2961_296180

theorem square_plot_side_length (area : ℝ) (side : ℝ) : 
  area = 2550.25 → side * side = area → side = 50.5 := by sorry

end NUMINAMATH_CALUDE_square_plot_side_length_l2961_296180


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2961_296106

theorem simplify_polynomial (p : ℝ) : 
  (3 * p^3 - 5*p + 6) + (4 - 6*p^2 + 2*p) = 3*p^3 - 6*p^2 - 3*p + 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2961_296106


namespace NUMINAMATH_CALUDE_inequality_proof_l2961_296144

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := 2 * (a + 1) * Real.log x - a * x

def g (x : ℝ) : ℝ := (1 / 2) * x^2 - x

theorem inequality_proof (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : -1 < a ∧ a < 7) 
  (hx₁ : x₁ > 1) 
  (hx₂ : x₂ > 1) 
  (hne : x₁ ≠ x₂) : 
  (f a x₁ - f a x₂) / (g x₁ - g x₂) > -1 := by
  sorry

end

end NUMINAMATH_CALUDE_inequality_proof_l2961_296144


namespace NUMINAMATH_CALUDE_radiator_water_fraction_l2961_296188

def radiator_capacity : ℚ := 20

def replacement_volume : ℚ := 5

def water_fraction_in_mixed_solution : ℚ := 1/2

def final_water_fraction (initial_water : ℚ) : ℚ :=
  let after_first_replacement := (initial_water - replacement_volume) / radiator_capacity
  let after_second_replacement := 
    (after_first_replacement * radiator_capacity - replacement_volume * after_first_replacement) / radiator_capacity
  let after_third_replacement := 
    (after_second_replacement * radiator_capacity - replacement_volume * after_second_replacement + 
     replacement_volume * water_fraction_in_mixed_solution) / radiator_capacity
  let after_fourth_replacement := 
    (after_third_replacement * radiator_capacity - replacement_volume * after_third_replacement + 
     replacement_volume * water_fraction_in_mixed_solution) / radiator_capacity
  after_fourth_replacement

theorem radiator_water_fraction :
  final_water_fraction radiator_capacity = 213 / 400 := by
  sorry

end NUMINAMATH_CALUDE_radiator_water_fraction_l2961_296188


namespace NUMINAMATH_CALUDE_abs_equation_solution_l2961_296148

theorem abs_equation_solution (x : ℝ) : |-5 + x| = 3 → x = 8 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_l2961_296148


namespace NUMINAMATH_CALUDE_conditions_implications_l2961_296159

-- Define the conditions
def p (a : ℝ) : Prop := ∀ x > 0, Monotone (fun x => Real.log x / Real.log (a - 1))
def q (a : ℝ) : Prop := (2 - a) / (a - 3) > 0
def r (a : ℝ) : Prop := a < 3
def s (a : ℝ) : Prop := ∀ x, ∃ y, y = Real.log (x^2 - 2*x + a) / Real.log 10

-- State the theorem
theorem conditions_implications (a : ℝ) :
  (p a → a > 2) ∧
  (q a ↔ (2 < a ∧ a < 3)) ∧
  (s a → a > 1) ∧
  ((p a → q a) ∧ ¬(q a → p a)) ∧
  ((r a → q a) ∧ ¬(q a → r a)) :=
sorry

end NUMINAMATH_CALUDE_conditions_implications_l2961_296159


namespace NUMINAMATH_CALUDE_safari_count_l2961_296107

/-- The total number of animals counted during the safari --/
def total_animals (antelopes rabbits hyenas wild_dogs leopards : ℕ) : ℕ :=
  antelopes + rabbits + hyenas + wild_dogs + leopards

/-- Theorem stating the total number of animals counted during the safari --/
theorem safari_count : ∃ (antelopes rabbits hyenas wild_dogs leopards : ℕ),
  antelopes = 80 ∧
  rabbits = antelopes + 34 ∧
  hyenas = antelopes + rabbits - 42 ∧
  wild_dogs = hyenas + 50 ∧
  leopards = rabbits / 2 ∧
  total_animals antelopes rabbits hyenas wild_dogs leopards = 605 := by
  sorry


end NUMINAMATH_CALUDE_safari_count_l2961_296107


namespace NUMINAMATH_CALUDE_y_axis_reflection_l2961_296162

/-- Given a point P(-2,3) in the Cartesian coordinate system, 
    its coordinates with respect to the y-axis are (2,3). -/
theorem y_axis_reflection :
  let P : ℝ × ℝ := (-2, 3)
  let reflected_P : ℝ × ℝ := (2, 3)
  reflected_P = (-(P.1), P.2) :=
by sorry

end NUMINAMATH_CALUDE_y_axis_reflection_l2961_296162


namespace NUMINAMATH_CALUDE_cyclic_quadrilaterals_area_l2961_296101

/-- Represents a convex cyclic quadrilateral --/
structure CyclicQuadrilateral where
  diag_angle : Real
  inscribed_radius : Real
  area : Real

/-- The theorem to be proved --/
theorem cyclic_quadrilaterals_area 
  (A B C : CyclicQuadrilateral)
  (h_radius : A.inscribed_radius = 1 ∧ B.inscribed_radius = 1 ∧ C.inscribed_radius = 1)
  (h_sin_A : Real.sin A.diag_angle = 2/3)
  (h_sin_B : Real.sin B.diag_angle = 3/5)
  (h_sin_C : Real.sin C.diag_angle = 6/7)
  (h_equal_area : A.area = B.area ∧ B.area = C.area)
  : A.area = 16/35 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilaterals_area_l2961_296101


namespace NUMINAMATH_CALUDE_complex_ratio_theorem_l2961_296114

theorem complex_ratio_theorem (a b : ℝ) (z : ℂ) (h1 : z = Complex.mk a b) 
  (h2 : ∃ (k : ℝ), z / Complex.mk 2 1 = Complex.mk 0 k) : b / a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_ratio_theorem_l2961_296114


namespace NUMINAMATH_CALUDE_bus_departure_theorem_l2961_296108

/-- Represents the rules for bus departure and current occupancy -/
structure BusOccupancy where
  min_departure : Nat
  max_departure : Nat
  current_occupancy : Nat
  departure_rule : min_departure > 15 ∧ max_departure ≤ 30
  occupancy_valid : current_occupancy < min_departure

/-- Calculates the number of additional people needed for the bus to depart -/
def additional_people_needed (bus : BusOccupancy) : Nat :=
  bus.min_departure - bus.current_occupancy

/-- Theorem stating that for a bus with specific occupancy rules and current state,
    the number of additional people needed is 7 -/
theorem bus_departure_theorem (bus : BusOccupancy)
    (h1 : bus.min_departure = 16)
    (h2 : bus.current_occupancy = 9) :
    additional_people_needed bus = 7 := by
  sorry

#eval additional_people_needed ⟨16, 30, 9, by simp, by simp⟩

end NUMINAMATH_CALUDE_bus_departure_theorem_l2961_296108


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2961_296112

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ)
  (h_arith : arithmetic_sequence a d)
  (h_sum : ∀ n : ℕ, S n = n * (2 * a 1 + (n - 1) * d) / 2)
  (h_positive : a 1 > 0)
  (h_condition : a 9 + a 10 = a 11) :
  (d < 0) ∧
  (∀ n : ℕ, n > 14 → S n ≤ 0) ∧
  (∃ n : ℕ, n = 14 ∧ S n > 0) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2961_296112


namespace NUMINAMATH_CALUDE_vector_sum_proof_l2961_296169

theorem vector_sum_proof : 
  let v1 : Fin 3 → ℝ := ![5, -3, 8]
  let v2 : Fin 3 → ℝ := ![-2, 4, 1]
  let v3 : Fin 3 → ℝ := ![3, -6, -9]
  v1 + v2 + v3 = ![6, -5, 0] := by
sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l2961_296169


namespace NUMINAMATH_CALUDE_inscribed_equal_angles_is_regular_circumscribed_equal_sides_is_regular_l2961_296190

-- Define a polygon type
structure Polygon where
  vertices : List ℝ × ℝ
  sides : Nat
  is_odd : Odd sides

-- Define properties for inscribed and circumscribed polygons
def is_inscribed (p : Polygon) : Prop := sorry
def is_circumscribed (p : Polygon) : Prop := sorry

-- Define properties for equal angles and equal sides
def has_equal_angles (p : Polygon) : Prop := sorry
def has_equal_sides (p : Polygon) : Prop := sorry

-- Define what it means for a polygon to be regular
def is_regular (p : Polygon) : Prop := sorry

-- Theorem for part a
theorem inscribed_equal_angles_is_regular (p : Polygon) 
  (h_inscribed : is_inscribed p) (h_equal_angles : has_equal_angles p) : 
  is_regular p := by sorry

-- Theorem for part b
theorem circumscribed_equal_sides_is_regular (p : Polygon) 
  (h_circumscribed : is_circumscribed p) (h_equal_sides : has_equal_sides p) : 
  is_regular p := by sorry

end NUMINAMATH_CALUDE_inscribed_equal_angles_is_regular_circumscribed_equal_sides_is_regular_l2961_296190


namespace NUMINAMATH_CALUDE_cuboid_third_edge_length_l2961_296135

/-- Given a cuboid with two edges of 4 cm each and a volume of 96 cm³, 
    prove that the length of the third edge is 6 cm. -/
theorem cuboid_third_edge_length 
  (edge1 : ℝ) (edge2 : ℝ) (volume : ℝ) (third_edge : ℝ) 
  (h1 : edge1 = 4) 
  (h2 : edge2 = 4) 
  (h3 : volume = 96) 
  (h4 : volume = edge1 * edge2 * third_edge) : 
  third_edge = 6 :=
sorry

end NUMINAMATH_CALUDE_cuboid_third_edge_length_l2961_296135


namespace NUMINAMATH_CALUDE_system_solution_l2961_296174

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  3 * x + y = 5 ∧ x + 3 * y = 7

-- State the theorem
theorem system_solution :
  ∃! (x y : ℝ), system x y ∧ x = 1 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2961_296174


namespace NUMINAMATH_CALUDE_tram_speed_l2961_296133

/-- Given a tram passing an observer in 2 seconds and traversing a 96-meter tunnel in 10 seconds
    at a constant speed, the speed of the tram is 12 meters per second. -/
theorem tram_speed (passing_time : ℝ) (tunnel_length : ℝ) (tunnel_time : ℝ)
    (h1 : passing_time = 2)
    (h2 : tunnel_length = 96)
    (h3 : tunnel_time = 10) :
  ∃ (v : ℝ), v = 12 ∧ v * passing_time = v * 2 ∧ v * tunnel_time = v * 2 + tunnel_length :=
by
  sorry

#check tram_speed

end NUMINAMATH_CALUDE_tram_speed_l2961_296133


namespace NUMINAMATH_CALUDE_book_sales_ratio_l2961_296185

theorem book_sales_ratio : 
  ∀ (wednesday thursday friday : ℕ),
  wednesday = 15 →
  thursday = 3 * wednesday →
  wednesday + thursday + friday = 69 →
  friday * 5 = thursday :=
λ wednesday thursday friday hw ht htot =>
  sorry

end NUMINAMATH_CALUDE_book_sales_ratio_l2961_296185


namespace NUMINAMATH_CALUDE_fraction_problem_l2961_296164

theorem fraction_problem (x : ℚ) : 150 * x = 37 + 1/2 → x = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2961_296164


namespace NUMINAMATH_CALUDE_angles_around_point_l2961_296111

theorem angles_around_point (a b c : ℝ) : 
  a + b + c = 360 →  -- sum of angles around a point is 360°
  c = 120 →          -- one angle is 120°
  a = b →            -- the other two angles are equal
  a = 120 :=         -- prove that each of the equal angles is 120°
by sorry

end NUMINAMATH_CALUDE_angles_around_point_l2961_296111


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2961_296179

theorem sum_of_reciprocals (x y : ℚ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0)
  (h3 : 1/x + 1/y = 4) (h4 : 1/x - 1/y = -8) : 
  x + y = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2961_296179


namespace NUMINAMATH_CALUDE_expression_equals_two_l2961_296187

/-- Given real numbers a, b, and c satisfying two conditions, 
    prove that a certain expression equals 2 -/
theorem expression_equals_two (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = 8)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 7) :
  b / (a + b) + c / (b + c) + a / (c + a) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_l2961_296187


namespace NUMINAMATH_CALUDE_right_triangle_area_l2961_296173

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) : (1/2) * a * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2961_296173


namespace NUMINAMATH_CALUDE_solution_existence_l2961_296113

theorem solution_existence (k : ℝ) : 
  (∃ x ∈ Set.Icc 0 2, k * 9^x - k * 3^(x + 1) + 6 * (k - 5) = 0) ↔ k ∈ Set.Icc (1/2) 8 := by
  sorry

end NUMINAMATH_CALUDE_solution_existence_l2961_296113


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2961_296157

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), Prime p ∧ p ∣ (7^15 + 9^17) ∧ ∀ q, Prime q → q ∣ (7^15 + 9^17) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l2961_296157


namespace NUMINAMATH_CALUDE_christian_future_age_l2961_296181

/-- The age of Brian in years -/
def brian_age : ℕ := sorry

/-- The age of Christian in years -/
def christian_age : ℕ := sorry

/-- The number of years in the future we're considering -/
def years_future : ℕ := 8

/-- Brian's age in the future -/
def brian_future_age : ℕ := 40

theorem christian_future_age :
  christian_age + years_future = 72 :=
by
  have h1 : christian_age = 2 * brian_age := sorry
  have h2 : brian_age + years_future = brian_future_age := sorry
  sorry

end NUMINAMATH_CALUDE_christian_future_age_l2961_296181


namespace NUMINAMATH_CALUDE_f_is_quadratic_l2961_296122

/-- A quadratic equation in terms of x is of the form ax^2 + bx + c = 0, where a ≠ 0, b, and c are real numbers. -/
def IsQuadraticEquation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing 2x^2 - 1 = 0 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : IsQuadraticEquation f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l2961_296122


namespace NUMINAMATH_CALUDE_largest_possible_a_l2961_296160

theorem largest_possible_a (a b c d : ℕ+) 
  (h1 : a < 3 * b)
  (h2 : b < 2 * c + 1)
  (h3 : c < 5 * d - 2)
  (h4 : d ≤ 50)
  (h5 : ∃ k : ℕ, d = 5 * k) :
  a ≤ 1481 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 1481 ∧
    a' < 3 * b' ∧
    b' < 2 * c' + 1 ∧
    c' < 5 * d' - 2 ∧
    d' ≤ 50 ∧
    ∃ k : ℕ, d' = 5 * k :=
by sorry

end NUMINAMATH_CALUDE_largest_possible_a_l2961_296160


namespace NUMINAMATH_CALUDE_ice_cream_cost_l2961_296172

/-- Given the following conditions:
    - Alok ordered 16 chapatis, 5 plates of rice, 7 plates of mixed vegetable, and 6 ice-cream cups
    - Cost of each chapati is Rs. 6
    - Cost of each plate of rice is Rs. 45
    - Cost of each plate of mixed vegetable is Rs. 70
    - Alok paid the cashier Rs. 985
    Prove that the cost of each ice-cream cup is Rs. 29 -/
theorem ice_cream_cost (chapati_count : ℕ) (rice_count : ℕ) (vegetable_count : ℕ) (ice_cream_count : ℕ)
                       (chapati_cost : ℕ) (rice_cost : ℕ) (vegetable_cost : ℕ) (total_paid : ℕ) :
  chapati_count = 16 →
  rice_count = 5 →
  vegetable_count = 7 →
  ice_cream_count = 6 →
  chapati_cost = 6 →
  rice_cost = 45 →
  vegetable_cost = 70 →
  total_paid = 985 →
  (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost)) / ice_cream_count = 29 := by
  sorry


end NUMINAMATH_CALUDE_ice_cream_cost_l2961_296172


namespace NUMINAMATH_CALUDE_part_one_part_two_l2961_296117

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x ∈ Set.Icc 1 2, (a - 1) * x - 1 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 1 > 0

-- Theorem for part (1)
theorem part_one (a : ℝ) : (p a ∧ q a) ↔ (3/2 < a ∧ a < 2) :=
sorry

-- Theorem for part (2)
theorem part_two (a : ℝ) : (¬(¬(p a) ∧ q a) ∧ (¬(p a) ∨ q a)) ↔ (a ≤ -2 ∨ (3/2 < a ∧ a < 2)) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2961_296117


namespace NUMINAMATH_CALUDE_smallest_n_without_quadratic_number_l2961_296161

def isQuadraticNumber (x : ℝ) : Prop :=
  ∃ (a b c : ℤ), a ≠ 0 ∧ 
  (|a| ≤ 10 ∧ |a| ≥ 1) ∧ 
  (|b| ≤ 10 ∧ |b| ≥ 1) ∧ 
  (|c| ≤ 10 ∧ |c| ≥ 1) ∧ 
  a * x^2 + b * x + c = 0

def hasQuadraticNumber (l r : ℝ) : Prop :=
  ∃ x, l < x ∧ x < r ∧ isQuadraticNumber x

def noQuadraticNumber (n : ℕ) : Prop :=
  ¬(hasQuadraticNumber (n - 1/3) n) ∨ ¬(hasQuadraticNumber n (n + 1/3))

theorem smallest_n_without_quadratic_number :
  (∀ m : ℕ, m < 11 → ¬(noQuadraticNumber m)) ∧ noQuadraticNumber 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_without_quadratic_number_l2961_296161


namespace NUMINAMATH_CALUDE_same_remainder_divisor_l2961_296183

theorem same_remainder_divisor : ∃ (d : ℕ), d > 0 ∧ 
  ∀ (k : ℕ), k > d → 
  (∃ (r₁ r₂ r₃ : ℕ), 
    480608 = k * r₁ + d ∧
    508811 = k * r₂ + d ∧
    723217 = k * r₃ + d) → False :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_same_remainder_divisor_l2961_296183


namespace NUMINAMATH_CALUDE_intersection_triangle_is_right_l2961_296175

/-- An ellipse with equation x²/m + y² = 1, where m > 1 -/
structure Ellipse where
  m : ℝ
  h_m : m > 1

/-- A hyperbola with equation x²/n - y² = 1, where n > 0 -/
structure Hyperbola where
  n : ℝ
  h_n : n > 0

/-- Two foci points -/
structure Foci where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- An intersection point of the ellipse and hyperbola -/
structure IntersectionPoint where
  P : ℝ × ℝ

/-- The main theorem stating that the triangle formed by the foci and the intersection point is right -/
theorem intersection_triangle_is_right (e : Ellipse) (h : Hyperbola) (f : Foci) (p : IntersectionPoint) :
  ∃ (x y : ℝ), x^2 + y^2 = (f.F₁.1 - f.F₂.1)^2 + (f.F₁.2 - f.F₂.2)^2 :=
sorry

end NUMINAMATH_CALUDE_intersection_triangle_is_right_l2961_296175


namespace NUMINAMATH_CALUDE_borrow_three_books_l2961_296158

/-- The number of ways to borrow at least one book out of three books -/
def borrow_methods (n : ℕ) : ℕ := 2^n - 1

/-- Theorem stating that the number of ways to borrow at least one book out of three books is 7 -/
theorem borrow_three_books : borrow_methods 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_borrow_three_books_l2961_296158


namespace NUMINAMATH_CALUDE_angle_ADC_measure_l2961_296156

theorem angle_ADC_measure (ABC BAC BCA BAD CAD ACD BCD ADC : Real) : 
  ABC = 60 →
  BAC = BAD + CAD →
  BAD = CAD →
  BCA = ACD + BCD →
  ACD = 2 * BCD →
  BAC + ABC + BCA = 180 →
  CAD + ACD + ADC = 180 →
  ADC = 100 :=
by sorry

end NUMINAMATH_CALUDE_angle_ADC_measure_l2961_296156


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2961_296166

theorem infinite_series_sum : 
  let series := fun k : ℕ => (3^(2^k)) / ((5^(2^k)) - 2)
  ∑' k, series k = 1 / (Real.sqrt 5 - 1) := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2961_296166


namespace NUMINAMATH_CALUDE_union_equals_M_l2961_296145

-- Define the set M
def M : Set ℝ := {y | ∃ x, y = 2^x}

-- Define the set S
def S : Set ℝ := {x | ∃ y, y = x - 1}

-- State the theorem
theorem union_equals_M : M ∪ S = M :=
sorry

end NUMINAMATH_CALUDE_union_equals_M_l2961_296145


namespace NUMINAMATH_CALUDE_coefficient_proof_l2961_296105

theorem coefficient_proof (x : ℕ) (some_number : ℕ) :
  x = 13 →
  (2^x) - (2^(x-2)) = some_number * (2^11) →
  some_number = 3 := by
sorry

end NUMINAMATH_CALUDE_coefficient_proof_l2961_296105


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2961_296155

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 1}
def B : Set ℝ := {x : ℝ | x - 4 < 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x > 1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2961_296155


namespace NUMINAMATH_CALUDE_complex_magnitude_and_real_part_sum_of_combinations_l2961_296196

-- Problem 1
theorem complex_magnitude_and_real_part (z : ℂ) (ω : ℝ) 
  (h1 : ω = z + 1/z)
  (h2 : -1 < ω)
  (h3 : ω < 2) :
  Complex.abs z = 1 ∧ ∃ (a : ℝ), z.re = a ∧ -1/2 < a ∧ a < 1 :=
sorry

-- Problem 2
theorem sum_of_combinations : 
  (Nat.choose 5 4) + (Nat.choose 6 4) + (Nat.choose 7 4) + 
  (Nat.choose 8 4) + (Nat.choose 9 4) + (Nat.choose 10 4) = 461 :=
sorry

end NUMINAMATH_CALUDE_complex_magnitude_and_real_part_sum_of_combinations_l2961_296196


namespace NUMINAMATH_CALUDE_pencil_pen_difference_l2961_296127

/-- Given a ratio of pens to pencils and the total number of pencils,
    calculate the difference between pencils and pens. -/
theorem pencil_pen_difference
  (ratio_pens : ℕ)
  (ratio_pencils : ℕ)
  (total_pencils : ℕ)
  (h_ratio : ratio_pens < ratio_pencils)
  (h_total : total_pencils = 36)
  (h_ratio_pencils : total_pencils % ratio_pencils = 0) :
  total_pencils - (total_pencils / ratio_pencils * ratio_pens) = 6 :=
by sorry

end NUMINAMATH_CALUDE_pencil_pen_difference_l2961_296127


namespace NUMINAMATH_CALUDE_garage_bikes_l2961_296184

/-- Given a number of wheels and the number of wheels required per bike, 
    calculate the number of bikes that can be assembled -/
def bikes_assembled (total_wheels : ℕ) (wheels_per_bike : ℕ) : ℕ :=
  total_wheels / wheels_per_bike

theorem garage_bikes : bikes_assembled 14 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_garage_bikes_l2961_296184


namespace NUMINAMATH_CALUDE_rug_profit_calculation_l2961_296134

/-- Calculate the profit from selling rugs -/
theorem rug_profit_calculation (cost_price selling_price number_of_rugs : ℕ) :
  let profit_per_rug := selling_price - cost_price
  let total_profit := number_of_rugs * profit_per_rug
  total_profit = number_of_rugs * (selling_price - cost_price) :=
by sorry

end NUMINAMATH_CALUDE_rug_profit_calculation_l2961_296134


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_equals_seven_l2961_296177

theorem complex_arithmetic_expression_equals_seven :
  (2 + 3/5 - (17/2 - 8/3) / (7/2)) * (15/2) = 7 := by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_equals_seven_l2961_296177


namespace NUMINAMATH_CALUDE_tan_sum_special_l2961_296198

theorem tan_sum_special (θ : Real) (h : Real.tan θ = 2) : Real.tan (θ + π/4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_special_l2961_296198


namespace NUMINAMATH_CALUDE_stone_137_is_5_l2961_296192

/-- Represents the number of stones in the sequence -/
def num_stones : ℕ := 11

/-- Represents the length of a full counting cycle -/
def cycle_length : ℕ := 20

/-- Represents the target count number -/
def target_count : ℕ := 137

/-- Represents the original stone number we want to prove -/
def original_stone : ℕ := 5

/-- Function to determine the stone number given a count in the sequence -/
def stone_at_count (count : ℕ) : ℕ :=
  sorry

theorem stone_137_is_5 : stone_at_count target_count = original_stone := by
  sorry

end NUMINAMATH_CALUDE_stone_137_is_5_l2961_296192


namespace NUMINAMATH_CALUDE_parabola_properties_l2961_296168

-- Define the parabola function
def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 4

-- Define the interval
def interval : Set ℝ := Set.Icc (-3) 2

-- Theorem statement
theorem parabola_properties :
  (∃ (x y : ℝ), x = -1 ∧ y = 1 ∧ ∀ (t : ℝ), f t ≥ f x) ∧ 
  (∀ (x₁ x₂ : ℝ), f x₁ = f x₂ → |x₁ + 1| = |x₂ + 1|) ∧
  (Set.Icc 1 28 = {y | ∃ x ∈ interval, f x = y}) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l2961_296168


namespace NUMINAMATH_CALUDE_correct_journey_equation_l2961_296171

/-- Represents the journey of a ship between two ports -/
def ship_journey (distance : ℝ) (flow_speed : ℝ) (ship_speed : ℝ) : Prop :=
  distance / (ship_speed + flow_speed) + distance / (ship_speed - flow_speed) = 8

/-- Theorem stating that the given equation correctly represents the ship's journey -/
theorem correct_journey_equation :
  ∀ x : ℝ, x > 4 → ship_journey 50 4 x :=
by
  sorry

end NUMINAMATH_CALUDE_correct_journey_equation_l2961_296171


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2961_296100

theorem circle_area_ratio (r₁ r₂ : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 3) 
  (h₃ : (r₁ + r₂)^2 + 40^2 = 41^2) (h₄ : 20 * (r₁ + r₂) = 300) :
  (π * r₁^2) / (π * r₂^2) = 16 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2961_296100


namespace NUMINAMATH_CALUDE_equation_solution_l2961_296147

theorem equation_solution : ∃! (x y : ℝ), 3*x^2 + 14*y^2 - 12*x*y + 6*x - 20*y + 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2961_296147


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l2961_296149

def f (x : ℝ) := x^3 - x^2 - x - 1

theorem root_exists_in_interval :
  (f 1 < 0) → (f 2 > 0) → ∃ x ∈ Set.Ioo 1 2, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l2961_296149


namespace NUMINAMATH_CALUDE_repeating_decimal_726_eq_fraction_l2961_296199

/-- The definition of a repeating decimal with period 726 -/
def repeating_decimal_726 : ℚ :=
  726 / 999

/-- Theorem stating that 0.726726726... equals 242/333 -/
theorem repeating_decimal_726_eq_fraction : repeating_decimal_726 = 242 / 333 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_726_eq_fraction_l2961_296199


namespace NUMINAMATH_CALUDE_hot_drink_price_range_l2961_296154

/-- Represents the price increase in yuan -/
def price_increase : ℝ → ℝ := λ x => x

/-- Represents the new price of a hot drink in yuan -/
def new_price : ℝ → ℝ := λ x => 1.5 + price_increase x

/-- Represents the daily sales volume as a function of price increase -/
def daily_sales : ℝ → ℝ := λ x => 800 - 20 * (10 * price_increase x)

/-- Represents the daily profit as a function of price increase -/
def daily_profit : ℝ → ℝ := λ x => (new_price x - 0.9) * daily_sales x

theorem hot_drink_price_range :
  ∃ (lower upper : ℝ), lower = 1.9 ∧ upper = 4.5 ∧
  ∀ x, daily_profit x ≥ 720 ↔ new_price x ∈ Set.Icc lower upper :=
by sorry

end NUMINAMATH_CALUDE_hot_drink_price_range_l2961_296154


namespace NUMINAMATH_CALUDE_second_bell_interval_l2961_296126

def bell_intervals (x : ℕ) : List ℕ := [5, x, 11, 15]

theorem second_bell_interval (x : ℕ) :
  (∃ (k : ℕ), k > 0 ∧ k * (Nat.lcm (Nat.lcm (Nat.lcm 5 x) 11) 15) = 1320) →
  x = 8 :=
by sorry

end NUMINAMATH_CALUDE_second_bell_interval_l2961_296126


namespace NUMINAMATH_CALUDE_at_most_two_solutions_l2961_296140

theorem at_most_two_solutions (m : ℕ) : 
  ∃ (a₁ a₂ : ℤ), ∀ (a : ℤ), 
    (⌊(a : ℝ) - Real.sqrt (a : ℝ)⌋ = m) → (a = a₁ ∨ a = a₂) :=
sorry

end NUMINAMATH_CALUDE_at_most_two_solutions_l2961_296140


namespace NUMINAMATH_CALUDE_smarties_leftover_l2961_296123

theorem smarties_leftover (m : ℕ) (h : m % 11 = 5) : (4 * m) % 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_smarties_leftover_l2961_296123


namespace NUMINAMATH_CALUDE_limes_given_to_sara_l2961_296128

/-- Given that Dan initially picked some limes and gave some to Sara, 
    prove that the number of limes Dan gave to Sara is equal to 
    the difference between his initial and final number of limes. -/
theorem limes_given_to_sara 
  (initial_limes : ℕ) 
  (final_limes : ℕ) 
  (h1 : initial_limes = 9)
  (h2 : final_limes = 5) :
  initial_limes - final_limes = 4 := by
  sorry

end NUMINAMATH_CALUDE_limes_given_to_sara_l2961_296128


namespace NUMINAMATH_CALUDE_trig_identity_l2961_296142

theorem trig_identity : 
  Real.sin (63 * π / 180) * Real.cos (18 * π / 180) + 
  Real.cos (63 * π / 180) * Real.cos (108 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2961_296142


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l2961_296116

-- Define the Cartesian coordinate system
def CartesianPoint := ℝ × ℝ

-- Define the quadrants
def first_quadrant (p : CartesianPoint) : Prop := p.1 > 0 ∧ p.2 > 0
def second_quadrant (p : CartesianPoint) : Prop := p.1 < 0 ∧ p.2 > 0
def third_quadrant (p : CartesianPoint) : Prop := p.1 < 0 ∧ p.2 < 0
def fourth_quadrant (p : CartesianPoint) : Prop := p.1 > 0 ∧ p.2 < 0

-- The point in question
def point : CartesianPoint := (-5, -1)

-- The theorem to prove
theorem point_in_third_quadrant : third_quadrant point := by sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l2961_296116


namespace NUMINAMATH_CALUDE_annual_reduction_equation_l2961_296131

/-- The total cost reduction percentage over two years -/
def total_reduction : ℝ := 0.36

/-- The average annual reduction percentage -/
def x : ℝ := sorry

/-- Theorem stating the relationship between the average annual reduction and total reduction -/
theorem annual_reduction_equation : (1 - x)^2 = 1 - total_reduction := by sorry

end NUMINAMATH_CALUDE_annual_reduction_equation_l2961_296131


namespace NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l2961_296110

/-- A function f: ℝ → ℝ satisfying f(x+y) ≤ f(x²+y) for all x, y ∈ ℝ is constant. -/
theorem function_satisfying_inequality_is_constant (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) ≤ f (x^2 + y)) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c := by
  sorry

end NUMINAMATH_CALUDE_function_satisfying_inequality_is_constant_l2961_296110


namespace NUMINAMATH_CALUDE_preimage_of_4_3_l2961_296125

def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2 * p.2, 2 * p.1 - p.2)

theorem preimage_of_4_3 :
  ∃ (p : ℝ × ℝ), f p = (4, 3) ∧ p = (2, 1) := by
sorry

end NUMINAMATH_CALUDE_preimage_of_4_3_l2961_296125


namespace NUMINAMATH_CALUDE_punch_bowl_problem_l2961_296150

/-- The capacity of the punch bowl in gallons -/
def bowl_capacity : ℝ := 16

/-- The amount of punch Sally drinks in gallons -/
def sally_drinks : ℝ := 2

/-- The amount of punch Mark adds to completely fill the bowl after Sally drinks -/
def final_addition : ℝ := 12

/-- The amount of punch Mark adds after his cousin drinks -/
def mark_addition : ℝ := 12

theorem punch_bowl_problem :
  ∃ (initial_amount : ℝ),
    initial_amount ≥ 0 ∧
    initial_amount ≤ bowl_capacity ∧
    (initial_amount / 2 + mark_addition - sally_drinks + final_addition = bowl_capacity) :=
by
  sorry

#check punch_bowl_problem

end NUMINAMATH_CALUDE_punch_bowl_problem_l2961_296150


namespace NUMINAMATH_CALUDE_am_hm_difference_bound_l2961_296151

theorem am_hm_difference_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x < y) :
  ((x - y)^2) / (2*(x + y)) < ((y - x)^2) / (8*x) := by
  sorry

end NUMINAMATH_CALUDE_am_hm_difference_bound_l2961_296151


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2961_296189

theorem tan_alpha_plus_pi_fourth (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  Real.tan (α + π/4) = 3/22 := by sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l2961_296189


namespace NUMINAMATH_CALUDE_flour_scoops_l2961_296121

/-- Given a bag of flour, the amount needed for a recipe, and the size of a measuring cup,
    calculate the number of scoops to remove from the bag. -/
def scoop_count (bag_size : ℚ) (recipe_amount : ℚ) (measure_size : ℚ) : ℚ :=
  (bag_size - recipe_amount) / measure_size

theorem flour_scoops :
  let bag_size : ℚ := 8
  let recipe_amount : ℚ := 6
  let measure_size : ℚ := 1/4
  scoop_count bag_size recipe_amount measure_size = 8 := by sorry

end NUMINAMATH_CALUDE_flour_scoops_l2961_296121


namespace NUMINAMATH_CALUDE_condition_relationship_l2961_296138

theorem condition_relationship (p q : Prop) 
  (h : (¬p → q) ∧ ¬(q → ¬p)) : 
  (p → ¬q) ∧ ¬(¬q → p) := by
sorry

end NUMINAMATH_CALUDE_condition_relationship_l2961_296138


namespace NUMINAMATH_CALUDE_summer_lecture_team_selection_probability_l2961_296176

/-- Represents the probability of a teacher being selected for the summer lecture team -/
def selection_probability (total : ℕ) (eliminated : ℕ) (team_size : ℕ) : ℚ :=
  team_size / (total - eliminated)

theorem summer_lecture_team_selection_probability :
  selection_probability 118 6 16 = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_summer_lecture_team_selection_probability_l2961_296176


namespace NUMINAMATH_CALUDE_election_win_margin_l2961_296104

theorem election_win_margin (total_votes : ℕ) (winner_votes : ℕ) : 
  winner_votes = (62 * total_votes) / 100 →
  winner_votes = 1054 →
  winner_votes - ((38 * total_votes) / 100) = 408 :=
by
  sorry

end NUMINAMATH_CALUDE_election_win_margin_l2961_296104


namespace NUMINAMATH_CALUDE_min_value_theorem_l2961_296120

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 6) :
  1 / (a + 2) + 2 / (b + 1) ≥ 9 / 10 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2 * b₀ = 6 ∧ 1 / (a₀ + 2) + 2 / (b₀ + 1) = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2961_296120


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l2961_296129

/-- Given a triangle ABC where:
  * The side opposite to angle A is 2
  * The side opposite to angle B is √2
  * Angle A measures 45°
Prove that angle B measures 30° -/
theorem triangle_angle_calculation (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  b = Real.sqrt 2 →
  A = π / 4 →
  B = π / 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l2961_296129


namespace NUMINAMATH_CALUDE_not_always_y_equals_a_when_x_zero_l2961_296130

/-- Linear regression model -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope
  x_bar : ℝ  -- mean of x
  y_bar : ℝ  -- mean of y

/-- Predicted y value for a given x -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.b * x + model.a

/-- The regression line passes through the point (x_bar, y_bar) -/
axiom passes_through_mean (model : LinearRegression) :
  predict model model.x_bar = model.y_bar

/-- b represents the average change in y for a unit increase in x -/
axiom slope_interpretation (model : LinearRegression) (x₁ x₂ : ℝ) :
  predict model x₂ - predict model x₁ = model.b * (x₂ - x₁)

/-- Sample data point -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Theorem: It is not necessarily true that y = a when x = 0 in the sample data -/
theorem not_always_y_equals_a_when_x_zero (model : LinearRegression) :
  ∃ (data : DataPoint), data.x = 0 ∧ data.y ≠ model.a :=
sorry

end NUMINAMATH_CALUDE_not_always_y_equals_a_when_x_zero_l2961_296130


namespace NUMINAMATH_CALUDE_average_weight_increase_l2961_296167

/-- Theorem: Increase in average weight when replacing a person in a group -/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 6 * initial_average
  let final_total := initial_total - 75 + 102
  let final_average := final_total / 6
  final_average - initial_average = 4.5 := by
sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2961_296167


namespace NUMINAMATH_CALUDE_basketball_team_composition_l2961_296152

-- Define the number of classes
def num_classes : ℕ := 8

-- Define the total number of players
def total_players : ℕ := 10

-- Define the function to calculate the number of composition methods
def composition_methods (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) k

-- Theorem statement
theorem basketball_team_composition :
  composition_methods (num_classes) (total_players - num_classes) = 36 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_composition_l2961_296152


namespace NUMINAMATH_CALUDE_nancy_homework_time_l2961_296182

/-- The time required to finish all problems -/
def time_to_finish (math_problems : Float) (spelling_problems : Float) (problems_per_hour : Float) : Float :=
  (math_problems + spelling_problems) / problems_per_hour

/-- Proof that Nancy will take 4.0 hours to finish all problems -/
theorem nancy_homework_time : 
  time_to_finish 17.0 15.0 8.0 = 4.0 := by
  sorry

end NUMINAMATH_CALUDE_nancy_homework_time_l2961_296182


namespace NUMINAMATH_CALUDE_cades_remaining_marbles_l2961_296197

/-- Proves that Cade has 79 marbles left after giving away 8 marbles from his initial 87 marbles. -/
theorem cades_remaining_marbles (initial_marbles : ℕ) (marbles_given_away : ℕ) 
  (h1 : initial_marbles = 87) 
  (h2 : marbles_given_away = 8) : 
  initial_marbles - marbles_given_away = 79 := by
  sorry

end NUMINAMATH_CALUDE_cades_remaining_marbles_l2961_296197


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l2961_296193

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_formula
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_a2 : a 2 = 9)
  (h_a4 : a 4 = 4) :
  ∀ n : ℕ, a n = 9 * (2/3)^(n - 2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l2961_296193


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l2961_296170

theorem price_reduction_percentage (initial_price final_price : ℝ) 
  (h1 : initial_price = 25)
  (h2 : final_price = 16)
  (h3 : ∃ x : ℝ, 0 < x ∧ x < 1 ∧ initial_price * (1 - x)^2 = final_price) :
  ∃ x : ℝ, x = 0.2 ∧ initial_price * (1 - x)^2 = final_price :=
by sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l2961_296170
