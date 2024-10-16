import Mathlib

namespace NUMINAMATH_CALUDE_line_parameterization_l1826_182672

/-- Given a line y = 2x - 17 parameterized by (x,y) = (f(t), 20t - 12), prove that f(t) = 10t + 5/2 -/
theorem line_parameterization (f : ℝ → ℝ) : 
  (∀ t : ℝ, 20*t - 12 = 2*(f t) - 17) → 
  (∀ t : ℝ, f t = 10*t + 5/2) := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l1826_182672


namespace NUMINAMATH_CALUDE_kathryn_picked_two_more_l1826_182624

/-- The number of pints of blueberries picked by Annie, Kathryn, and Ben -/
structure BlueberryPicking where
  annie : ℕ
  kathryn : ℕ
  ben : ℕ

/-- The conditions of the blueberry picking problem -/
def BlueberryPickingConditions (p : BlueberryPicking) : Prop :=
  p.annie = 8 ∧
  p.kathryn > p.annie ∧
  p.ben = p.kathryn - 3 ∧
  p.annie + p.kathryn + p.ben = 25

/-- The theorem stating that Kathryn picked 2 more pints than Annie -/
theorem kathryn_picked_two_more (p : BlueberryPicking) 
  (h : BlueberryPickingConditions p) : p.kathryn - p.annie = 2 := by
  sorry

end NUMINAMATH_CALUDE_kathryn_picked_two_more_l1826_182624


namespace NUMINAMATH_CALUDE_unique_solution_l1826_182652

def product_of_digits (n : ℕ) : ℕ := sorry

theorem unique_solution : ∃! x : ℕ+, 
  (x : ℕ) > 0 ∧ product_of_digits x = x^2 - 10*x - 22 ∧ x = 12 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1826_182652


namespace NUMINAMATH_CALUDE_ratio_is_three_halves_l1826_182660

/-- Represents a rectangular parallelepiped with dimensions a, b, and c -/
structure RectangularParallelepiped (α : Type*) [LinearOrderedField α] where
  a : α
  b : α
  c : α

/-- The ratio of the sum of squares of sides of triangle KLM to the square of the parallelepiped's diagonal -/
def triangle_to_diagonal_ratio {α : Type*} [LinearOrderedField α] (p : RectangularParallelepiped α) : α :=
  (3 : α) / 2

/-- Theorem stating that the ratio is always 3/2 for any rectangular parallelepiped -/
theorem ratio_is_three_halves {α : Type*} [LinearOrderedField α] (p : RectangularParallelepiped α) :
  triangle_to_diagonal_ratio p = (3 : α) / 2 := by
  sorry

#check ratio_is_three_halves

end NUMINAMATH_CALUDE_ratio_is_three_halves_l1826_182660


namespace NUMINAMATH_CALUDE_seventh_oblong_number_l1826_182697

/-- Defines an oblong number for a given positive integer n -/
def oblong_number (n : ℕ) : ℕ := n * (n + 1)

/-- Theorem stating that the 7th oblong number is 56 -/
theorem seventh_oblong_number : oblong_number 7 = 56 := by
  sorry

end NUMINAMATH_CALUDE_seventh_oblong_number_l1826_182697


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l1826_182604

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 1 ∧ 
   ∀ y : ℝ, y^2 - 4*y + m = 1 → y = x) ↔ 
  m = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l1826_182604


namespace NUMINAMATH_CALUDE_unique_m_exists_l1826_182689

theorem unique_m_exists : ∃! m : ℤ,
  30 ≤ m ∧ m ≤ 80 ∧
  ∃ k : ℤ, m = 6 * k ∧
  m % 8 = 2 ∧
  m % 5 = 2 ∧
  m = 42 := by sorry

end NUMINAMATH_CALUDE_unique_m_exists_l1826_182689


namespace NUMINAMATH_CALUDE_arithmetic_sequence_index_l1826_182675

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_index :
  ∀ n : ℕ,
  arithmetic_sequence 1 3 n = 2014 → n = 672 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_index_l1826_182675


namespace NUMINAMATH_CALUDE_negative_one_to_zero_equals_one_l1826_182679

theorem negative_one_to_zero_equals_one :
  (-1 : ℝ) ^ (0 : ℝ) = 1 := by sorry

end NUMINAMATH_CALUDE_negative_one_to_zero_equals_one_l1826_182679


namespace NUMINAMATH_CALUDE_portias_school_students_l1826_182613

theorem portias_school_students (portia_students lara_students : ℕ) 
  (h1 : portia_students = 4 * lara_students)
  (h2 : portia_students + lara_students = 2500) : 
  portia_students = 2000 := by
  sorry

end NUMINAMATH_CALUDE_portias_school_students_l1826_182613


namespace NUMINAMATH_CALUDE_aquarium_animals_l1826_182690

theorem aquarium_animals (num_aquariums : ℕ) (total_animals : ℕ) 
  (h1 : num_aquariums = 26)
  (h2 : total_animals = 52)
  (h3 : ∃ (animals_per_aquarium : ℕ), 
    animals_per_aquarium > 1 ∧ 
    animals_per_aquarium % 2 = 1 ∧
    num_aquariums * animals_per_aquarium = total_animals) :
  ∃ (animals_per_aquarium : ℕ), 
    animals_per_aquarium = 13 ∧
    animals_per_aquarium > 1 ∧ 
    animals_per_aquarium % 2 = 1 ∧
    num_aquariums * animals_per_aquarium = total_animals :=
by sorry

end NUMINAMATH_CALUDE_aquarium_animals_l1826_182690


namespace NUMINAMATH_CALUDE_volume_removed_tetrahedra_l1826_182630

/-- The volume of tetrahedra removed from a cube when slicing corners to form octagonal faces -/
theorem volume_removed_tetrahedra (cube_edge : ℝ) (h : cube_edge = 2) :
  let octagon_side := 2 * (Real.sqrt 2 - 1)
  let tetrahedron_height := 2 / Real.sqrt 2
  let base_area := 2 * (3 - 2 * Real.sqrt 2)
  let single_tetrahedron_volume := (1 / 3) * base_area * tetrahedron_height
  8 * single_tetrahedron_volume = (32 * (3 - 2 * Real.sqrt 2)) / 3 :=
by sorry

end NUMINAMATH_CALUDE_volume_removed_tetrahedra_l1826_182630


namespace NUMINAMATH_CALUDE_fencing_cost_l1826_182618

/-- Given a rectangular field with sides in ratio 3:4 and area 9408 sq. m,
    prove that the cost of fencing at 25 paise per metre is 98 rupees. -/
theorem fencing_cost (length width : ℝ) (area perimeter cost_per_metre total_cost : ℝ) : 
  length / width = 3 / 4 →
  area = 9408 →
  area = length * width →
  perimeter = 2 * (length + width) →
  cost_per_metre = 25 / 100 →
  total_cost = perimeter * cost_per_metre →
  total_cost = 98 := by
sorry

end NUMINAMATH_CALUDE_fencing_cost_l1826_182618


namespace NUMINAMATH_CALUDE_no_blue_frogs_l1826_182653

-- Define the types of frogs
inductive FrogColor
| Red
| Blue
| Other

-- Define the type of islanders
inductive IslanderType
| TruthTeller
| Liar

-- Define the islanders
structure Islander where
  name : String
  type : IslanderType
  color : FrogColor

-- Define the island
structure Island where
  inhabitants : List Islander
  has_blue_frogs : Bool

-- Define the statements made by the islanders
def bre_statement (island : Island) : Prop :=
  ¬ island.has_blue_frogs

def ke_statement (bre : Islander) : Prop :=
  bre.type = IslanderType.Liar ∧ bre.color = FrogColor.Blue

def keks_statement (bre : Islander) : Prop :=
  bre.type = IslanderType.Liar ∧ bre.color = FrogColor.Red

-- Theorem: Given the statements and the fact that islanders are either truth-tellers or liars,
-- there are no blue frogs on the island
theorem no_blue_frogs (island : Island) (bre ke keks : Islander) :
  (bre.type = IslanderType.TruthTeller ∨ bre.type = IslanderType.Liar) →
  (ke.type = IslanderType.TruthTeller ∨ ke.type = IslanderType.Liar) →
  (keks.type = IslanderType.TruthTeller ∨ keks.type = IslanderType.Liar) →
  (bre.type = IslanderType.TruthTeller ↔ bre_statement island) →
  (ke.type = IslanderType.TruthTeller ↔ ke_statement bre) →
  (keks.type = IslanderType.TruthTeller ↔ keks_statement bre) →
  ¬ island.has_blue_frogs :=
sorry

end NUMINAMATH_CALUDE_no_blue_frogs_l1826_182653


namespace NUMINAMATH_CALUDE_unique_p_q_for_inequality_l1826_182600

theorem unique_p_q_for_inequality :
  ∀ (p q : ℝ),
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |Real.sqrt (1 - x^2) - p*x - q| ≤ (Real.sqrt 2 - 1) / 2) →
    p = -1 ∧ q = (1 + Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_p_q_for_inequality_l1826_182600


namespace NUMINAMATH_CALUDE_inscribed_sphere_theorem_l1826_182683

/-- A truncated triangular pyramid with an inscribed sphere -/
structure TruncatedPyramid where
  /-- Height of the pyramid -/
  h : ℝ
  /-- Radius of the circle described around the first base -/
  R₁ : ℝ
  /-- Radius of the circle described around the second base -/
  R₂ : ℝ
  /-- Distance between the center of the first base circle and the point where the sphere touches it -/
  O₁T₁ : ℝ
  /-- Distance between the center of the second base circle and the point where the sphere touches it -/
  O₂T₂ : ℝ
  /-- All lengths are positive -/
  h_pos : 0 < h
  R₁_pos : 0 < R₁
  R₂_pos : 0 < R₂
  O₁T₁_pos : 0 < O₁T₁
  O₂T₂_pos : 0 < O₂T₂
  /-- The sphere touches the bases inside the circles -/
  O₁T₁_le_R₁ : O₁T₁ ≤ R₁
  O₂T₂_le_R₂ : O₂T₂ ≤ R₂

/-- The main theorem about the inscribed sphere in a truncated triangular pyramid -/
theorem inscribed_sphere_theorem (p : TruncatedPyramid) :
    p.R₁ * p.R₂ * p.h^2 = (p.R₁^2 - p.O₁T₁^2) * (p.R₂^2 - p.O₂T₂^2) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_theorem_l1826_182683


namespace NUMINAMATH_CALUDE_power_one_sixth_equals_one_l1826_182611

def is_greatest_power_of_two_factor (a : ℕ) : Prop :=
  2^a ∣ 180 ∧ ∀ k > a, ¬(2^k ∣ 180)

def is_greatest_power_of_three_factor (b : ℕ) : Prop :=
  3^b ∣ 180 ∧ ∀ k > b, ¬(3^k ∣ 180)

theorem power_one_sixth_equals_one (a b : ℕ) 
  (h1 : is_greatest_power_of_two_factor a) 
  (h2 : is_greatest_power_of_three_factor b) : 
  (1/6 : ℚ)^(b - a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_one_sixth_equals_one_l1826_182611


namespace NUMINAMATH_CALUDE_diana_bike_time_l1826_182645

/-- Proves that Diana will take 6 hours to get home given the specified conditions -/
theorem diana_bike_time : 
  let total_distance : ℝ := 10
  let initial_speed : ℝ := 3
  let initial_time : ℝ := 2
  let tired_speed : ℝ := 1
  let initial_distance := initial_speed * initial_time
  let remaining_distance := total_distance - initial_distance
  let tired_time := remaining_distance / tired_speed
  initial_time + tired_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_diana_bike_time_l1826_182645


namespace NUMINAMATH_CALUDE_problem_distribution_l1826_182655

def num_problems : ℕ := 5
def num_friends : ℕ := 12

theorem problem_distribution :
  (num_friends ^ num_problems : ℕ) = 248832 :=
by sorry

end NUMINAMATH_CALUDE_problem_distribution_l1826_182655


namespace NUMINAMATH_CALUDE_pizza_toppings_l1826_182629

/-- Given a pizza with 24 slices, where 15 slices have pepperoni and 14 slices have mushrooms,
    prove that 5 slices have both pepperoni and mushrooms. -/
theorem pizza_toppings (total : ℕ) (pepperoni : ℕ) (mushrooms : ℕ) 
  (h_total : total = 24)
  (h_pepperoni : pepperoni = 15)
  (h_mushrooms : mushrooms = 14) :
  pepperoni + mushrooms - total = 5 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l1826_182629


namespace NUMINAMATH_CALUDE_revenue_change_after_price_and_sales_change_l1826_182664

theorem revenue_change_after_price_and_sales_change 
  (original_price original_quantity : ℝ) 
  (price_increase_percent : ℝ) 
  (sales_decrease_percent : ℝ) : 
  price_increase_percent = 60 → 
  sales_decrease_percent = 35 → 
  let new_price := original_price * (1 + price_increase_percent / 100)
  let new_quantity := original_quantity * (1 - sales_decrease_percent / 100)
  let original_revenue := original_price * original_quantity
  let new_revenue := new_price * new_quantity
  (new_revenue - original_revenue) / original_revenue * 100 = 4 := by
sorry

end NUMINAMATH_CALUDE_revenue_change_after_price_and_sales_change_l1826_182664


namespace NUMINAMATH_CALUDE_max_xy_value_l1826_182638

theorem max_xy_value (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : 2 * x + y = 1) :
  x * y ≤ 1 / 8 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ x * y = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_max_xy_value_l1826_182638


namespace NUMINAMATH_CALUDE_reflections_composition_is_translation_l1826_182661

/-- Four distinct points on a circle -/
structure CirclePoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D
  on_circle : ∃ (center : ℝ × ℝ) (radius : ℝ), 
    (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2 ∧
    (B.1 - center.1)^2 + (B.2 - center.2)^2 = radius^2 ∧
    (C.1 - center.1)^2 + (C.2 - center.2)^2 = radius^2 ∧
    (D.1 - center.1)^2 + (D.2 - center.2)^2 = radius^2

/-- Reflection across a line defined by two points -/
def reflect (p q : ℝ × ℝ) (x : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Translation of a point -/
def translate (v : ℝ × ℝ) (x : ℝ × ℝ) : ℝ × ℝ := (x.1 + v.1, x.2 + v.2)

/-- The main theorem stating that the composition of reflections is a translation -/
theorem reflections_composition_is_translation (points : CirclePoints) :
  ∃ (v : ℝ × ℝ), ∀ (x : ℝ × ℝ),
    reflect points.D points.A (reflect points.C points.D (reflect points.B points.C (reflect points.A points.B x))) = translate v x :=
sorry

end NUMINAMATH_CALUDE_reflections_composition_is_translation_l1826_182661


namespace NUMINAMATH_CALUDE_batsmans_average_increase_l1826_182676

theorem batsmans_average_increase 
  (score_17th : ℕ) 
  (average_after_17th : ℚ) 
  (h1 : score_17th = 66) 
  (h2 : average_after_17th = 18) : 
  average_after_17th - (((17 : ℕ) * average_after_17th - score_17th) / 16 : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_batsmans_average_increase_l1826_182676


namespace NUMINAMATH_CALUDE_ahmed_goats_l1826_182694

/-- Given information about goats owned by Adam, Andrew, and Ahmed -/
theorem ahmed_goats (adam : ℕ) (andrew : ℕ) (ahmed : ℕ) : 
  adam = 7 →
  andrew = 2 * adam + 5 →
  ahmed = andrew - 6 →
  ahmed = 13 := by sorry

end NUMINAMATH_CALUDE_ahmed_goats_l1826_182694


namespace NUMINAMATH_CALUDE_preimage_of_neg_three_two_l1826_182699

def f (x y : ℝ) : ℝ × ℝ := (x * y, x + y)

theorem preimage_of_neg_three_two :
  {p : ℝ × ℝ | f p.1 p.2 = (-3, 2)} = {(3, -1), (-1, 3)} := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_neg_three_two_l1826_182699


namespace NUMINAMATH_CALUDE_solve_percentage_equation_l1826_182656

theorem solve_percentage_equation (x : ℝ) : 0.60 * x = (1 / 3) * x + 110 → x = 412.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_percentage_equation_l1826_182656


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l1826_182632

/-- Given two squares ABCD and CEFG with the same side length sharing a common vertex C,
    the ratio of the shaded area to the area of square ABCD is 2 - √2 -/
theorem shaded_area_ratio (l : ℝ) (h : l > 0) : 
  let diagonal := l * Real.sqrt 2
  let small_side := diagonal - l
  let shaded_area := l^2 - 2 * (1/2 * small_side * l)
  shaded_area / l^2 = 2 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_l1826_182632


namespace NUMINAMATH_CALUDE_set_intersection_empty_implies_a_range_l1826_182642

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | |x - a| < 1}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 5}

-- State the theorem
theorem set_intersection_empty_implies_a_range (a : ℝ) : 
  A a ∩ B = ∅ → a ≤ 0 ∨ a ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_empty_implies_a_range_l1826_182642


namespace NUMINAMATH_CALUDE_smaller_pond_green_percentage_is_twenty_percent_l1826_182608

/-- Represents the duck population in two ponds -/
structure DuckPonds where
  total_ducks : ℕ
  smaller_pond_ducks : ℕ
  larger_pond_ducks : ℕ
  larger_pond_green_percentage : ℚ
  total_green_percentage : ℚ

/-- Calculates the percentage of green ducks in the smaller pond -/
def smaller_pond_green_percentage (ponds : DuckPonds) : ℚ :=
  let total_green_ducks := ponds.total_ducks * ponds.total_green_percentage
  let larger_pond_green_ducks := ponds.larger_pond_ducks * ponds.larger_pond_green_percentage
  let smaller_pond_green_ducks := total_green_ducks - larger_pond_green_ducks
  smaller_pond_green_ducks / ponds.smaller_pond_ducks

/-- Theorem: The percentage of green ducks in the smaller pond is 20% -/
theorem smaller_pond_green_percentage_is_twenty_percent (ponds : DuckPonds) 
  (h1 : ponds.total_ducks = 100)
  (h2 : ponds.smaller_pond_ducks = 45)
  (h3 : ponds.larger_pond_ducks = 55)
  (h4 : ponds.larger_pond_green_percentage = 2/5)
  (h5 : ponds.total_green_percentage = 31/100) :
  smaller_pond_green_percentage ponds = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_smaller_pond_green_percentage_is_twenty_percent_l1826_182608


namespace NUMINAMATH_CALUDE_original_ratio_l1826_182688

theorem original_ratio (x y : ℕ) (h1 : y = 16) (h2 : x + 12 = y) :
  ∃ (a b : ℕ), a = 1 ∧ b = 4 ∧ x * b = y * a :=
by sorry

end NUMINAMATH_CALUDE_original_ratio_l1826_182688


namespace NUMINAMATH_CALUDE_a_2_value_a_n_formula_l1826_182633

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

axiom a_1 : sequence_a 1 = 1

axiom relation (n : ℕ) (hn : n > 0) : 
  2 * S n / n = sequence_a (n + 1) - (1/3) * n^2 - n - 2/3

theorem a_2_value : sequence_a 2 = 4 := by sorry

theorem a_n_formula (n : ℕ) (hn : n > 0) : sequence_a n = n^2 := by sorry

end NUMINAMATH_CALUDE_a_2_value_a_n_formula_l1826_182633


namespace NUMINAMATH_CALUDE_max_sum_of_other_roots_l1826_182670

/-- Given a polynomial x^3 - kx^2 + 20x - 15 with 3 roots, one of which is 3,
    the sum of the other two roots is at most 5. -/
theorem max_sum_of_other_roots (k : ℝ) :
  let p : ℝ → ℝ := λ x => x^3 - k*x^2 + 20*x - 15
  ∃ (r₁ r₂ : ℝ), (p 3 = 0 ∧ p r₁ = 0 ∧ p r₂ = 0) → r₁ + r₂ ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_other_roots_l1826_182670


namespace NUMINAMATH_CALUDE_prob_6_to_7_l1826_182641

-- Define a normally distributed random variable
def X : Real → Real := sorry

-- Define the probability density function for X
def pdf (x : Real) : Real := sorry

-- Define the cumulative distribution function for X
def cdf (x : Real) : Real := sorry

-- Given probabilities
axiom prob_1sigma : (cdf 6 - cdf 4) = 0.6826
axiom prob_2sigma : (cdf 7 - cdf 3) = 0.9544
axiom prob_3sigma : (cdf 8 - cdf 2) = 0.9974

-- The statement to prove
theorem prob_6_to_7 : (cdf 7 - cdf 6) = 0.1359 := by sorry

end NUMINAMATH_CALUDE_prob_6_to_7_l1826_182641


namespace NUMINAMATH_CALUDE_weight_loss_difference_l1826_182650

/-- Given the weight loss of three people, prove how much more Veronica lost compared to Seth. -/
theorem weight_loss_difference (seth_loss jerome_loss veronica_loss total_loss : ℝ) : 
  seth_loss = 17.5 →
  jerome_loss = 3 * seth_loss →
  total_loss = 89 →
  total_loss = seth_loss + jerome_loss + veronica_loss →
  veronica_loss > seth_loss →
  veronica_loss - seth_loss = 1.5 := by
  sorry

#check weight_loss_difference

end NUMINAMATH_CALUDE_weight_loss_difference_l1826_182650


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1826_182616

/-- Proves that the speed of a train is approximately 80 km/hr given specific conditions -/
theorem train_speed_calculation (train_length : Real) (crossing_time : Real) (man_speed_kmh : Real) :
  train_length = 220 →
  crossing_time = 10.999120070394369 →
  man_speed_kmh = 8 →
  ∃ (train_speed_kmh : Real), abs (train_speed_kmh - 80) < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_calculation_l1826_182616


namespace NUMINAMATH_CALUDE_absolute_difference_60th_terms_l1826_182631

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem absolute_difference_60th_terms : 
  let C := arithmetic_sequence 25 15
  let D := arithmetic_sequence 40 (-15)
  |C 60 - D 60| = 1755 := by
sorry

end NUMINAMATH_CALUDE_absolute_difference_60th_terms_l1826_182631


namespace NUMINAMATH_CALUDE_product_difference_squared_l1826_182639

theorem product_difference_squared : 2012 * 2016 - 2014^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_squared_l1826_182639


namespace NUMINAMATH_CALUDE_min_cone_volume_with_sphere_l1826_182677

/-- The minimum volume of a cone containing a sphere of radius 1 that touches the base of the cone -/
theorem min_cone_volume_with_sphere (h r : ℝ) : 
  h > 0 → r > 0 → (1 : ℝ) ≤ h →
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ x^2 + (y - 1)^2 = r^2 ∧ y = h - 1) →
  (1/3 * π * r^2 * h) ≥ 8*π/3 :=
by sorry

end NUMINAMATH_CALUDE_min_cone_volume_with_sphere_l1826_182677


namespace NUMINAMATH_CALUDE_minimum_correct_problems_l1826_182602

def total_problems : ℕ := 25
def attempted_problems : ℕ := 21
def unanswered_problems : ℕ := total_problems - attempted_problems
def correct_points : ℕ := 7
def incorrect_points : ℤ := -1
def unanswered_points : ℕ := 2
def minimum_score : ℕ := 120

def score (correct : ℕ) : ℤ :=
  (correct * correct_points : ℤ) + 
  ((attempted_problems - correct) * incorrect_points) + 
  (unanswered_problems * unanswered_points)

theorem minimum_correct_problems : 
  ∀ x : ℕ, x ≥ 17 ↔ score x ≥ minimum_score :=
by sorry

end NUMINAMATH_CALUDE_minimum_correct_problems_l1826_182602


namespace NUMINAMATH_CALUDE_grocery_shop_sales_l1826_182681

/-- A grocery shop's sales problem -/
theorem grocery_shop_sales 
  (sales_4_months : List ℕ)
  (average_6_months : ℕ)
  (sale_6th_month : ℕ)
  (h1 : sales_4_months = [6735, 6927, 7230, 6562])
  (h2 : average_6_months = 6500)
  (h3 : sale_6th_month = 4691)
  : ∃ (sale_3rd_month : ℕ), 
    sale_3rd_month = 6 * average_6_months - (sales_4_months.sum + sale_6th_month) :=
by
  sorry

#check grocery_shop_sales

end NUMINAMATH_CALUDE_grocery_shop_sales_l1826_182681


namespace NUMINAMATH_CALUDE_function_overlap_with_inverse_l1826_182665

theorem function_overlap_with_inverse (a b c d : ℝ) (h1 : a ≠ 0 ∨ c ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ (a * x + b) / (c * x + d)
  (∀ x, f (f x) = x) →
  ((a + d = 0 ∧ ∃ k, f = λ x ↦ (k * x + b) / (c * x - k)) ∨ f = id) :=
by sorry

end NUMINAMATH_CALUDE_function_overlap_with_inverse_l1826_182665


namespace NUMINAMATH_CALUDE_remaining_wire_length_l1826_182636

/-- Given a wire of length 60 cm and a square with side length 9 cm made from this wire,
    the remaining wire length is 24 cm. -/
theorem remaining_wire_length (total_wire : ℝ) (square_side : ℝ) (remaining_wire : ℝ) :
  total_wire = 60 ∧ square_side = 9 →
  remaining_wire = total_wire - 4 * square_side →
  remaining_wire = 24 := by
sorry

end NUMINAMATH_CALUDE_remaining_wire_length_l1826_182636


namespace NUMINAMATH_CALUDE_min_value_of_sum_perpendicular_vectors_l1826_182643

/-- Given vectors a = (x, -1) and b = (y, 2) where a ⊥ b, the minimum value of |a + b| is 3 -/
theorem min_value_of_sum_perpendicular_vectors 
  (x y : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (x, -1)) 
  (hb : b = (y, 2)) 
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : 
  (∀ (x' y' : ℝ) (a' b' : ℝ × ℝ), 
    a' = (x', -1) → b' = (y', 2) → a'.1 * b'.1 + a'.2 * b'.2 = 0 → 
    Real.sqrt ((a'.1 + b'.1)^2 + (a'.2 + b'.2)^2) ≥ 3) ∧ 
  (∃ (x' y' : ℝ) (a' b' : ℝ × ℝ), 
    a' = (x', -1) ∧ b' = (y', 2) ∧ a'.1 * b'.1 + a'.2 * b'.2 = 0 ∧
    Real.sqrt ((a'.1 + b'.1)^2 + (a'.2 + b'.2)^2) = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_perpendicular_vectors_l1826_182643


namespace NUMINAMATH_CALUDE_cos_negative_sixty_degrees_l1826_182678

theorem cos_negative_sixty_degrees : Real.cos (-(60 * π / 180)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_sixty_degrees_l1826_182678


namespace NUMINAMATH_CALUDE_non_sunday_avg_is_120_l1826_182666

/-- Represents a library's visitor statistics for a month. -/
structure LibraryStats where
  total_days : Nat
  sunday_count : Nat
  sunday_avg : Nat
  overall_avg : Nat

/-- Calculates the average number of visitors on non-Sunday days. -/
def non_sunday_avg (stats : LibraryStats) : Rat :=
  let non_sunday_days := stats.total_days - stats.sunday_count
  let total_visitors := stats.overall_avg * stats.total_days
  let sunday_visitors := stats.sunday_avg * stats.sunday_count
  (total_visitors - sunday_visitors) / non_sunday_days

/-- Theorem stating the average number of visitors on non-Sunday days. -/
theorem non_sunday_avg_is_120 (stats : LibraryStats) 
  (h1 : stats.total_days = 30)
  (h2 : stats.sunday_count = 5)
  (h3 : stats.sunday_avg = 150)
  (h4 : stats.overall_avg = 125) :
  non_sunday_avg stats = 120 := by
  sorry

#eval non_sunday_avg ⟨30, 5, 150, 125⟩

end NUMINAMATH_CALUDE_non_sunday_avg_is_120_l1826_182666


namespace NUMINAMATH_CALUDE_line_equation_l1826_182634

/-- A line passing through (1,2) and intersecting x^2 + y^2 = 9 with chord length 4√2 -/
def line_through_circle (l : Set (ℝ × ℝ)) : Prop :=
  ∃ (A B : ℝ × ℝ),
    (1, 2) ∈ l ∧
    A ∈ l ∧ B ∈ l ∧
    A.1^2 + A.2^2 = 9 ∧
    B.1^2 + B.2^2 = 9 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 32

/-- The equation of the line is either x = 1 or 3x - 4y + 5 = 0 -/
theorem line_equation (l : Set (ℝ × ℝ)) (h : line_through_circle l) :
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x = 1) ∨
  (∀ (x y : ℝ), (x, y) ∈ l ↔ 3*x - 4*y + 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l1826_182634


namespace NUMINAMATH_CALUDE_stewart_farm_horse_food_l1826_182623

/-- The Stewart farm problem -/
theorem stewart_farm_horse_food (sheep : ℕ) (horses : ℕ) (food_per_horse : ℕ) :
  sheep = 16 →
  7 * sheep = 2 * horses →
  food_per_horse = 230 →
  horses * food_per_horse = 12880 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_horse_food_l1826_182623


namespace NUMINAMATH_CALUDE_proportion_fourth_number_l1826_182663

theorem proportion_fourth_number (x y : ℝ) : 
  (0.75 : ℝ) / x = 5 / y → x = 1.05 → y = 7 := by sorry

end NUMINAMATH_CALUDE_proportion_fourth_number_l1826_182663


namespace NUMINAMATH_CALUDE_boy_girl_sum_equal_l1826_182667

/-- Represents a child in the line -/
inductive Child
  | Boy : Child
  | Girl : Child

/-- The line of children -/
def Line (n : ℕ) := Vector Child (2 * n)

/-- Count children to the right of a position -/
def countRight (line : Line n) (pos : Fin (2 * n)) : ℕ := sorry

/-- Count children to the left of a position -/
def countLeft (line : Line n) (pos : Fin (2 * n)) : ℕ := sorry

/-- Sum of counts for boys -/
def boySum (line : Line n) : ℕ := sorry

/-- Sum of counts for girls -/
def girlSum (line : Line n) : ℕ := sorry

/-- The main theorem: boySum equals girlSum for any valid line -/
theorem boy_girl_sum_equal (n : ℕ) (line : Line n) 
  (h : ∀ i : Fin (2 * n), (i.val < n → line.get i = Child.Boy) ∧ (i.val ≥ n → line.get i = Child.Girl)) :
  boySum line = girlSum line := by sorry

end NUMINAMATH_CALUDE_boy_girl_sum_equal_l1826_182667


namespace NUMINAMATH_CALUDE_tyler_clay_age_sum_l1826_182607

theorem tyler_clay_age_sum :
  ∀ (tyler_age clay_age : ℕ),
    tyler_age = 5 →
    tyler_age = 3 * clay_age + 1 →
    tyler_age + clay_age = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_tyler_clay_age_sum_l1826_182607


namespace NUMINAMATH_CALUDE_liams_numbers_l1826_182669

theorem liams_numbers (x y : ℤ) : 
  (3 * x + 2 * y = 75) →  -- Sum of five numbers is 75
  (x = 15) →              -- The number written three times is 15
  (x * y % 5 = 0) →       -- Product of the two numbers is a multiple of 5
  (y = 15) :=             -- The other number (written twice) is 15
by sorry

end NUMINAMATH_CALUDE_liams_numbers_l1826_182669


namespace NUMINAMATH_CALUDE_franks_age_l1826_182686

theorem franks_age (frank_age : ℕ) (gabriel_age : ℕ) : 
  gabriel_age = frank_age - 3 →
  frank_age + gabriel_age = 17 →
  frank_age = 10 := by
sorry

end NUMINAMATH_CALUDE_franks_age_l1826_182686


namespace NUMINAMATH_CALUDE_magic_box_theorem_l1826_182609

theorem magic_box_theorem (m : ℝ) : m^2 - 2*m - 1 = 2 → m = 3 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_magic_box_theorem_l1826_182609


namespace NUMINAMATH_CALUDE_unique_number_l1826_182685

theorem unique_number : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_unique_number_l1826_182685


namespace NUMINAMATH_CALUDE_highest_class_strength_l1826_182635

theorem highest_class_strength (total : ℕ) (g1 g2 g3 : ℕ) : 
  total = 333 →
  g1 + g2 + g3 = total →
  5 * g1 = 3 * g2 →
  11 * g2 = 7 * g3 →
  g1 ≤ g2 ∧ g2 ≤ g3 →
  g3 = 165 := by
sorry

end NUMINAMATH_CALUDE_highest_class_strength_l1826_182635


namespace NUMINAMATH_CALUDE_bottles_per_player_first_break_l1826_182692

/-- Proves that each player took 2 bottles during the first break of a soccer match --/
theorem bottles_per_player_first_break :
  let total_bottles : ℕ := 4 * 12  -- 4 dozen
  let num_players : ℕ := 11
  let bottles_remaining : ℕ := 15
  let bottles_end_game : ℕ := num_players * 1  -- each player takes 1 bottle at the end

  let bottles_first_break : ℕ := total_bottles - bottles_end_game - bottles_remaining

  (bottles_first_break / num_players : ℚ) = 2 := by
sorry

end NUMINAMATH_CALUDE_bottles_per_player_first_break_l1826_182692


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1826_182673

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℕ, x > 0) ↔ (∃ x : ℕ, x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1826_182673


namespace NUMINAMATH_CALUDE_quadratic_expression_minimum_l1826_182684

theorem quadratic_expression_minimum :
  ∀ x y : ℝ, 2 * x^2 + 3 * y^2 - 12 * x + 6 * y + 25 ≥ 4 ∧
  ∃ x₀ y₀ : ℝ, 2 * x₀^2 + 3 * y₀^2 - 12 * x₀ + 6 * y₀ + 25 = 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_expression_minimum_l1826_182684


namespace NUMINAMATH_CALUDE_focus_coordinates_l1826_182620

/-- Represents an ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  major_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ)
  minor_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ)

/-- Calculates the coordinates of the focus with greater y-coordinate for a given ellipse -/
def focus_with_greater_y (e : Ellipse) : ℝ × ℝ :=
  sorry

/-- Theorem stating that for the given ellipse, the focus with greater y-coordinate is at (0, √5/2) -/
theorem focus_coordinates (e : Ellipse) 
  (h1 : e.center = (0, 0))
  (h2 : e.major_axis_endpoints = ((0, 3), (0, -3)))
  (h3 : e.minor_axis_endpoints = ((2, 0), (-2, 0))) :
  focus_with_greater_y e = (0, Real.sqrt 5 / 2) :=
sorry

end NUMINAMATH_CALUDE_focus_coordinates_l1826_182620


namespace NUMINAMATH_CALUDE_max_quotient_bound_l1826_182621

theorem max_quotient_bound (a b : ℝ) 
  (ha : 400 ≤ a ∧ a ≤ 800)
  (hb : 400 ≤ b ∧ b ≤ 1600)
  (hab : a + b ≤ 2000) :
  b / a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_quotient_bound_l1826_182621


namespace NUMINAMATH_CALUDE_rectangular_wall_area_l1826_182648

theorem rectangular_wall_area : 
  ∀ (width length area : ℝ),
    width = 5.4 →
    length = 2.5 →
    area = width * length →
    area = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_wall_area_l1826_182648


namespace NUMINAMATH_CALUDE_max_value_of_z_l1826_182601

theorem max_value_of_z (x y : ℝ) (h1 : |x| + |y| ≤ 4) (h2 : 2*x + y - 4 ≤ 0) :
  ∃ (z : ℝ), z = 2*x - y ∧ z ≤ 20/3 ∧ ∃ (x' y' : ℝ), |x'| + |y'| ≤ 4 ∧ 2*x' + y' - 4 ≤ 0 ∧ 2*x' - y' = 20/3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_z_l1826_182601


namespace NUMINAMATH_CALUDE_candidate_count_l1826_182605

theorem candidate_count (selection_ways : ℕ) : 
  (selection_ways = 90) → 
  (∃ n : ℕ, n * (n - 1) = selection_ways ∧ n > 1) → 
  (∃ n : ℕ, n * (n - 1) = selection_ways ∧ n = 10) :=
by
  sorry

end NUMINAMATH_CALUDE_candidate_count_l1826_182605


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l1826_182610

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 6 * Nat.factorial 6 + 2 * Nat.factorial 6 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l1826_182610


namespace NUMINAMATH_CALUDE_count_pairs_theorem_l1826_182603

def X : Finset Nat := Finset.range 10

def intersection_set : Finset Nat := {5, 7, 8}

def count_valid_pairs (X : Finset Nat) (intersection_set : Finset Nat) : Nat :=
  let remaining_elements := X \ intersection_set
  3^(remaining_elements.card) - 1

theorem count_pairs_theorem (X : Finset Nat) (intersection_set : Finset Nat) :
  X = Finset.range 10 →
  intersection_set = {5, 7, 8} →
  count_valid_pairs X intersection_set = 2186 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_theorem_l1826_182603


namespace NUMINAMATH_CALUDE_cubic_function_property_l1826_182696

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

-- State the theorem
theorem cubic_function_property (a b : ℝ) :
  f a b 4 = 0 → f a b (-4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l1826_182696


namespace NUMINAMATH_CALUDE_can_distribution_l1826_182668

theorem can_distribution (total_cans : Nat) (volume_difference : Real) (total_volume : Real) :
  total_cans = 140 →
  volume_difference = 2.5 →
  total_volume = 60 →
  ∃ (large_cans small_cans : Nat) (small_volume : Real),
    large_cans + small_cans = total_cans ∧
    large_cans * (small_volume + volume_difference) = total_volume ∧
    small_cans * small_volume = total_volume ∧
    large_cans = 20 ∧
    small_cans = 120 := by
  sorry

#check can_distribution

end NUMINAMATH_CALUDE_can_distribution_l1826_182668


namespace NUMINAMATH_CALUDE_donut_combinations_l1826_182627

/-- The number of donut types available -/
def num_types : ℕ := 5

/-- The total number of donuts to be purchased -/
def total_donuts : ℕ := 8

/-- The minimum number of type A donuts required -/
def min_type_a : ℕ := 2

/-- The minimum number of donuts required for each of the other types -/
def min_other_types : ℕ := 1

/-- The number of remaining donuts to be distributed -/
def remaining_donuts : ℕ := total_donuts - (min_type_a + (num_types - 1) * min_other_types)

/-- The number of ways to distribute the remaining donuts -/
def num_combinations : ℕ := num_types + (num_types.choose 2)

theorem donut_combinations :
  num_combinations = 15 :=
sorry

end NUMINAMATH_CALUDE_donut_combinations_l1826_182627


namespace NUMINAMATH_CALUDE_boston_distance_l1826_182687

/-- The distance between Cincinnati and Atlanta in miles -/
def distance_to_atlanta : ℕ := 440

/-- The maximum distance the cyclists can bike in a day -/
def max_daily_distance : ℕ := 40

/-- The number of days it takes to reach Atlanta -/
def days_to_atlanta : ℕ := distance_to_atlanta / max_daily_distance

/-- The distance between Cincinnati and Boston in miles -/
def distance_to_boston : ℕ := days_to_atlanta * max_daily_distance

/-- Theorem stating that the distance to Boston is 440 miles -/
theorem boston_distance : distance_to_boston = 440 := by
  sorry

end NUMINAMATH_CALUDE_boston_distance_l1826_182687


namespace NUMINAMATH_CALUDE_unique_non_divisible_by_3_l1826_182626

def is_divisible_by_3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def units_digit (n : ℕ) : ℕ := n % 10

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem unique_non_divisible_by_3 :
  let numbers : List ℕ := [3543, 3555, 3567, 3573, 3581]
  ∀ n ∈ numbers, ¬(is_divisible_by_3 n) → n = 3581 ∧ units_digit n + tens_digit n = 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_non_divisible_by_3_l1826_182626


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1826_182644

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1826_182644


namespace NUMINAMATH_CALUDE_parallel_vectors_implies_m_eq_neg_one_l1826_182628

/-- Two 2D vectors are parallel if the cross product of their components is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_implies_m_eq_neg_one (m : ℝ) :
  let a : ℝ × ℝ := (m, -1)
  let b : ℝ × ℝ := (1, m + 2)
  parallel a b → m = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_implies_m_eq_neg_one_l1826_182628


namespace NUMINAMATH_CALUDE_bank_deposit_calculation_l1826_182698

/-- Calculates the total amount of principal and interest for a fixed deposit -/
def totalAmount (principal : ℝ) (rate : ℝ) (years : ℝ) : ℝ :=
  principal * (1 + rate * years)

/-- Calculates the amount left after paying interest tax -/
def amountAfterTax (totalAmount : ℝ) (principal : ℝ) (taxRate : ℝ) : ℝ :=
  totalAmount - (totalAmount - principal) * taxRate

theorem bank_deposit_calculation :
  let principal : ℝ := 1000
  let rate : ℝ := 0.0225
  let years : ℝ := 1
  let taxRate : ℝ := 0.20
  let total := totalAmount principal rate years
  let afterTax := amountAfterTax total principal taxRate
  total = 1022.5 ∧ afterTax = 1018 := by sorry

end NUMINAMATH_CALUDE_bank_deposit_calculation_l1826_182698


namespace NUMINAMATH_CALUDE_trig_identity_l1826_182658

theorem trig_identity (α : ℝ) : 
  (Real.cos (π / 2 - α / 4) - Real.sin (π / 2 - α / 4) * Real.tan (α / 8)) / 
  (Real.sin (7 * π / 2 - α / 4) + Real.sin (α / 4 - 3 * π) * Real.tan (α / 8)) = 
  -Real.tan (α / 8) := by sorry

end NUMINAMATH_CALUDE_trig_identity_l1826_182658


namespace NUMINAMATH_CALUDE_scatter_plot_correlation_l1826_182657

/-- Represents a scatter plot of two variables -/
structure ScatterPlot where
  bottomLeft : Bool
  topRight : Bool

/-- Defines positive correlation between two variables -/
def positivelyCorrelated (x y : ℝ → ℝ) : Prop :=
  ∀ a b, a < b → x a < x b ∧ y a < y b

/-- Theorem: If a scatter plot goes from bottom left to top right, 
    the variables are positively correlated -/
theorem scatter_plot_correlation (plot : ScatterPlot) (x y : ℝ → ℝ) :
  plot.bottomLeft ∧ plot.topRight → positivelyCorrelated x y := by
  sorry


end NUMINAMATH_CALUDE_scatter_plot_correlation_l1826_182657


namespace NUMINAMATH_CALUDE_proportional_function_and_point_l1826_182647

/-- A function representing the relationship between x and y -/
def f (x : ℝ) : ℝ := -2 * x + 2

theorem proportional_function_and_point (k : ℝ) :
  (∀ x y, y + 4 = k * (x - 3)) →  -- Condition 1
  (f 1 = 0) →                    -- Condition 2
  (∃ m, f (m + 1) = 2 * m) →     -- Condition 3
  (∀ x, f x = -2 * x + 2) ∧      -- Conclusion 1
  (f 1 = 0 ∧ f 2 = 0)            -- Conclusion 2 (coordinates of M)
  := by sorry

end NUMINAMATH_CALUDE_proportional_function_and_point_l1826_182647


namespace NUMINAMATH_CALUDE_angle_supplement_theorem_l1826_182674

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the complement of an angle
def complement (α : Angle) : Angle :=
  { degrees := 90 - α.degrees - 1,
    minutes := 60 - α.minutes }

-- Define the supplement of an angle
def supplement (α : Angle) : Angle :=
  { degrees := 180 - α.degrees - 1,
    minutes := 60 - α.minutes }

theorem angle_supplement_theorem (α : Angle) :
  complement α = { degrees := 54, minutes := 32 } →
  supplement α = { degrees := 144, minutes := 32 } :=
by sorry

end NUMINAMATH_CALUDE_angle_supplement_theorem_l1826_182674


namespace NUMINAMATH_CALUDE_square_sum_identity_l1826_182614

theorem square_sum_identity (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_identity_l1826_182614


namespace NUMINAMATH_CALUDE_min_value_of_f_b_minimizes_f_l1826_182606

def f (b : ℝ) : ℝ := 2 * b^2 + 8 * b - 4

theorem min_value_of_f (b : ℝ) (h : b ∈ Set.Icc (-10) 0) :
  f b ≥ f (-2) := by sorry

theorem b_minimizes_f :
  ∃ b ∈ Set.Icc (-10 : ℝ) 0, ∀ x ∈ Set.Icc (-10 : ℝ) 0, f b ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_b_minimizes_f_l1826_182606


namespace NUMINAMATH_CALUDE_square_equation_solution_l1826_182695

theorem square_equation_solution : ∃ x : ℝ, (12 - x)^2 = (x + 3)^2 ∧ x = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l1826_182695


namespace NUMINAMATH_CALUDE_square_of_cube_of_third_smallest_prime_l1826_182671

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- State the theorem
theorem square_of_cube_of_third_smallest_prime :
  (third_smallest_prime ^ 3) ^ 2 = 15625 := by
  sorry

end NUMINAMATH_CALUDE_square_of_cube_of_third_smallest_prime_l1826_182671


namespace NUMINAMATH_CALUDE_f_at_2_l1826_182617

def f (x : ℝ) : ℝ := 3 * x^3 - 5 * x + 1

theorem f_at_2 : f 2 = 15 := by sorry

end NUMINAMATH_CALUDE_f_at_2_l1826_182617


namespace NUMINAMATH_CALUDE_remainder_seven_divisors_l1826_182637

theorem remainder_seven_divisors (n : ℕ) : 
  (∃ (divisors : Finset ℕ), 
    divisors = {d : ℕ | d > 7 ∧ 54 % d = 0} ∧ 
    Finset.card divisors = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_divisors_l1826_182637


namespace NUMINAMATH_CALUDE_system_solution_l1826_182682

theorem system_solution :
  let x : ℚ := -7/3
  let y : ℚ := -1/9
  (4 * x - 3 * y = -9) ∧ (5 * x + 6 * y = -3) := by sorry

end NUMINAMATH_CALUDE_system_solution_l1826_182682


namespace NUMINAMATH_CALUDE_mans_upstream_speed_l1826_182625

/-- Calculates the upstream speed of a man given his still water speed and downstream speed. -/
def upstream_speed (still_water_speed downstream_speed : ℝ) : ℝ :=
  2 * still_water_speed - downstream_speed

/-- Theorem: Given a man's speed in still water of 40 kmph and downstream speed of 48 kmph, 
    his upstream speed is 32 kmph. -/
theorem mans_upstream_speed : 
  upstream_speed 40 48 = 32 := by
  sorry

end NUMINAMATH_CALUDE_mans_upstream_speed_l1826_182625


namespace NUMINAMATH_CALUDE_vector_computation_l1826_182619

def c : Fin 3 → ℝ := ![(-3), 5, 2]
def d : Fin 3 → ℝ := ![5, (-1), 3]

theorem vector_computation :
  (2 • c - 5 • d + c) = ![(-34), 20, (-9)] := by sorry

end NUMINAMATH_CALUDE_vector_computation_l1826_182619


namespace NUMINAMATH_CALUDE_a_positive_sufficient_not_necessary_for_a_squared_plus_a_nonnegative_l1826_182654

theorem a_positive_sufficient_not_necessary_for_a_squared_plus_a_nonnegative :
  (∀ a : ℝ, a > 0 → a^2 + a ≥ 0) ∧
  (∃ a : ℝ, a^2 + a ≥ 0 ∧ ¬(a > 0)) :=
by sorry

end NUMINAMATH_CALUDE_a_positive_sufficient_not_necessary_for_a_squared_plus_a_nonnegative_l1826_182654


namespace NUMINAMATH_CALUDE_obtuse_angle_line_range_l1826_182662

/-- The slope of a line forming an obtuse angle with the x-axis is negative -/
def obtuse_angle_slope (a : ℝ) : Prop := a^2 + 2*a < 0

/-- The range of a for a line (a^2 + 2a)x - y + 1 = 0 forming an obtuse angle -/
theorem obtuse_angle_line_range (a : ℝ) : 
  obtuse_angle_slope a ↔ -2 < a ∧ a < 0 := by sorry

end NUMINAMATH_CALUDE_obtuse_angle_line_range_l1826_182662


namespace NUMINAMATH_CALUDE_binomial_coefficient_22_15_l1826_182640

theorem binomial_coefficient_22_15 (h1 : Nat.choose 21 13 = 20349)
                                   (h2 : Nat.choose 21 14 = 11628)
                                   (h3 : Nat.choose 23 15 = 490314) :
  Nat.choose 22 15 = 458337 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_22_15_l1826_182640


namespace NUMINAMATH_CALUDE_connie_marbles_l1826_182651

/-- Calculates the number of marbles Connie has after giving some away -/
def marblesRemaining (initial : ℕ) (givenAway : ℕ) : ℕ :=
  initial - givenAway

/-- Proves that Connie has 3 marbles remaining after giving away 70 from her initial 73 -/
theorem connie_marbles : marblesRemaining 73 70 = 3 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_l1826_182651


namespace NUMINAMATH_CALUDE_equation_solution_l1826_182615

theorem equation_solution (x y z : ℝ) 
  (eq1 : 4*x - 5*y - z = 0)
  (eq2 : x + 5*y - 18*z = 0)
  (h : z ≠ 0) :
  (x^2 + 4*x*y) / (y^2 + z^2) = 3622 / 9256 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1826_182615


namespace NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l1826_182646

theorem cryptarithmetic_puzzle (T W O F U R : ℕ) : 
  (T = 9) →
  (O % 2 = 1) →
  (T + T + W + W = F * 1000 + O * 100 + U * 10 + R) →
  (T ≠ W ∧ T ≠ O ∧ T ≠ F ∧ T ≠ U ∧ T ≠ R ∧
   W ≠ O ∧ W ≠ F ∧ W ≠ U ∧ W ≠ R ∧
   O ≠ F ∧ O ≠ U ∧ O ≠ R ∧
   F ≠ U ∧ F ≠ R ∧
   U ≠ R) →
  (T < 10 ∧ W < 10 ∧ O < 10 ∧ F < 10 ∧ U < 10 ∧ R < 10) →
  W = 1 := by
sorry

end NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l1826_182646


namespace NUMINAMATH_CALUDE_problem_solution_l1826_182659

theorem problem_solution (f : ℝ → ℝ) (m : ℝ) (a b c : ℝ) : 
  (∀ x, f x = m - |x - 2|) →
  ({x | f (x + 2) ≥ 0} = Set.Icc (-1) 1) →
  (1/a + 1/(2*b) + 1/(3*c) = m) →
  (m = 1 ∧ a + 2*b + 3*c ≥ 9) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1826_182659


namespace NUMINAMATH_CALUDE_certain_number_problem_l1826_182612

theorem certain_number_problem (x : ℤ) : 17 * (x + 99) = 3111 ↔ x = 84 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1826_182612


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1826_182691

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  q ≠ 1 →
  (∀ n : ℕ, a (n + 1) = q * a n) →
  a 1 + a 2 + a 3 + a 4 + a 5 = 6 →
  a 1^2 + a 2^2 + a 3^2 + a 4^2 + a 5^2 = 18 →
  a 1 - a 2 + a 3 - a 4 + a 5 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1826_182691


namespace NUMINAMATH_CALUDE_factorization_ax2_minus_a_l1826_182622

theorem factorization_ax2_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) := by sorry

end NUMINAMATH_CALUDE_factorization_ax2_minus_a_l1826_182622


namespace NUMINAMATH_CALUDE_hiking_distance_proof_l1826_182649

def hiking_distance (total distance_car_to_stream distance_meadow_to_campsite : ℝ) : Prop :=
  ∃ distance_stream_to_meadow : ℝ,
    distance_stream_to_meadow = total - (distance_car_to_stream + distance_meadow_to_campsite) ∧
    distance_stream_to_meadow = 0.4

theorem hiking_distance_proof :
  hiking_distance 0.7 0.2 0.1 :=
by
  sorry

end NUMINAMATH_CALUDE_hiking_distance_proof_l1826_182649


namespace NUMINAMATH_CALUDE_recurring_decimal_division_l1826_182693

def repeating_decimal_to_fraction (a b c : ℕ) : ℚ :=
  (a * 1000 + b * 100 + c * 10 + a * 1 + b * (1/10) + c * (1/100)) / 999

theorem recurring_decimal_division (a b c d e f : ℕ) :
  (repeating_decimal_to_fraction a b c) / (1 + repeating_decimal_to_fraction d e f) = 714 / 419 :=
by
  sorry

end NUMINAMATH_CALUDE_recurring_decimal_division_l1826_182693


namespace NUMINAMATH_CALUDE_class_average_calculation_l1826_182680

theorem class_average_calculation (total_students : ℕ) (monday_students : ℕ) (tuesday_students : ℕ)
  (monday_average : ℚ) (tuesday_average : ℚ) :
  total_students = 28 →
  monday_students = 24 →
  tuesday_students = 4 →
  monday_average = 82/100 →
  tuesday_average = 90/100 →
  let overall_average := (monday_students * monday_average + tuesday_students * tuesday_average) / total_students
  ∃ ε > 0, |overall_average - 83/100| < ε :=
by sorry

end NUMINAMATH_CALUDE_class_average_calculation_l1826_182680
