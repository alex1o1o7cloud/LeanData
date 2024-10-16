import Mathlib

namespace NUMINAMATH_CALUDE_reinforcement_size_l2822_282248

/-- Calculates the size of a reinforcement given initial garrison size, provision duration, and new provision duration after reinforcement arrival. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) (days_before_reinforcement : ℕ) (new_duration : ℕ) : ℕ :=
  let remaining_provisions := initial_garrison * (initial_duration - days_before_reinforcement)
  (remaining_provisions / new_duration) - initial_garrison

/-- Proves that the reinforcement size is 1900 given the problem conditions. -/
theorem reinforcement_size :
  calculate_reinforcement 2000 54 15 20 = 1900 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_size_l2822_282248


namespace NUMINAMATH_CALUDE_milk_distribution_l2822_282251

/-- Given a total number of milk bottles, number of cartons, and number of bags per carton,
    calculate the number of bottles in one bag. -/
def bottles_per_bag (total_bottles : ℕ) (num_cartons : ℕ) (bags_per_carton : ℕ) : ℕ :=
  total_bottles / (num_cartons * bags_per_carton)

/-- Prove that given 180 total bottles, 3 cartons, and 4 bags per carton,
    the number of bottles in one bag is 15. -/
theorem milk_distribution :
  bottles_per_bag 180 3 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_milk_distribution_l2822_282251


namespace NUMINAMATH_CALUDE_lattice_triangle_properties_l2822_282299

/-- A lattice point in the xy-plane -/
structure LatticePoint where
  x : Int
  y : Int

/-- A triangle with vertices at lattice points -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Count of lattice points on a side (excluding endpoints) -/
def latticePointsOnSide (P Q : LatticePoint) : Nat :=
  sorry

/-- Area of a triangle with lattice point vertices -/
def triangleArea (t : LatticeTriangle) : Int :=
  sorry

theorem lattice_triangle_properties (t : LatticeTriangle) :
  (latticePointsOnSide t.A t.B % 2 = 1 ∧ latticePointsOnSide t.A t.C % 2 = 1 →
    latticePointsOnSide t.B t.C % 2 = 1) ∧
  (latticePointsOnSide t.A t.B = 3 ∧ latticePointsOnSide t.A t.C = 3 →
    ∃ k : Int, triangleArea t = 8 * k) :=
  sorry

end NUMINAMATH_CALUDE_lattice_triangle_properties_l2822_282299


namespace NUMINAMATH_CALUDE_min_integer_solution_2x_minus_1_geq_5_l2822_282276

theorem min_integer_solution_2x_minus_1_geq_5 :
  ∀ x : ℤ, (2 * x - 1 ≥ 5) → x ≥ 3 ∧ ∀ y : ℤ, (2 * y - 1 ≥ 5) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_min_integer_solution_2x_minus_1_geq_5_l2822_282276


namespace NUMINAMATH_CALUDE_lawn_length_l2822_282210

/-- The length of a rectangular lawn given specific conditions -/
theorem lawn_length (width : ℝ) (road_width : ℝ) (gravel_cost : ℝ) (total_cost : ℝ) : 
  width = 35 →
  road_width = 4 →
  gravel_cost = 0.75 →
  total_cost = 258 →
  ∃ (length : ℝ), length = 51 ∧ 
    total_cost = gravel_cost * (road_width * length + road_width * width) :=
by sorry

end NUMINAMATH_CALUDE_lawn_length_l2822_282210


namespace NUMINAMATH_CALUDE_trig_identity_l2822_282280

theorem trig_identity : 
  Real.sin (44 * π / 180) * Real.cos (14 * π / 180) - 
  Real.cos (44 * π / 180) * Real.cos (76 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2822_282280


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l2822_282241

theorem smallest_number_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < 255 → (¬(11 ∣ (y + 9)) ∨ ¬(24 ∣ (y + 9)))) ∧ 
  (11 ∣ (255 + 9)) ∧ 
  (24 ∣ (255 + 9)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l2822_282241


namespace NUMINAMATH_CALUDE_artist_paintings_l2822_282250

/-- Calculates the number of paintings an artist can make in a given number of weeks -/
def paintings_made (hours_per_week : ℕ) (hours_per_painting : ℕ) (num_weeks : ℕ) : ℕ :=
  (hours_per_week / hours_per_painting) * num_weeks

/-- Proves that an artist spending 30 hours per week painting, taking 3 hours per painting, can make 40 paintings in 4 weeks -/
theorem artist_paintings : paintings_made 30 3 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_artist_paintings_l2822_282250


namespace NUMINAMATH_CALUDE_non_negative_y_range_l2822_282295

theorem non_negative_y_range (x : Real) :
  0 ≤ x ∧ x ≤ Real.pi / 2 →
  (∃ y : Real, y = 4 * Real.cos x * Real.sin x + 2 * Real.cos x - 2 * Real.sin x - 1 ∧ y ≥ 0) ↔
  0 ≤ x ∧ x ≤ Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_non_negative_y_range_l2822_282295


namespace NUMINAMATH_CALUDE_range_when_p_true_range_when_p_and_q_true_l2822_282246

-- Define proposition p
def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 3*x + m = 0

-- Define proposition q
def is_ellipse_with_x_foci (m : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (9 - m) + y^2 / (m - 2) = 1 ∧ 
  9 - m > 0 ∧ m - 2 > 0 ∧ 9 - m > m - 2

-- Theorem 1
theorem range_when_p_true (m : ℝ) :
  has_real_roots m → m ≤ 9/4 := by sorry

-- Theorem 2
theorem range_when_p_and_q_true (m : ℝ) :
  has_real_roots m ∧ is_ellipse_with_x_foci m → 2 < m ∧ m ≤ 9/4 := by sorry

end NUMINAMATH_CALUDE_range_when_p_true_range_when_p_and_q_true_l2822_282246


namespace NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l2822_282211

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then x * Real.log x - a * x^2 else a^x

theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x y, x < y → f a x > f a y) → a ∈ Set.Icc (1/2) 1 := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l2822_282211


namespace NUMINAMATH_CALUDE_passenger_gate_probability_l2822_282230

def num_gates : ℕ := 15
def distance_between_gates : ℕ := 90
def max_walking_distance : ℕ := 360

theorem passenger_gate_probability : 
  let total_possibilities := num_gates * (num_gates - 1)
  let valid_possibilities := (
    2 * (4 + 5 + 6 + 7) +  -- Gates 1,2,3,4 and 12,13,14,15
    4 * 8 +                -- Gates 5,6,10,11
    3 * 8                  -- Gates 7,8,9
  )
  (valid_possibilities : ℚ) / total_possibilities = 10 / 21 :=
sorry

end NUMINAMATH_CALUDE_passenger_gate_probability_l2822_282230


namespace NUMINAMATH_CALUDE_paper_clip_count_l2822_282263

theorem paper_clip_count (num_boxes : ℕ) (clips_per_box : ℕ) 
  (h1 : num_boxes = 9) (h2 : clips_per_box = 9) : 
  num_boxes * clips_per_box = 81 := by
  sorry

end NUMINAMATH_CALUDE_paper_clip_count_l2822_282263


namespace NUMINAMATH_CALUDE_cube_paint_theorem_l2822_282206

/-- 
Given a cube with side length n, prove that if exactly one-third of the total number of faces 
of the n³ unit cubes (after cutting) are blue, then n = 3.
-/
theorem cube_paint_theorem (n : ℕ) (h : n > 0) : 
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → n = 3 := by
  sorry

#check cube_paint_theorem

end NUMINAMATH_CALUDE_cube_paint_theorem_l2822_282206


namespace NUMINAMATH_CALUDE_pascal_triangle_45th_number_l2822_282270

theorem pascal_triangle_45th_number (n : ℕ) (k : ℕ) : 
  n = 50 → k = 44 → Nat.choose n k = 13983816 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_45th_number_l2822_282270


namespace NUMINAMATH_CALUDE_frog_jump_probability_l2822_282228

/-- Represents a jump in a random direction -/
structure Jump where
  length : ℝ
  direction : ℝ × ℝ

/-- Represents the frog's journey -/
def FrogJourney := List Jump

/-- Calculate the final position of the frog after a series of jumps -/
def finalPosition (journey : FrogJourney) : ℝ × ℝ := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Generate a random journey for the frog -/
def randomJourney : FrogJourney := sorry

/-- Probability that the frog's final position is within 2 meters of the start -/
def probabilityWithinTwoMeters : ℝ := sorry

/-- Theorem stating the probability is approximately 1/10 -/
theorem frog_jump_probability :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |probabilityWithinTwoMeters - 1/10| < ε := by sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l2822_282228


namespace NUMINAMATH_CALUDE_shortest_distance_between_circles_l2822_282264

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles : 
  let center1 : ℝ × ℝ := (5, 3)
  let radius1 : ℝ := 12
  let center2 : ℝ × ℝ := (2, -1)
  let radius2 : ℝ := 6
  let distance_between_centers : ℝ := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  let shortest_distance : ℝ := max 0 (distance_between_centers - (radius1 + radius2))
  shortest_distance = 1 :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_between_circles_l2822_282264


namespace NUMINAMATH_CALUDE_polygon_diagonals_sides_l2822_282205

theorem polygon_diagonals_sides (n : ℕ) (h : n > 2) : 
  n * (n - 3) / 2 = 2 * n → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_sides_l2822_282205


namespace NUMINAMATH_CALUDE_sixth_term_value_l2822_282221

def sequence_property (s : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → s (n + 1) = (1 / 4) * (s n + s (n + 2))

theorem sixth_term_value (s : ℕ → ℚ) :
  sequence_property s →
  s 1 = 3 →
  s 5 = 48 →
  s 6 = 2001 / 14 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_value_l2822_282221


namespace NUMINAMATH_CALUDE_prime_factorization_sum_l2822_282285

theorem prime_factorization_sum (w x y z : ℕ) : 
  2^w * 3^x * 5^y * 7^z = 1260 → 2*w + 3*x + 5*y + 7*z = 22 := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_sum_l2822_282285


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2822_282218

theorem inequality_system_solution_set :
  let S := {x : ℝ | x + 5 < 4 ∧ (3 * x + 1) / 2 ≥ 2 * x - 1}
  S = {x : ℝ | x < -1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2822_282218


namespace NUMINAMATH_CALUDE_infinite_geometric_series_sum_l2822_282258

theorem infinite_geometric_series_sum : 
  let a : ℚ := 2/5
  let r : ℚ := 1/2
  let series_sum := a / (1 - r)
  series_sum = 4/5 := by sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_sum_l2822_282258


namespace NUMINAMATH_CALUDE_isosceles_probability_2020gon_l2822_282266

/-- The number of vertices in the regular polygon -/
def n : ℕ := 2020

/-- The probability of forming an isosceles triangle by randomly selecting
    three distinct vertices from a regular n-gon -/
def isosceles_probability (n : ℕ) : ℚ :=
  (n * ((n - 2) / 2)) / Nat.choose n 3

/-- Theorem stating that the probability of forming an isosceles triangle
    by randomly selecting three distinct vertices from a regular 2020-gon
    is 1/673 -/
theorem isosceles_probability_2020gon :
  isosceles_probability n = 1 / 673 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_probability_2020gon_l2822_282266


namespace NUMINAMATH_CALUDE_weekly_pay_solution_l2822_282214

def weekly_pay_problem (y_pay : ℝ) (x_percentage : ℝ) : Prop :=
  let x_pay := x_percentage * y_pay
  x_pay + y_pay = 638

theorem weekly_pay_solution :
  weekly_pay_problem 290 1.2 :=
by sorry

end NUMINAMATH_CALUDE_weekly_pay_solution_l2822_282214


namespace NUMINAMATH_CALUDE_ice_cream_survey_l2822_282237

theorem ice_cream_survey (total : ℕ) (strawberry_percent : ℚ) (vanilla_percent : ℚ) (chocolate_percent : ℚ)
  (h_total : total = 500)
  (h_strawberry : strawberry_percent = 46 / 100)
  (h_vanilla : vanilla_percent = 71 / 100)
  (h_chocolate : chocolate_percent = 85 / 100) :
  ∃ (all_three : ℕ), all_three ≥ 10 ∧
    (strawberry_percent * total + vanilla_percent * total + chocolate_percent * total
      = (total - all_three) * 2 + all_three * 3) :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_survey_l2822_282237


namespace NUMINAMATH_CALUDE_sandy_nickels_borrowed_l2822_282272

/-- Given the initial number of nickels and the remaining number of nickels,
    calculate the number of nickels borrowed. -/
def nickels_borrowed (initial : Nat) (remaining : Nat) : Nat :=
  initial - remaining

theorem sandy_nickels_borrowed :
  let initial_nickels : Nat := 31
  let remaining_nickels : Nat := 11
  nickels_borrowed initial_nickels remaining_nickels = 20 := by
  sorry

end NUMINAMATH_CALUDE_sandy_nickels_borrowed_l2822_282272


namespace NUMINAMATH_CALUDE_ali_fish_weight_l2822_282204

theorem ali_fish_weight (peter_weight joey_weight ali_weight : ℝ) 
  (h1 : ali_weight = 2 * peter_weight)
  (h2 : joey_weight = peter_weight + 1)
  (h3 : peter_weight + joey_weight + ali_weight = 25) :
  ali_weight = 12 := by
sorry

end NUMINAMATH_CALUDE_ali_fish_weight_l2822_282204


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2822_282231

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) + x = x * f y + f x

theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, FunctionalEquation f → f (1/2) = 0 → f (-201) = 403 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2822_282231


namespace NUMINAMATH_CALUDE_factor_expression_l2822_282213

theorem factor_expression (x : ℝ) : 
  (20 * x^4 + 100 * x^2 - 10) - (-5 * x^4 + 15 * x^2 - 10) = 5 * x^2 * (5 * x^2 + 17) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2822_282213


namespace NUMINAMATH_CALUDE_product_mod_nineteen_l2822_282226

theorem product_mod_nineteen : (2001 * 2002 * 2003 * 2004 * 2005) % 19 = 11 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_nineteen_l2822_282226


namespace NUMINAMATH_CALUDE_zeta_power_sum_l2822_282287

theorem zeta_power_sum (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 1)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 5)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 9) :
  ζ₁^8 + ζ₂^8 + ζ₃^8 = 179 := by
  sorry

end NUMINAMATH_CALUDE_zeta_power_sum_l2822_282287


namespace NUMINAMATH_CALUDE_binomial_coefficient_congruence_l2822_282243

theorem binomial_coefficient_congruence (n p : ℕ) (h_prime : Nat.Prime p) (h_n_gt_p : n > p) :
  (n.choose p) ≡ (n / p : ℕ) [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_congruence_l2822_282243


namespace NUMINAMATH_CALUDE_least_subtraction_for_common_remainder_l2822_282293

theorem least_subtraction_for_common_remainder (n : ℕ) : 
  (∃ (x : ℕ), x ≤ n ∧ 
   (642 - x) % 11 = 4 ∧ 
   (642 - x) % 13 = 4 ∧ 
   (642 - x) % 17 = 4) → 
  (∃ (x : ℕ), x ≤ n ∧ 
   (642 - x) % 11 = 4 ∧ 
   (642 - x) % 13 = 4 ∧ 
   (642 - x) % 17 = 4 ∧
   ∀ (y : ℕ), y < x → 
     ((642 - y) % 11 ≠ 4 ∨ 
      (642 - y) % 13 ≠ 4 ∨ 
      (642 - y) % 17 ≠ 4)) :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_common_remainder_l2822_282293


namespace NUMINAMATH_CALUDE_subset_condition_intersection_condition_l2822_282247

def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}

def B (a : ℝ) : Set ℝ := {x | (x - a)*(x - 3*a) < 0}

theorem subset_condition (a : ℝ) : A ⊆ B a ↔ 4/3 ≤ a ∧ a ≤ 2 := by sorry

theorem intersection_condition (a : ℝ) : A ∩ B a = {x | 3 < x ∧ x < 4} ↔ a = 3 := by sorry

end NUMINAMATH_CALUDE_subset_condition_intersection_condition_l2822_282247


namespace NUMINAMATH_CALUDE_fraction_reduction_l2822_282281

theorem fraction_reduction (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^3 - b^3) / (a*b) - (a*b^2 - b^3) / (a*b - a^3) = (a^2 + a*b + b^2) / b :=
by sorry

end NUMINAMATH_CALUDE_fraction_reduction_l2822_282281


namespace NUMINAMATH_CALUDE_trapezium_area_from_equilateral_triangles_l2822_282269

theorem trapezium_area_from_equilateral_triangles 
  (triangle_area : ℝ) 
  (h : ℝ) -- height of small triangle
  (b : ℝ) -- base of small triangle
  (h_pos : h > 0)
  (b_pos : b > 0)
  (area_eq : (1/2) * b * h = triangle_area)
  (triangle_area_val : triangle_area = 4) :
  let trapezium_area := (1/2) * (4*h + 5*h) * (5/2*b)
  trapezium_area = 90 := by
sorry

end NUMINAMATH_CALUDE_trapezium_area_from_equilateral_triangles_l2822_282269


namespace NUMINAMATH_CALUDE_profit_percentage_is_fifty_percent_l2822_282249

def purchase_price : ℕ := 14000
def repair_cost : ℕ := 5000
def transportation_charges : ℕ := 1000
def selling_price : ℕ := 30000

def total_cost : ℕ := purchase_price + repair_cost + transportation_charges
def profit : ℕ := selling_price - total_cost

theorem profit_percentage_is_fifty_percent :
  (profit : ℚ) / (total_cost : ℚ) * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_profit_percentage_is_fifty_percent_l2822_282249


namespace NUMINAMATH_CALUDE_smallest_power_of_three_l2822_282271

theorem smallest_power_of_three : ∃ n : ℕ, 3^n = 729 ∧ 3^n < 1000 ∧ ∀ m : ℕ, m < n → 3^m < 729 := by
  sorry

end NUMINAMATH_CALUDE_smallest_power_of_three_l2822_282271


namespace NUMINAMATH_CALUDE_proposition_d_true_others_false_l2822_282267

theorem proposition_d_true_others_false :
  (∃ x : ℝ, 3 * x^2 - 4 = 6 * x) ∧
  ¬(∀ x : ℝ, (x - Real.sqrt 2)^2 > 0) ∧
  ¬(∀ x : ℚ, x^2 > 0) ∧
  ¬(∃ x : ℤ, 3 * x = 128) :=
by sorry

end NUMINAMATH_CALUDE_proposition_d_true_others_false_l2822_282267


namespace NUMINAMATH_CALUDE_tom_not_in_middle_seat_l2822_282286

-- Define the people
inductive Person : Type
| Andy : Person
| Jen : Person
| Sally : Person
| Mike : Person
| Tom : Person

-- Define a seating arrangement as a function from seat number to person
def Seating := Fin 5 → Person

-- Andy is not beside Jen
def AndyNotBesideJen (s : Seating) : Prop :=
  ∀ i : Fin 4, s i ≠ Person.Andy ∨ s i.succ ≠ Person.Jen

-- Sally is beside Mike
def SallyBesideMike (s : Seating) : Prop :=
  ∃ i : Fin 4, (s i = Person.Sally ∧ s i.succ = Person.Mike) ∨
               (s i = Person.Mike ∧ s i.succ = Person.Sally)

-- The middle seat is the third seat
def MiddleSeat : Fin 5 := ⟨2, by norm_num⟩

-- Theorem: Tom cannot sit in the middle seat
theorem tom_not_in_middle_seat :
  ∀ s : Seating, AndyNotBesideJen s → SallyBesideMike s →
  s MiddleSeat ≠ Person.Tom :=
by sorry

end NUMINAMATH_CALUDE_tom_not_in_middle_seat_l2822_282286


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2822_282256

theorem inequality_system_solution_set :
  ∀ x : ℝ, (x - 1 < 0 ∧ x + 1 > 0) ↔ (-1 < x ∧ x < 1) := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2822_282256


namespace NUMINAMATH_CALUDE_min_y_l2822_282265

variable (a b c d : ℝ)
variable (x : ℝ)

def y (x : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + c*(x - d)^2

theorem min_y :
  ∃ (x_min : ℝ), (∀ (x : ℝ), y x_min ≤ y x) ∧ x_min = (a + b + c*d) / (2 + c) :=
sorry

end NUMINAMATH_CALUDE_min_y_l2822_282265


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l2822_282219

theorem danny_bottle_caps (found_new : ℕ) (total_after : ℕ) (difference : ℕ) 
  (h1 : found_new = 50)
  (h2 : total_after = 60)
  (h3 : found_new = difference + 44) : 
  found_new - difference = 6 := by
sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l2822_282219


namespace NUMINAMATH_CALUDE_lcm_9_12_15_l2822_282235

theorem lcm_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_9_12_15_l2822_282235


namespace NUMINAMATH_CALUDE_sqrt_three_squared_l2822_282277

theorem sqrt_three_squared : (Real.sqrt 3)^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_squared_l2822_282277


namespace NUMINAMATH_CALUDE_profit_maximum_l2822_282224

/-- Profit function for a product -/
def profit (m : ℝ) : ℝ := (m - 8) * (900 - 15 * m)

/-- Maximum profit expression -/
def max_profit_expr (m : ℝ) : ℝ := -15 * (m - 34)^2 + 10140

theorem profit_maximum :
  ∃ (m : ℝ), 
    (∀ (x : ℝ), profit x ≤ profit m) ∧
    (profit m = max_profit_expr m) ∧
    (m = 34) :=
sorry

end NUMINAMATH_CALUDE_profit_maximum_l2822_282224


namespace NUMINAMATH_CALUDE_ratio_AD_DC_is_3_to_2_l2822_282212

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = 6 ∧ BC = 8 ∧ AC = 10

-- Define point D on AC
def point_D_on_AC (A C D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * A.1 + (1 - t) * C.1, t * A.2 + (1 - t) * C.2)

-- Define BD = 7
def BD_equals_7 (B D : ℝ × ℝ) : Prop :=
  Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 7

-- Theorem statement
theorem ratio_AD_DC_is_3_to_2 
  (A B C D : ℝ × ℝ) 
  (h_triangle : triangle_ABC A B C) 
  (h_D_on_AC : point_D_on_AC A C D) 
  (h_BD : BD_equals_7 B D) : 
  ∃ (AD DC : ℝ), AD / DC = 3 / 2 := 
sorry

end NUMINAMATH_CALUDE_ratio_AD_DC_is_3_to_2_l2822_282212


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2822_282234

theorem unique_solution_for_equation : ∃! (n : ℕ), n > 0 ∧ n^2 + n + 6*n = 210 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2822_282234


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2822_282216

theorem partial_fraction_decomposition :
  ∃ (A B : ℚ), 
    (∀ x : ℚ, x ≠ 7 ∧ x ≠ -9 → 
      (2 * x + 4) / (x^2 + 2*x - 63) = A / (x - 7) + B / (x + 9)) ∧
    A = 9/8 ∧ B = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2822_282216


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l2822_282254

theorem zeros_before_first_nonzero_digit (n : ℕ) (d : ℕ) (h : d = 2^7 * 5^9) :
  (∃ k : ℕ, (3 : ℚ) / d = (k : ℚ) / 10^9 ∧ 1 ≤ k ∧ k < 10) →
  (∃ m : ℕ, (3 : ℚ) / d = (m : ℚ) / 10^8 ∧ 10 ≤ m) →
  n = 8 :=
sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l2822_282254


namespace NUMINAMATH_CALUDE_parabola_c_value_l2822_282297

/-- A parabola passing through three given points has a specific c value -/
theorem parabola_c_value (b c : ℝ) :
  (1^2 + b*1 + c = 2) ∧ 
  (4^2 + b*4 + c = 5) ∧ 
  (7^2 + b*7 + c = 2) →
  c = 9 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2822_282297


namespace NUMINAMATH_CALUDE_cubic_equation_unique_solution_l2822_282208

theorem cubic_equation_unique_solution :
  ∃! (x y : ℕ+), (y : ℤ)^3 = (x : ℤ)^3 + 8*(x : ℤ)^2 - 6*(x : ℤ) + 8 ∧ x = 9 ∧ y = 11 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_unique_solution_l2822_282208


namespace NUMINAMATH_CALUDE_cream_needed_proof_l2822_282233

/-- The amount of additional cream needed when given a total required amount and an available amount -/
def additional_cream_needed (total_required : ℕ) (available : ℕ) : ℕ :=
  total_required - available

/-- Theorem stating that given 300 lbs total required and 149 lbs available, 151 lbs additional cream is needed -/
theorem cream_needed_proof :
  additional_cream_needed 300 149 = 151 := by
  sorry

end NUMINAMATH_CALUDE_cream_needed_proof_l2822_282233


namespace NUMINAMATH_CALUDE_head_start_problem_l2822_282298

/-- The head start problem -/
theorem head_start_problem (cristina_speed nicky_speed : ℝ) (catch_up_time : ℝ) 
  (h1 : cristina_speed = 5)
  (h2 : nicky_speed = 3)
  (h3 : catch_up_time = 24) :
  cristina_speed * catch_up_time - nicky_speed * catch_up_time = 48 := by
  sorry

#check head_start_problem

end NUMINAMATH_CALUDE_head_start_problem_l2822_282298


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_l2822_282203

theorem fraction_inequality_solution (x : ℝ) : 
  (x ≠ 5) → (x / (x - 5) ≥ 0 ↔ x ∈ Set.Ici 5 ∪ Set.Iic 0) :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_l2822_282203


namespace NUMINAMATH_CALUDE_student_arrangement_l2822_282260

/-- The number of arrangements for n male and m female students -/
def arrangement_count (n m : ℕ) : ℕ := sorry

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of permutations of k items from n items -/
def permute (n k : ℕ) : ℕ := sorry

theorem student_arrangement :
  let total_male : ℕ := 5
  let total_female : ℕ := 5
  let females_between : ℕ := 2
  let males_at_ends : ℕ := 2
  
  arrangement_count total_male total_female = 
    choose total_female females_between * 
    permute (total_male - 2) males_at_ends * 
    permute (total_male + total_female - females_between - males_at_ends - 2) 
            (total_male + total_female - females_between - males_at_ends - 2) :=
by sorry

end NUMINAMATH_CALUDE_student_arrangement_l2822_282260


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l2822_282202

theorem least_addition_for_divisibility :
  ∃! x : ℕ, x < 103 ∧ (3457 + x) % 103 = 0 ∧ ∀ y : ℕ, y < x → (3457 + y) % 103 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l2822_282202


namespace NUMINAMATH_CALUDE_f_inequality_range_l2822_282209

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + x else Real.log x / Real.log 0.3

theorem f_inequality_range (t : ℝ) :
  (∀ x, f x ≤ t^2/4 - t + 1) ↔ t ∈ Set.Iic 1 ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_f_inequality_range_l2822_282209


namespace NUMINAMATH_CALUDE_bill_denomination_l2822_282275

theorem bill_denomination (total_amount : ℕ) (num_bills : ℕ) (h1 : total_amount = 45) (h2 : num_bills = 9) :
  total_amount / num_bills = 5 := by
sorry

end NUMINAMATH_CALUDE_bill_denomination_l2822_282275


namespace NUMINAMATH_CALUDE_function_is_even_l2822_282294

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_is_even
  (f : ℝ → ℝ)
  (h1 : has_period f 4)
  (h2 : ∀ x, f (2 + x) = f (2 - x)) :
  is_even_function f :=
sorry

end NUMINAMATH_CALUDE_function_is_even_l2822_282294


namespace NUMINAMATH_CALUDE_most_likely_outcome_is_equal_distribution_l2822_282296

def probability_of_outcome (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

theorem most_likely_outcome_is_equal_distribution :
  ∀ k : ℕ, k ≤ 8 →
    probability_of_outcome 8 4 ≥ probability_of_outcome 8 k :=
sorry

end NUMINAMATH_CALUDE_most_likely_outcome_is_equal_distribution_l2822_282296


namespace NUMINAMATH_CALUDE_largest_solution_and_ratio_l2822_282268

theorem largest_solution_and_ratio : ∃ (a b c d : ℤ),
  let x : ℝ := (a + b * Real.sqrt c) / d
  ∀ y : ℝ, (6 * y / 5 - 2 = 4 / y) → y ≤ x ∧
  x = (5 + Real.sqrt 145) / 6 ∧
  a * c * d / b = 4350 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_and_ratio_l2822_282268


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_l2822_282289

theorem min_value_sqrt_sum (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0)
  (h4 : a * b + b * c + c * a = a + b + c) (h5 : a + b + c > 0) :
  Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ≥ 2 := by
  sorry

#check min_value_sqrt_sum

end NUMINAMATH_CALUDE_min_value_sqrt_sum_l2822_282289


namespace NUMINAMATH_CALUDE_quadratic_two_zeros_l2822_282240

theorem quadratic_two_zeros (a b c : ℝ) (h : a * c < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_zeros_l2822_282240


namespace NUMINAMATH_CALUDE_extreme_values_imply_a_range_l2822_282236

/-- A function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - 2*x + a * Real.log x

/-- The derivative of f(x) with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := x - 2 + a / x

/-- Theorem stating that if f(x) has two distinct extreme values, then 0 < a < 1 -/
theorem extreme_values_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
    f_derivative a x₁ = 0 ∧ f_derivative a x₂ = 0) →
  0 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_extreme_values_imply_a_range_l2822_282236


namespace NUMINAMATH_CALUDE_product_of_first_five_l2822_282215

def is_on_line (x y : ℝ) : Prop :=
  3 * x + y = 0

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → is_on_line (a (n+1)) (a n)

theorem product_of_first_five (a : ℕ → ℝ) :
  sequence_property a → a 2 = 6 → a 1 * a 2 * a 3 * a 4 * a 5 = -32 := by
  sorry

end NUMINAMATH_CALUDE_product_of_first_five_l2822_282215


namespace NUMINAMATH_CALUDE_largest_mu_inequality_l2822_282220

theorem largest_mu_inequality : 
  ∃ (μ : ℝ), (∀ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≥ 2*a*b + μ*b*c + 3*c*d) ∧ 
  (∀ (μ' : ℝ), (∀ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    a^2 + b^2 + c^2 + d^2 ≥ 2*a*b + μ'*b*c + 3*c*d) → μ' ≤ μ) ∧ 
  μ = 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_mu_inequality_l2822_282220


namespace NUMINAMATH_CALUDE_abc_value_l2822_282222

theorem abc_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 45 * Real.rpow 3 (1/3))
  (hac : a * c = 63 * Real.rpow 3 (1/3))
  (hbc : b * c = 28 * Real.rpow 3 (1/3)) :
  a * b * c = 630 := by
sorry

end NUMINAMATH_CALUDE_abc_value_l2822_282222


namespace NUMINAMATH_CALUDE_bella_soccer_goals_l2822_282262

def goals_first_6 : List Nat := [5, 3, 2, 4, 1, 6]

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

theorem bella_soccer_goals :
  ∀ (g7 g8 : Nat),
    g7 < 10 →
    g8 < 10 →
    is_integer ((goals_first_6.sum + g7) / 7) →
    is_integer ((goals_first_6.sum + g7 + g8) / 8) →
    g7 * g8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_bella_soccer_goals_l2822_282262


namespace NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l2822_282288

theorem neg_p_sufficient_not_necessary_for_neg_q :
  let p := {x : ℝ | x < -1}
  let q := {x : ℝ | x < -4}
  (∀ x, x ∉ p → x ∉ q) ∧ (∃ x, x ∉ q ∧ x ∈ p) := by
  sorry

end NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l2822_282288


namespace NUMINAMATH_CALUDE_floor_of_3_2_l2822_282201

theorem floor_of_3_2 : ⌊(3.2 : ℝ)⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_floor_of_3_2_l2822_282201


namespace NUMINAMATH_CALUDE_find_m_l2822_282259

theorem find_m (A B : Set ℕ) (m : ℕ) : 
  A = {1, 2, 3} →
  B = {2, m, 4} →
  A ∩ B = {2, 3} →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_find_m_l2822_282259


namespace NUMINAMATH_CALUDE_division_equality_l2822_282283

theorem division_equality : (203515 : ℕ) / 2015 = 101 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l2822_282283


namespace NUMINAMATH_CALUDE_quadratic_less_than_linear_l2822_282292

theorem quadratic_less_than_linear (x : ℝ) : -1/2 * x^2 + 2*x < -x + 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_less_than_linear_l2822_282292


namespace NUMINAMATH_CALUDE_parity_of_D_2024_2025_2026_l2822_282225

def D : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | 3 => 2
  | n + 4 => D (n + 3) + D (n + 1)

theorem parity_of_D_2024_2025_2026 :
  Odd (D 2024) ∧ Even (D 2025) ∧ Even (D 2026) := by
  sorry

end NUMINAMATH_CALUDE_parity_of_D_2024_2025_2026_l2822_282225


namespace NUMINAMATH_CALUDE_linear_system_ratio_l2822_282244

/-- Given a system of linear equations with a nontrivial solution, prove that xz/y^2 = 26/9 -/
theorem linear_system_ratio (x y z k : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x + k * y + 4 * z = 0 →
  4 * x + k * y - 3 * z = 0 →
  x + 3 * y - 2 * z = 0 →
  x * z / (y ^ 2) = 26 / 9 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_ratio_l2822_282244


namespace NUMINAMATH_CALUDE_polygon_sides_l2822_282255

theorem polygon_sides (interior_angle_sum : ℝ) : interior_angle_sum = 540 → ∃ n : ℕ, n = 5 ∧ (n - 2) * 180 = interior_angle_sum := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2822_282255


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l2822_282227

theorem simplify_complex_fraction (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 5) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  ((x - 1) * (x - 5)) / ((x - 3) * (x - 4) * (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l2822_282227


namespace NUMINAMATH_CALUDE_ground_mince_calculation_l2822_282232

/-- The total amount of ground mince used for lasagnas and cottage pies -/
def total_ground_mince (num_lasagnas : ℕ) (mince_per_lasagna : ℕ) 
                       (num_cottage_pies : ℕ) (mince_per_cottage_pie : ℕ) : ℕ :=
  num_lasagnas * mince_per_lasagna + num_cottage_pies * mince_per_cottage_pie

/-- Theorem stating the total amount of ground mince used -/
theorem ground_mince_calculation :
  total_ground_mince 100 2 100 3 = 500 := by
  sorry

end NUMINAMATH_CALUDE_ground_mince_calculation_l2822_282232


namespace NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_l2822_282223

theorem unique_x_with_three_prime_divisors (x n : ℕ) : 
  x = 9^n - 1 →
  (∃ p q r : ℕ, Prime p ∧ Prime q ∧ Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r) →
  13 ∣ x →
  x = 728 :=
sorry

end NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_l2822_282223


namespace NUMINAMATH_CALUDE_angle_inequality_l2822_282282

open Real

theorem angle_inequality (a b c : ℝ) 
  (ha : a = sin (33 * π / 180))
  (hb : b = cos (55 * π / 180))
  (hc : c = tan (55 * π / 180)) :
  c > b ∧ b > a :=
by sorry

end NUMINAMATH_CALUDE_angle_inequality_l2822_282282


namespace NUMINAMATH_CALUDE_merchant_profit_l2822_282239

theorem merchant_profit (cost : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : 
  markup_percent = 20 → 
  discount_percent = 10 → 
  let marked_price := cost * (1 + markup_percent / 100)
  let final_price := marked_price * (1 - discount_percent / 100)
  let profit_percent := (final_price - cost) / cost * 100
  profit_percent = 8 := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_l2822_282239


namespace NUMINAMATH_CALUDE_inequality_proof_l2822_282238

theorem inequality_proof (a : ℝ) (ha : a > 0) : 2 * a / (1 + a^2) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2822_282238


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2822_282284

/-- A line passing through (2,0) and tangent to y = 1/x has equation x + y - 2 = 0 -/
theorem tangent_line_equation : ∃ (k : ℝ),
  (∀ x y : ℝ, y = k * (x - 2) → y = 1 / x → x * x * k - 2 * x * k - 1 = 0) ∧
  (4 * k * k + 4 * k = 0) ∧
  (∀ x y : ℝ, y = k * (x - 2) ↔ x + y - 2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_tangent_line_equation_l2822_282284


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l2822_282278

theorem min_value_squared_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 1) :
  x^2 + y^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l2822_282278


namespace NUMINAMATH_CALUDE_multiple_properties_l2822_282291

theorem multiple_properties (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 4 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  (∃ n : ℤ, b = 4 * n) ∧ (∃ p : ℤ, a - b = 4 * p) := by
  sorry

end NUMINAMATH_CALUDE_multiple_properties_l2822_282291


namespace NUMINAMATH_CALUDE_get_ready_time_l2822_282261

/-- The time it takes Jack to put on his own shoes, in minutes. -/
def jack_shoes_time : ℕ := 4

/-- The additional time it takes Jack to help a toddler with their shoes, in minutes. -/
def additional_toddler_time : ℕ := 3

/-- The number of toddlers Jack needs to help. -/
def number_of_toddlers : ℕ := 2

/-- The total time it takes for Jack and his toddlers to get ready, in minutes. -/
def total_time : ℕ := jack_shoes_time + number_of_toddlers * (jack_shoes_time + additional_toddler_time)

theorem get_ready_time : total_time = 18 := by
  sorry

end NUMINAMATH_CALUDE_get_ready_time_l2822_282261


namespace NUMINAMATH_CALUDE_probability_closer_to_center_l2822_282274

theorem probability_closer_to_center (r : ℝ) (h : r = 3) : 
  (π * (r / 2)^2) / (π * r^2) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_closer_to_center_l2822_282274


namespace NUMINAMATH_CALUDE_shelby_gold_stars_l2822_282290

def gold_stars_problem (yesterday : ℕ) (total : ℕ) : Prop :=
  ∃ today : ℕ, yesterday + today = total

theorem shelby_gold_stars :
  gold_stars_problem 4 7 → ∃ today : ℕ, today = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_shelby_gold_stars_l2822_282290


namespace NUMINAMATH_CALUDE_system_solution_cubic_equation_solution_l2822_282252

-- Problem 1: System of equations
theorem system_solution :
  ∃! (x y : ℝ), 3 * x + 2 * y = 19 ∧ 2 * x - y = 1 ∧ x = 3 ∧ y = 5 := by
  sorry

-- Problem 2: Cubic equation
theorem cubic_equation_solution :
  ∃! x : ℝ, (2 * x - 1)^3 = -8 ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_cubic_equation_solution_l2822_282252


namespace NUMINAMATH_CALUDE_max_profit_theorem_l2822_282229

/-- Represents the profit function for a souvenir shop -/
def profit_function (x : ℝ) : ℝ := -20 * x + 3200

/-- Represents the constraint on the number of type A souvenirs -/
def constraint (x : ℝ) : Prop := x ≥ 10

/-- Theorem stating the maximum profit and the number of type A souvenirs that achieves it -/
theorem max_profit_theorem :
  ∃ (x : ℝ), constraint x ∧
  (∀ (y : ℝ), constraint y → profit_function x ≥ profit_function y) ∧
  x = 10 ∧ profit_function x = 3000 :=
sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l2822_282229


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficient_l2822_282207

theorem quadratic_equation_coefficient : ∀ a b c : ℝ,
  (∀ x, 3 * x^2 + 1 = 6 * x ↔ a * x^2 + b * x + c = 0) →
  a = 3 →
  b = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficient_l2822_282207


namespace NUMINAMATH_CALUDE_spurs_team_size_l2822_282273

/-- The number of players on the Spurs basketball team -/
def num_players : ℕ := 242 / 11

/-- The number of basketballs each player has -/
def basketballs_per_player : ℕ := 11

/-- The total number of basketballs -/
def total_basketballs : ℕ := 242

/-- Theorem stating that the number of players is 22 -/
theorem spurs_team_size : num_players = 22 := by
  sorry

end NUMINAMATH_CALUDE_spurs_team_size_l2822_282273


namespace NUMINAMATH_CALUDE_cube_has_twelve_edges_l2822_282217

/-- A cube is a three-dimensional shape with six square faces. -/
structure Cube where
  -- We don't need to specify any fields for this definition

/-- The number of edges in a cube. -/
def num_edges (c : Cube) : ℕ := 12

/-- Theorem: A cube has 12 edges. -/
theorem cube_has_twelve_edges (c : Cube) : num_edges c = 12 := by
  sorry

end NUMINAMATH_CALUDE_cube_has_twelve_edges_l2822_282217


namespace NUMINAMATH_CALUDE_projectile_max_height_l2822_282257

/-- Represents the elevation of a projectile at time t -/
def elevation (t : ℝ) : ℝ := 200 * t - 10 * t^2

/-- The time at which the projectile reaches its maximum height -/
def max_height_time : ℝ := 10

theorem projectile_max_height :
  ∀ t : ℝ, elevation t ≤ elevation max_height_time ∧
  elevation max_height_time = 1000 := by
  sorry

#check projectile_max_height

end NUMINAMATH_CALUDE_projectile_max_height_l2822_282257


namespace NUMINAMATH_CALUDE_coffee_consumption_theorem_l2822_282245

/-- Represents the relationship between sleep and coffee consumption -/
def coffee_sleep_relation (sleep : ℝ) (coffee : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ sleep * coffee = k

theorem coffee_consumption_theorem (sleep_monday sleep_tuesday coffee_monday : ℝ) 
  (h1 : sleep_monday > 0)
  (h2 : sleep_tuesday > 0)
  (h3 : coffee_monday > 0)
  (h4 : coffee_sleep_relation sleep_monday coffee_monday)
  (h5 : coffee_sleep_relation sleep_tuesday (sleep_monday * coffee_monday / sleep_tuesday))
  (h6 : sleep_monday = 9)
  (h7 : sleep_tuesday = 6)
  (h8 : coffee_monday = 2) :
  sleep_monday * coffee_monday / sleep_tuesday = 3 := by
  sorry

#check coffee_consumption_theorem

end NUMINAMATH_CALUDE_coffee_consumption_theorem_l2822_282245


namespace NUMINAMATH_CALUDE_sector_central_angle_l2822_282242

/-- The central angle of a sector with radius 1 cm and arc length 2 cm is 2 radians. -/
theorem sector_central_angle (radius : ℝ) (arc_length : ℝ) (central_angle : ℝ) : 
  radius = 1 → arc_length = 2 → arc_length = radius * central_angle → central_angle = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2822_282242


namespace NUMINAMATH_CALUDE_jerome_bicycle_trip_distance_l2822_282200

/-- The total distance of Jerome's bicycle trip -/
def total_distance (daily_distance : ℕ) (num_days : ℕ) (last_day_distance : ℕ) : ℕ :=
  daily_distance * num_days + last_day_distance

/-- Theorem stating that Jerome's bicycle trip is 150 miles long -/
theorem jerome_bicycle_trip_distance :
  total_distance 12 12 6 = 150 := by
  sorry

end NUMINAMATH_CALUDE_jerome_bicycle_trip_distance_l2822_282200


namespace NUMINAMATH_CALUDE_union_determines_x_l2822_282253

def A : Set ℕ := {1, 3}
def B (x : ℕ) : Set ℕ := {2, x}

theorem union_determines_x (x : ℕ) : A ∪ B x = {1, 2, 3, 4} → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_union_determines_x_l2822_282253


namespace NUMINAMATH_CALUDE_no_distinct_roots_l2822_282279

theorem no_distinct_roots : ¬∃ (a b c : ℝ), 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  (a^2 - 2*b*a + c^2 = 0) ∧
  (b^2 - 2*c*b + a^2 = 0) ∧
  (c^2 - 2*a*c + b^2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_distinct_roots_l2822_282279
