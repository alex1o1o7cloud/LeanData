import Mathlib

namespace NUMINAMATH_CALUDE_table_price_is_56_l3240_324038

/-- The price of a chair in dollars -/
def chair_price : ℝ := sorry

/-- The price of a table in dollars -/
def table_price : ℝ := sorry

/-- The condition that the price of 2 chairs and 1 table is 60% of the price of 1 chair and 2 tables -/
axiom price_ratio : 2 * chair_price + table_price = 0.6 * (chair_price + 2 * table_price)

/-- The condition that the price of 1 table and 1 chair is $64 -/
axiom total_price : chair_price + table_price = 64

/-- Theorem stating that the price of 1 table is $56 -/
theorem table_price_is_56 : table_price = 56 := by sorry

end NUMINAMATH_CALUDE_table_price_is_56_l3240_324038


namespace NUMINAMATH_CALUDE_trailing_zeros_of_power_sum_l3240_324042

theorem trailing_zeros_of_power_sum : ∃ n : ℕ, n > 0 ∧ 
  (4^(5^6) + 6^(5^4) : ℕ) % (10^n) = 0 ∧ 
  (4^(5^6) + 6^(5^4) : ℕ) % (10^(n+1)) ≠ 0 ∧ 
  n = 5 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_power_sum_l3240_324042


namespace NUMINAMATH_CALUDE_tile_border_ratio_l3240_324076

theorem tile_border_ratio :
  ∀ (n s d : ℝ),
  n > 0 →
  s > 0 →
  d > 0 →
  n = 24 →
  (24 * s)^2 / (24 * s + 25 * d)^2 = 64 / 100 →
  d / s = 6 / 25 := by
sorry

end NUMINAMATH_CALUDE_tile_border_ratio_l3240_324076


namespace NUMINAMATH_CALUDE_ordered_pairs_satisfying_inequalities_l3240_324099

theorem ordered_pairs_satisfying_inequalities :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (a b : ℤ), (a, b) ∈ s ↔ 
      (a^2 + b^2 < 16 ∧ 
       a^2 + b^2 < 8*a ∧ 
       a^2 + b^2 < 8*b)) ∧
    s.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_ordered_pairs_satisfying_inequalities_l3240_324099


namespace NUMINAMATH_CALUDE_population_change_factors_l3240_324002

-- Define the factors that can affect population change
inductive PopulationFactor
  | NaturalGrowth
  | Migration
  | Mortality
  | BirthRate

-- Define a function that determines if a factor affects population change
def affectsPopulationChange (factor : PopulationFactor) : Prop :=
  match factor with
  | PopulationFactor.NaturalGrowth => true
  | PopulationFactor.Migration => true
  | _ => false

-- Theorem stating that population change is determined by natural growth and migration
theorem population_change_factors :
  ∀ (factor : PopulationFactor),
    affectsPopulationChange factor ↔
      (factor = PopulationFactor.NaturalGrowth ∨ factor = PopulationFactor.Migration) :=
by
  sorry


end NUMINAMATH_CALUDE_population_change_factors_l3240_324002


namespace NUMINAMATH_CALUDE_solution_set_for_a_eq_2_range_of_a_l3240_324016

-- Define the function f
def f (a x : ℝ) : ℝ := |a - 3*x| - |2 + x|

-- Theorem for part (1)
theorem solution_set_for_a_eq_2 :
  {x : ℝ | f 2 x ≤ 3} = {x : ℝ | -3/4 ≤ x ∧ x ≤ 7/2} := by sorry

-- Theorem for part (2)
theorem range_of_a :
  {a : ℝ | ∃ x, f a x ≥ 1 ∧ ∃ y, a + 2*|2 + y| = 0} = {a : ℝ | a ≥ -5/2} := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_eq_2_range_of_a_l3240_324016


namespace NUMINAMATH_CALUDE_alpha_range_l3240_324068

theorem alpha_range (α : Real) (h1 : 0 ≤ α) (h2 : α ≤ π) 
  (h3 : ∀ x : Real, 8 * x^2 - (8 * Real.sin α) * x + Real.cos (2 * α) ≥ 0) :
  α ∈ Set.Icc 0 (π / 6) ∪ Set.Icc (5 * π / 6) π := by
  sorry

end NUMINAMATH_CALUDE_alpha_range_l3240_324068


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l3240_324026

theorem quadratic_roots_condition (d : ℝ) : 
  (∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) → 
  d = 9.8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l3240_324026


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3240_324065

theorem inequality_solution_set :
  {x : ℝ | 1 + x > 6 - 4 * x} = {x : ℝ | x > 1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3240_324065


namespace NUMINAMATH_CALUDE_impossible_sequence_is_invalid_l3240_324058

/-- Represents a sequence of letters --/
def Sequence := List Nat

/-- Checks if a sequence is valid according to the letter printing process --/
def is_valid_sequence (s : Sequence) : Prop :=
  ∀ i j, i < j → (s.indexOf i < s.indexOf j → ∀ k, i < k ∧ k < j → s.indexOf k < s.indexOf j)

/-- The impossible sequence --/
def impossible_sequence : Sequence := [4, 5, 2, 3, 1]

/-- Theorem stating that the impossible sequence is indeed impossible --/
theorem impossible_sequence_is_invalid : 
  ¬ is_valid_sequence impossible_sequence := by sorry

end NUMINAMATH_CALUDE_impossible_sequence_is_invalid_l3240_324058


namespace NUMINAMATH_CALUDE_ford_vehicle_count_l3240_324050

/-- Represents the number of vehicles of each brand on Louie's store parking lot -/
structure VehicleCounts where
  D : ℕ  -- Dodge
  H : ℕ  -- Hyundai
  K : ℕ  -- Kia
  Ho : ℕ -- Honda
  F : ℕ  -- Ford

/-- Conditions for the vehicle counts -/
def satisfiesConditions (v : VehicleCounts) : Prop :=
  v.D + v.H + v.K + v.Ho + v.F = 1000 ∧
  (35 : ℕ) * (v.D + v.H + v.K + v.Ho + v.F) = 100 * v.D ∧
  (10 : ℕ) * (v.D + v.H + v.K + v.Ho + v.F) = 100 * v.H ∧
  v.K = 2 * v.Ho + 50 ∧
  v.F = v.D - 200

theorem ford_vehicle_count (v : VehicleCounts) 
  (h : satisfiesConditions v) : v.F = 150 := by
  sorry

end NUMINAMATH_CALUDE_ford_vehicle_count_l3240_324050


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3240_324084

theorem rationalize_denominator : (1 : ℝ) / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3240_324084


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l3240_324024

theorem arithmetic_evaluation : 6 / 3 - 2 - 8 + 2 * 8 = 8 := by sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l3240_324024


namespace NUMINAMATH_CALUDE_power_division_equality_l3240_324061

theorem power_division_equality : (3 : ℕ)^15 / (27 : ℕ)^3 = 729 := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l3240_324061


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l3240_324075

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (1 / (x + 3))
  else if x < -3 then Int.floor (1 / (x + 3))
  else 0  -- arbitrary value for x = -3, as g is not defined there

-- Theorem statement
theorem zero_not_in_range_of_g : ∀ x : ℝ, g x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l3240_324075


namespace NUMINAMATH_CALUDE_michelle_needs_one_more_rack_l3240_324009

/-- Represents the pasta making scenario for Michelle -/
structure PastaMaking where
  flour_per_pound : ℕ  -- cups of flour needed per pound of pasta
  pounds_per_rack : ℕ  -- pounds of pasta that can fit on one rack
  owned_racks : ℕ     -- number of racks Michelle currently owns
  flour_bags : ℕ      -- number of flour bags
  cups_per_bag : ℕ    -- cups of flour in each bag

/-- Calculates the number of additional racks Michelle needs -/
def additional_racks_needed (pm : PastaMaking) : ℕ :=
  let total_flour := pm.flour_bags * pm.cups_per_bag
  let total_pounds := total_flour / pm.flour_per_pound
  let total_racks_needed := (total_pounds + pm.pounds_per_rack - 1) / pm.pounds_per_rack
  (total_racks_needed - pm.owned_racks).max 0

/-- Theorem stating that Michelle needs one more rack -/
theorem michelle_needs_one_more_rack :
  let pm : PastaMaking := {
    flour_per_pound := 2,
    pounds_per_rack := 3,
    owned_racks := 3,
    flour_bags := 3,
    cups_per_bag := 8
  }
  additional_racks_needed pm = 1 := by sorry

end NUMINAMATH_CALUDE_michelle_needs_one_more_rack_l3240_324009


namespace NUMINAMATH_CALUDE_max_value_of_quadratic_l3240_324097

open Real

theorem max_value_of_quadratic (x : ℝ) (h : 0 < x ∧ x < 1) : 
  ∃ (max_val : ℝ), max_val = 1/4 ∧ ∀ y, 0 < y ∧ y < 1 → y * (1 - y) ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_quadratic_l3240_324097


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3240_324045

theorem purely_imaginary_complex_number (m : ℝ) : 
  (((m - 1) * Complex.I + (m^2 - 1) : ℂ).re = 0 ∧ ((m - 1) * Complex.I + (m^2 - 1) : ℂ).im ≠ 0) → 
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3240_324045


namespace NUMINAMATH_CALUDE_liar_count_l3240_324028

/-- Represents a candidate's statement about the number of lies told before their turn. -/
structure CandidateStatement where
  position : Nat
  claimed_lies : Nat
  is_truthful : Bool

/-- The debate scenario with 12 candidates. -/
def debate_scenario (statements : Vector CandidateStatement 12) : Prop :=
  (∀ i : Fin 12, (statements.get i).position = i.val + 1) ∧
  (∀ i : Fin 12, (statements.get i).claimed_lies = i.val + 1) ∧
  (∃ i : Fin 12, (statements.get i).is_truthful)

/-- The theorem to be proved. -/
theorem liar_count (statements : Vector CandidateStatement 12) 
  (h : debate_scenario statements) : 
  (statements.toList.filter (fun s => !s.is_truthful)).length = 11 := by
  sorry


end NUMINAMATH_CALUDE_liar_count_l3240_324028


namespace NUMINAMATH_CALUDE_intersecting_rectangles_area_l3240_324069

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℕ
  length : ℕ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.width * r.length

/-- Represents the overlap between two rectangles -/
structure Overlap where
  width : ℕ
  length : ℕ

/-- Calculates the area of overlap -/
def overlapArea (o : Overlap) : ℕ := o.width * o.length

theorem intersecting_rectangles_area (r1 r2 r3 : Rectangle) 
  (o12 o13 o23 o123 : Overlap) : 
  r1.width = 4 → r1.length = 12 →
  r2.width = 5 → r2.length = 10 →
  r3.width = 3 → r3.length = 6 →
  o12.width = 4 → o12.length = 5 →
  o13.width = 3 → o13.length = 4 →
  o23.width = 3 → o23.length = 3 →
  o123.width = 3 → o123.length = 3 →
  area r1 + area r2 + area r3 - (overlapArea o12 + overlapArea o13 + overlapArea o23) + overlapArea o123 = 84 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_rectangles_area_l3240_324069


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_square_plus_one_l3240_324022

theorem negation_of_universal_positive_square_plus_one (p : Prop) :
  (p ↔ ∀ x : ℝ, x^2 + 1 > 0) →
  (¬p ↔ ∃ x : ℝ, x^2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_square_plus_one_l3240_324022


namespace NUMINAMATH_CALUDE_sound_distance_at_10C_l3240_324031

-- Define the relationship between temperature and speed of sound
def speed_of_sound (temp : Int) : Int :=
  match temp with
  | -20 => 318
  | -10 => 324
  | 0 => 330
  | 10 => 336
  | 20 => 342
  | 30 => 348
  | _ => 0  -- For temperatures not in the table

-- Theorem statement
theorem sound_distance_at_10C (temp : Int) (time : Int) :
  temp = 10 ∧ time = 4 → speed_of_sound temp * time = 1344 := by
  sorry

end NUMINAMATH_CALUDE_sound_distance_at_10C_l3240_324031


namespace NUMINAMATH_CALUDE_f_three_point_five_l3240_324020

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def periodic_neg (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f x

def identity_on_interval (f : ℝ → ℝ) : Prop := ∀ x, 0 < x → x < 1 → f x = x

theorem f_three_point_five 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_periodic : periodic_neg f) 
  (h_identity : identity_on_interval f) : 
  f 3.5 = -0.5 := by
sorry

end NUMINAMATH_CALUDE_f_three_point_five_l3240_324020


namespace NUMINAMATH_CALUDE_parabola_decreases_left_of_vertex_given_parabola_decreases_left_of_vertex_l3240_324040

/-- Represents a parabola of the form y = (x - h)^2 + k -/
structure Parabola where
  h : ℝ
  k : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  (x - p.h)^2 + p.k

theorem parabola_decreases_left_of_vertex (p : Parabola) :
  ∀ x₁ x₂, x₁ < x₂ → x₂ < p.h → p.y_coord x₁ > p.y_coord x₂ := by
  sorry

/-- The specific parabola y = (x - 2)^2 + 1 -/
def given_parabola : Parabola :=
  { h := 2, k := 1 }

theorem given_parabola_decreases_left_of_vertex :
  ∀ x₁ x₂, x₁ < x₂ → x₂ < 2 → given_parabola.y_coord x₁ > given_parabola.y_coord x₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_decreases_left_of_vertex_given_parabola_decreases_left_of_vertex_l3240_324040


namespace NUMINAMATH_CALUDE_product_of_numbers_l3240_324080

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 37) (h2 : x - y = 5) : x * y = 336 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3240_324080


namespace NUMINAMATH_CALUDE_special_triangle_perimeter_l3240_324041

/-- A triangle with specific properties -/
structure SpecialTriangle where
  /-- One side of the triangle -/
  a : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Radius of the circumscribed circle -/
  R : ℝ
  /-- The side length is positive -/
  a_pos : 0 < a
  /-- The inscribed radius is positive -/
  r_pos : 0 < r
  /-- The circumscribed radius is positive -/
  R_pos : 0 < R

/-- Theorem: The perimeter of the special triangle is 24 -/
theorem special_triangle_perimeter (t : SpecialTriangle)
    (h1 : t.a = 6)
    (h2 : t.r = 2)
    (h3 : t.R = 5) :
    ∃ (b c : ℝ), b > 0 ∧ c > 0 ∧ t.a + b + c = 24 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_perimeter_l3240_324041


namespace NUMINAMATH_CALUDE_range_of_a_l3240_324078

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (A a ∪ B a = Set.univ) → a ∈ Set.Iic 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3240_324078


namespace NUMINAMATH_CALUDE_dog_park_problem_l3240_324056

theorem dog_park_problem (total_dogs : ℕ) (spotted_dogs : ℕ) (pointy_eared_dogs : ℕ) :
  spotted_dogs = 15 →
  2 * spotted_dogs = total_dogs →
  5 * pointy_eared_dogs = total_dogs →
  pointy_eared_dogs = 6 := by
  sorry

end NUMINAMATH_CALUDE_dog_park_problem_l3240_324056


namespace NUMINAMATH_CALUDE_solve_candy_problem_l3240_324091

def candy_problem (debby_candy : ℕ) (sister_candy : ℕ) (remaining_candy : ℕ) : Prop :=
  let total_candy := debby_candy + sister_candy
  let eaten_candy := total_candy - remaining_candy
  eaten_candy = 35

theorem solve_candy_problem :
  candy_problem 32 42 39 := by
  sorry

end NUMINAMATH_CALUDE_solve_candy_problem_l3240_324091


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l3240_324096

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 2*y - 13 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 12*x + 16*y - 25 = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := 4*x + 3*y - 2 = 0

-- Theorem statement
theorem common_chord_of_circles :
  ∀ x y : ℝ, (C₁ x y ∧ C₂ x y) → common_chord x y :=
by sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l3240_324096


namespace NUMINAMATH_CALUDE_jeremy_payment_l3240_324013

theorem jeremy_payment (rate : ℚ) (rooms : ℚ) (h1 : rate = 13 / 3) (h2 : rooms = 5 / 2) :
  rate * rooms = 65 / 6 := by sorry

end NUMINAMATH_CALUDE_jeremy_payment_l3240_324013


namespace NUMINAMATH_CALUDE_smallest_with_24_factors_div_by_18_and_30_is_360_l3240_324006

/-- The smallest integer with 24 positive factors that is divisible by both 18 and 30 -/
def smallest_with_24_factors_div_by_18_and_30 : ℕ := 360

/-- Proposition: The smallest integer with 24 positive factors that is divisible by both 18 and 30 is 360 -/
theorem smallest_with_24_factors_div_by_18_and_30_is_360 :
  ∀ y : ℕ, 
    (Finset.card (Nat.divisors y) = 24) → 
    (18 ∣ y) → 
    (30 ∣ y) → 
    y ≥ smallest_with_24_factors_div_by_18_and_30 :=
sorry

end NUMINAMATH_CALUDE_smallest_with_24_factors_div_by_18_and_30_is_360_l3240_324006


namespace NUMINAMATH_CALUDE_fourth_root_of_sum_of_powers_of_two_l3240_324082

theorem fourth_root_of_sum_of_powers_of_two :
  (2^3 + 2^4 + 2^5 + 2^6 : ℝ)^(1/4) = 2^(3/4) * 15^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_sum_of_powers_of_two_l3240_324082


namespace NUMINAMATH_CALUDE_correct_quotient_proof_l3240_324037

theorem correct_quotient_proof (N : ℕ) (h1 : N % 21 = 0) (h2 : N / 12 = 49) : N / 21 = 28 := by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_proof_l3240_324037


namespace NUMINAMATH_CALUDE_f_geq_two_range_of_x_l3240_324081

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Theorem 1: f(x) ≥ 2 for all real x
theorem f_geq_two (x : ℝ) : f x ≥ 2 := by
  sorry

-- Theorem 2: If f(x) ≥ (|2b+1| - |1-b|) / |b| for all non-zero real b,
-- then x ≤ -1.5 or x ≥ 1.5
theorem range_of_x (x : ℝ) 
  (h : ∀ b : ℝ, b ≠ 0 → f x ≥ (|2*b + 1| - |1 - b|) / |b|) : 
  x ≤ -1.5 ∨ x ≥ 1.5 := by
  sorry

end NUMINAMATH_CALUDE_f_geq_two_range_of_x_l3240_324081


namespace NUMINAMATH_CALUDE_problem_solution_l3240_324062

def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 2 * x

theorem problem_solution :
  (∀ x : ℝ, f 3 x ≥ 3 ↔ x ≥ 0) ∧
  (∀ x : ℝ, (f a x ≤ 0 ↔ x ≤ -2) → (a = 2 ∨ a = -6)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3240_324062


namespace NUMINAMATH_CALUDE_cube_opposite_face_l3240_324054

-- Define the faces of the cube
inductive Face : Type
| X | Y | Z | U | V | W

-- Define the adjacency relation
def adjacent : Face → Face → Prop := sorry

-- Define the opposite relation
def opposite : Face → Face → Prop := sorry

-- State the theorem
theorem cube_opposite_face :
  (∀ f : Face, f ≠ Face.X ∧ f ≠ Face.Y → adjacent Face.X f) →
  opposite Face.X Face.Y := by sorry

end NUMINAMATH_CALUDE_cube_opposite_face_l3240_324054


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3240_324083

/-- A quadratic function with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_coefficient (a b c : ℤ) :
  (∀ x, QuadraticFunction a b c x = a * (x - 1)^2 - 2) →
  QuadraticFunction a b c 3 = 7 →
  a = 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3240_324083


namespace NUMINAMATH_CALUDE_jacket_final_price_l3240_324051

/-- Calculates the final price of an item after two discounts and a tax --/
def finalPrice (originalPrice firstDiscount secondDiscount taxRate : ℝ) : ℝ :=
  let priceAfterFirstDiscount := originalPrice * (1 - firstDiscount)
  let priceAfterSecondDiscount := priceAfterFirstDiscount * (1 - secondDiscount)
  priceAfterSecondDiscount * (1 + taxRate)

/-- Theorem stating that the final price of the jacket is approximately $77.11 --/
theorem jacket_final_price :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (finalPrice 120 0.3 0.15 0.08 - 77.11) < ε :=
sorry

end NUMINAMATH_CALUDE_jacket_final_price_l3240_324051


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l3240_324059

/-- Calculates the average speed of a round trip given the speed of the outbound journey and the fact that the return journey takes twice as long. -/
theorem round_trip_average_speed (outbound_speed : ℝ) :
  outbound_speed = 51 →
  (2 * outbound_speed) / 3 = 34 := by
  sorry

#check round_trip_average_speed

end NUMINAMATH_CALUDE_round_trip_average_speed_l3240_324059


namespace NUMINAMATH_CALUDE_tim_prank_combinations_l3240_324066

theorem tim_prank_combinations :
  let day1_choices : ℕ := 1
  let day2_choices : ℕ := 2
  let day3_choices : ℕ := 6
  let day4_choices : ℕ := 5
  let day5_choices : ℕ := 1
  day1_choices * day2_choices * day3_choices * day4_choices * day5_choices = 60 :=
by sorry

end NUMINAMATH_CALUDE_tim_prank_combinations_l3240_324066


namespace NUMINAMATH_CALUDE_mixed_nuts_cost_per_serving_l3240_324048

/-- Calculates the cost per serving of mixed nuts in cents -/
def cost_per_serving (bag_cost : ℚ) (bag_content : ℚ) (coupon_value : ℚ) (serving_size : ℚ) : ℚ :=
  ((bag_cost - coupon_value) / bag_content) * serving_size * 100

/-- Theorem: The cost per serving of mixed nuts is 50 cents -/
theorem mixed_nuts_cost_per_serving :
  cost_per_serving 25 40 5 1 = 50 := by
  sorry

#eval cost_per_serving 25 40 5 1

end NUMINAMATH_CALUDE_mixed_nuts_cost_per_serving_l3240_324048


namespace NUMINAMATH_CALUDE_even_increasing_negative_ordering_l3240_324057

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f is increasing on (-∞, 0) if f(x) < f(y) whenever x < y < 0 -/
def IncreasingOnNegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → y < 0 → f x < f y

theorem even_increasing_negative_ordering (f : ℝ → ℝ) 
    (h_even : IsEven f) (h_incr : IncreasingOnNegative f) : 
    f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_negative_ordering_l3240_324057


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3240_324093

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 10)^2 ≥ 100 ∧
  ∃ α₀ β₀ : ℝ, (3 * Real.cos α₀ + 4 * Real.sin β₀ - 7)^2 + (3 * Real.sin α₀ + 4 * Real.cos β₀ - 10)^2 = 100 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3240_324093


namespace NUMINAMATH_CALUDE_complex_multiplication_division_l3240_324072

theorem complex_multiplication_division (z₁ z₂ : ℂ) :
  z₁ = 1 + Complex.I →
  z₂ = 2 - Complex.I →
  (z₁ * z₂) / Complex.I = 1 - 3 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_division_l3240_324072


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l3240_324064

/-- A quadratic polynomial q(x) satisfying specific conditions -/
def q (x : ℚ) : ℚ := (6/7) * x^2 - (2/7) * x + 2

/-- Theorem stating that q(x) satisfies the given conditions -/
theorem q_satisfies_conditions : 
  q (-2) = 6 ∧ q 0 = 2 ∧ q 3 = 8 := by
  sorry

#eval q (-2)
#eval q 0
#eval q 3

end NUMINAMATH_CALUDE_q_satisfies_conditions_l3240_324064


namespace NUMINAMATH_CALUDE_min_value_implies_a_bound_l3240_324011

/-- The piecewise function f(x) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 1 else a*x^2 - x + 2

/-- Theorem stating that if the minimum value of f(x) is -1, then a ≥ 1/12 --/
theorem min_value_implies_a_bound (a : ℝ) :
  (∀ x, f a x ≥ -1) ∧ (∃ x, f a x = -1) → a ≥ 1/12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_implies_a_bound_l3240_324011


namespace NUMINAMATH_CALUDE_quadrilateral_area_inequalities_l3240_324017

/-- Properties of a quadrilateral -/
structure Quadrilateral where
  S : ℝ  -- Area
  a : ℝ  -- Side length
  b : ℝ  -- Side length
  c : ℝ  -- Side length
  d : ℝ  -- Side length
  e : ℝ  -- Diagonal length
  f : ℝ  -- Diagonal length
  m : ℝ  -- Midpoint segment length
  n : ℝ  -- Midpoint segment length
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  hd : 0 < d
  he : 0 < e
  hf : 0 < f
  hm : 0 < m
  hn : 0 < n
  hS : 0 < S

/-- Theorem: Area inequalities for a quadrilateral -/
theorem quadrilateral_area_inequalities (q : Quadrilateral) :
  q.S ≤ (1/4) * (q.e^2 + q.f^2) ∧
  q.S ≤ (1/2) * (q.m^2 + q.n^2) ∧
  q.S ≤ (1/4) * (q.a + q.c) * (q.b + q.d) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_inequalities_l3240_324017


namespace NUMINAMATH_CALUDE_sqrt_108_simplification_l3240_324094

theorem sqrt_108_simplification : Real.sqrt 108 = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_108_simplification_l3240_324094


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3240_324077

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a →
  (a 5 + a 6 + a 7 = 1) →
  (a 3 + a 9 = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3240_324077


namespace NUMINAMATH_CALUDE_stock_investment_l3240_324085

theorem stock_investment (annual_income : ℝ) (stock_percentage : ℝ) (stock_price : ℝ) :
  annual_income = 2000 ∧ 
  stock_percentage = 40 ∧ 
  stock_price = 136 →
  ∃ amount_invested : ℝ, amount_invested = 6800 ∧
    annual_income = (amount_invested / stock_price) * (stock_percentage / 100) * 100 :=
by sorry

end NUMINAMATH_CALUDE_stock_investment_l3240_324085


namespace NUMINAMATH_CALUDE_theater_eye_colors_l3240_324092

theorem theater_eye_colors (total : ℕ) (blue : ℕ) (brown : ℕ) (black : ℕ) (green : ℕ)
  (h_total : total = 100)
  (h_blue : blue = 19)
  (h_brown : brown = total / 2)
  (h_black : black = total / 4)
  (h_green : green = total - (blue + brown + black)) :
  green = 6 := by
sorry

end NUMINAMATH_CALUDE_theater_eye_colors_l3240_324092


namespace NUMINAMATH_CALUDE_parabola_equation_l3240_324074

/-- Given a parabola y^2 = 2px (p > 0) and a line with slope 1 passing through its focus,
    intersecting the parabola at points A and B, if |AB| = 8, then the equation of the parabola is y^2 = 4x -/
theorem parabola_equation (p : ℝ) (A B : ℝ × ℝ) (h_p : p > 0) : 
  (∀ x y, y^2 = 2*p*x → (∃ t, y = x - p/2 + t)) →  -- Line passes through focus (p/2, 0) with slope 1
  (A.2^2 = 2*p*A.1 ∧ B.2^2 = 2*p*B.1) →            -- A and B are on the parabola
  (A.2 = A.1 - p/2 ∧ B.2 = B.1 - p/2) →            -- A and B are on the line
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64 →             -- |AB|^2 = 8^2 = 64
  (∀ x y, y^2 = 4*x ↔ y^2 = 2*p*x) :=               -- The parabola equation is y^2 = 4x
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3240_324074


namespace NUMINAMATH_CALUDE_valid_outfit_choices_l3240_324007

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of pants available -/
def num_pants : ℕ := 5

/-- The number of hats available -/
def num_hats : ℕ := 7

/-- The number of colors available for each item -/
def num_colors : ℕ := 5

/-- The total number of possible outfits -/
def total_outfits : ℕ := num_shirts * num_pants * num_hats

/-- The number of outfits where pants and hat are the same color -/
def matching_pants_hat_outfits : ℕ := num_colors * num_shirts

/-- The number of valid outfit choices -/
def valid_outfits : ℕ := total_outfits - matching_pants_hat_outfits

theorem valid_outfit_choices :
  valid_outfits = 240 :=
by sorry

end NUMINAMATH_CALUDE_valid_outfit_choices_l3240_324007


namespace NUMINAMATH_CALUDE_exists_non_intersecting_circle_exists_regular_polygon_M_properties_l3240_324021

/-- Line system M: x cos θ + (y-1) sin θ = 1, where 0 ≤ θ ≤ 2π -/
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ p.1 * Real.cos θ + (p.2 - 1) * Real.sin θ = 1}

/-- There exists a circle that does not intersect any of the lines in M -/
theorem exists_non_intersecting_circle (M : Set (ℝ × ℝ)) : 
  ∃ (c : ℝ × ℝ) (r : ℝ), ∀ p ∈ M, (p.1 - c.1)^2 + (p.2 - c.2)^2 > r^2 := by sorry

/-- For any integer n ≥ 3, there exists a regular n-sided polygon whose edges all lie on lines in M -/
theorem exists_regular_polygon (M : Set (ℝ × ℝ)) (n : ℕ) (hn : n ≥ 3) :
  ∃ (polygon : Fin n → ℝ × ℝ), 
    (∀ i : Fin n, ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ 
      (polygon i).1 * Real.cos θ + ((polygon i).2 - 1) * Real.sin θ = 1) ∧
    (∀ i j : Fin n, (polygon i).1^2 + (polygon i).2^2 = (polygon j).1^2 + (polygon j).2^2) := by sorry

/-- Main theorem combining the two properties -/
theorem M_properties : 
  (∃ (c : ℝ × ℝ) (r : ℝ), ∀ p ∈ M, (p.1 - c.1)^2 + (p.2 - c.2)^2 > r^2) ∧
  (∀ (n : ℕ), n ≥ 3 → 
    ∃ (polygon : Fin n → ℝ × ℝ), 
      (∀ i : Fin n, ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi ∧ 
        (polygon i).1 * Real.cos θ + ((polygon i).2 - 1) * Real.sin θ = 1) ∧
      (∀ i j : Fin n, (polygon i).1^2 + (polygon i).2^2 = (polygon j).1^2 + (polygon j).2^2)) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_intersecting_circle_exists_regular_polygon_M_properties_l3240_324021


namespace NUMINAMATH_CALUDE_range_of_m_for_inequality_l3240_324025

theorem range_of_m_for_inequality (m : ℝ) : 
  (∀ x : ℝ, (2 : ℝ)^(-x^2 - x) > (1/2 : ℝ)^(2*x^2 - m*x + m + 4)) ↔ 
  -3 < m ∧ m < 5 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_for_inequality_l3240_324025


namespace NUMINAMATH_CALUDE_sum_of_squares_constant_l3240_324035

/-- A regular polygon with n vertices and circumradius r -/
structure RegularPolygon where
  n : ℕ
  r : ℝ
  h_n : n ≥ 3
  h_r : r > 0

/-- The sum of squares of distances from a point on the circumcircle to all vertices -/
def sum_of_squares (poly : RegularPolygon) (P : ℝ × ℝ) : ℝ :=
  sorry

/-- The theorem stating that the sum of squares is constant for any point on the circumcircle -/
theorem sum_of_squares_constant (poly : RegularPolygon) :
  ∀ P : ℝ × ℝ, (P.1 - poly.r)^2 + P.2^2 = poly.r^2 →
    sum_of_squares poly P = 2 * poly.n * poly.r^2 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_constant_l3240_324035


namespace NUMINAMATH_CALUDE_no_good_points_iff_a_in_range_l3240_324088

def f (a x : ℝ) : ℝ := x^2 + 2*a*x + 1

def has_no_good_points (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x ≠ x

theorem no_good_points_iff_a_in_range :
  ∀ a : ℝ, has_no_good_points a ↔ -1/2 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_no_good_points_iff_a_in_range_l3240_324088


namespace NUMINAMATH_CALUDE_sheila_fewer_acorns_l3240_324039

/-- The number of acorns Shawna, Sheila, and Danny have altogether -/
def total_acorns : ℕ := 80

/-- The number of acorns Shawna has -/
def shawna_acorns : ℕ := 7

/-- The ratio of Sheila's acorns to Shawna's acorns -/
def sheila_ratio : ℕ := 5

/-- The number of acorns Sheila has -/
def sheila_acorns : ℕ := sheila_ratio * shawna_acorns

/-- The number of acorns Danny has -/
def danny_acorns : ℕ := total_acorns - sheila_acorns - shawna_acorns

/-- The difference in acorns between Danny and Sheila -/
def acorn_difference : ℕ := danny_acorns - sheila_acorns

theorem sheila_fewer_acorns : acorn_difference = 3 := by
  sorry

end NUMINAMATH_CALUDE_sheila_fewer_acorns_l3240_324039


namespace NUMINAMATH_CALUDE_small_paintings_sold_l3240_324067

/-- Given the prices of paintings and sales information, prove the number of small paintings sold. -/
theorem small_paintings_sold
  (large_price : ℕ)
  (small_price : ℕ)
  (large_sold : ℕ)
  (total_earnings : ℕ)
  (h1 : large_price = 100)
  (h2 : small_price = 80)
  (h3 : large_sold = 5)
  (h4 : total_earnings = 1140) :
  (total_earnings - large_price * large_sold) / small_price = 8 := by
  sorry

end NUMINAMATH_CALUDE_small_paintings_sold_l3240_324067


namespace NUMINAMATH_CALUDE_paths_through_B_l3240_324001

/-- The number of paths between two points on a grid -/
def grid_paths (right : ℕ) (down : ℕ) : ℕ :=
  Nat.choose (right + down) down

/-- The position of point A -/
def point_A : ℕ × ℕ := (0, 0)

/-- The position of point B relative to A -/
def A_to_B : ℕ × ℕ := (4, 2)

/-- The position of point C relative to B -/
def B_to_C : ℕ × ℕ := (3, 2)

/-- The total number of steps from A to C -/
def total_steps : ℕ := A_to_B.1 + A_to_B.2 + B_to_C.1 + B_to_C.2

theorem paths_through_B : 
  grid_paths A_to_B.1 A_to_B.2 * grid_paths B_to_C.1 B_to_C.2 = 150 ∧ 
  total_steps = 11 := by
  sorry

end NUMINAMATH_CALUDE_paths_through_B_l3240_324001


namespace NUMINAMATH_CALUDE_tan_double_angle_special_point_l3240_324019

/-- Given a point P(1, -2) in the plane, and an angle α whose terminal side passes through P,
    prove that tan(2α) = 4/3 -/
theorem tan_double_angle_special_point (α : ℝ) :
  (∃ P : ℝ × ℝ, P.1 = 1 ∧ P.2 = -2 ∧ Real.tan α = P.2 / P.1) →
  Real.tan (2 * α) = 4/3 := by
sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_point_l3240_324019


namespace NUMINAMATH_CALUDE_usual_time_to_catch_bus_l3240_324036

theorem usual_time_to_catch_bus (usual_speed : ℝ) (usual_time : ℝ) : 
  usual_time > 0 → usual_speed > 0 →
  (4/5 * usual_speed) * (usual_time + 5) = usual_speed * usual_time →
  usual_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_usual_time_to_catch_bus_l3240_324036


namespace NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l3240_324073

/-- Given a hyperbola passing through a specific point, prove the coordinates of its foci -/
theorem hyperbola_foci_coordinates :
  ∀ (a : ℝ),
  (((2 * Real.sqrt 2) ^ 2) / a ^ 2) - 1 ^ 2 = 1 →
  ∃ (c : ℝ),
  c ^ 2 = 5 ∧
  (∀ (x y : ℝ), x ^ 2 / a ^ 2 - y ^ 2 = 1 → 
    ((x = c ∧ y = 0) ∨ (x = -c ∧ y = 0))) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_coordinates_l3240_324073


namespace NUMINAMATH_CALUDE_original_raspberry_count_l3240_324055

/-- The number of lemon candies Liam originally had -/
def original_lemon : ℕ := sorry

/-- The number of raspberry candies Liam originally had -/
def original_raspberry : ℕ := sorry

/-- The condition that Liam originally had three times as many raspberry candies as lemon candies -/
axiom original_ratio : original_raspberry = 3 * original_lemon

/-- The condition that after giving away 15 raspberry candies and 5 lemon candies, 
    he has five times as many raspberry candies as lemon candies -/
axiom new_ratio : original_raspberry - 15 = 5 * (original_lemon - 5)

/-- The theorem stating that the original number of raspberry candies is 15 -/
theorem original_raspberry_count : original_raspberry = 15 := by sorry

end NUMINAMATH_CALUDE_original_raspberry_count_l3240_324055


namespace NUMINAMATH_CALUDE_f_of_g_10_l3240_324004

-- Define the functions g and f
def g (x : ℝ) : ℝ := 2 * x + 6
def f (x : ℝ) : ℝ := 4 * x - 8

-- State the theorem
theorem f_of_g_10 : f (g 10) = 96 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_10_l3240_324004


namespace NUMINAMATH_CALUDE_count_male_students_l3240_324027

theorem count_male_students (total : ℕ) (girls : ℕ) (h1 : total = 13) (h2 : girls = 6) :
  total - girls = 7 := by
  sorry

end NUMINAMATH_CALUDE_count_male_students_l3240_324027


namespace NUMINAMATH_CALUDE_divisibility_of_fourth_power_minus_one_l3240_324034

theorem divisibility_of_fourth_power_minus_one (a : ℤ) : 
  ¬(5 ∣ a) → (5 ∣ (a^4 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_fourth_power_minus_one_l3240_324034


namespace NUMINAMATH_CALUDE_max_product_sum_2000_l3240_324098

theorem max_product_sum_2000 :
  ∃ (a b : ℤ), a + b = 2000 ∧
  ∀ (x y : ℤ), x + y = 2000 → x * y ≤ a * b ∧
  a * b = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2000_l3240_324098


namespace NUMINAMATH_CALUDE_investment_problem_l3240_324010

theorem investment_problem (amount_at_5_percent : ℝ) (total_with_interest : ℝ) :
  amount_at_5_percent = 600 →
  total_with_interest = 1054 →
  ∃ (total_investment : ℝ),
    total_investment = 1034 ∧
    amount_at_5_percent + amount_at_5_percent * 0.05 +
    (total_investment - amount_at_5_percent) +
    (total_investment - amount_at_5_percent) * 0.06 = total_with_interest :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l3240_324010


namespace NUMINAMATH_CALUDE_point_on_h_graph_coordinate_sum_l3240_324030

theorem point_on_h_graph_coordinate_sum : 
  ∀ (g h : ℝ → ℝ),
  g 4 = -5 →
  (∀ x, h x = (g x)^2 + 3) →
  4 + h 4 = 32 := by
sorry

end NUMINAMATH_CALUDE_point_on_h_graph_coordinate_sum_l3240_324030


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l3240_324079

theorem multiply_mixed_number : 7 * (9 + 2/5) = 65 + 4/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l3240_324079


namespace NUMINAMATH_CALUDE_mohamed_age_ratio_l3240_324095

/-- Represents a person's age -/
structure Age :=
  (value : ℕ)

/-- Represents the current year -/
def currentYear : ℕ := 2023

theorem mohamed_age_ratio (kody : Age) (mohamed : Age) :
  kody.value = 32 →
  (currentYear - 4 : ℕ) - kody.value + 4 = 2 * ((currentYear - 4 : ℕ) - mohamed.value + 4) →
  ∃ k : ℕ, mohamed.value = 30 * k →
  mohamed.value / 30 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mohamed_age_ratio_l3240_324095


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3240_324000

/-- A geometric sequence with five terms where the first term is -1 and the last term is -2 -/
def GeometricSequence (x y z : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ x = -1 * r ∧ y = x * r ∧ z = y * r ∧ -2 = z * r

/-- The product of the middle three terms of the geometric sequence equals ±2√2 -/
theorem geometric_sequence_product (x y z : ℝ) :
  GeometricSequence x y z → x * y * z = 2 * Real.sqrt 2 ∨ x * y * z = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3240_324000


namespace NUMINAMATH_CALUDE_remainder_of_2857916_div_4_l3240_324087

theorem remainder_of_2857916_div_4 : 2857916 % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2857916_div_4_l3240_324087


namespace NUMINAMATH_CALUDE_identical_balls_distribution_seven_balls_four_boxes_l3240_324086

theorem identical_balls_distribution (n m : ℕ) (hn : n ≥ m) :
  (Nat.choose (n + m - 1) (m - 1) : ℕ) = (Nat.choose (n - 1) (m - 1) : ℕ) := by
  sorry

theorem seven_balls_four_boxes :
  (Nat.choose 6 3 : ℕ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_identical_balls_distribution_seven_balls_four_boxes_l3240_324086


namespace NUMINAMATH_CALUDE_queen_high_school_teachers_l3240_324047

/-- The number of teachers at Queen High School -/
def num_teachers (total_students : ℕ) (classes_per_student : ℕ) (students_per_class : ℕ) (classes_per_teacher : ℕ) : ℕ :=
  (total_students * classes_per_student) / (students_per_class * classes_per_teacher)

/-- Theorem: There are 72 teachers at Queen High School -/
theorem queen_high_school_teachers :
  num_teachers 1500 6 25 5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_queen_high_school_teachers_l3240_324047


namespace NUMINAMATH_CALUDE_linear_function_proof_l3240_324060

/-- A linear function passing through (-2, 0) with the form y = ax + 1 -/
def linear_function (x : ℝ) : ℝ → ℝ := λ a ↦ a * x + 1

theorem linear_function_proof :
  ∃ a : ℝ, (∀ x : ℝ, linear_function x a = (1/2) * x + 1) ∧ linear_function (-2) a = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_linear_function_proof_l3240_324060


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3240_324043

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 * y + 6) = 5 → y = (19 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3240_324043


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l3240_324090

/-- Proves that given the specified blanket purchases and average price, 
    the unknown rate for two blankets must be 275. -/
theorem unknown_blanket_rate 
  (price1 : ℕ) (count1 : ℕ) 
  (price2 : ℕ) (count2 : ℕ) 
  (count3 : ℕ) 
  (avg_price : ℕ) 
  (h1 : price1 = 100) 
  (h2 : count1 = 3) 
  (h3 : price2 = 150) 
  (h4 : count2 = 5) 
  (h5 : count3 = 2) 
  (h6 : avg_price = 160) 
  (h7 : (price1 * count1 + price2 * count2 + count3 * unknown_rate) / (count1 + count2 + count3) = avg_price) : 
  unknown_rate = 275 := by
  sorry

#check unknown_blanket_rate

end NUMINAMATH_CALUDE_unknown_blanket_rate_l3240_324090


namespace NUMINAMATH_CALUDE_original_speed_before_training_l3240_324032

/-- Represents the skipping speed of a person -/
structure SkippingSpeed :=
  (skips : ℕ)
  (minutes : ℕ)

/-- Calculates the skips per minute -/
def skipsPerMinute (speed : SkippingSpeed) : ℚ :=
  speed.skips / speed.minutes

theorem original_speed_before_training
  (after_training : SkippingSpeed)
  (h_doubles : after_training.skips = 700 ∧ after_training.minutes = 5) :
  let before_training := SkippingSpeed.mk (after_training.skips / 2) after_training.minutes
  skipsPerMinute before_training = 70 := by
sorry

end NUMINAMATH_CALUDE_original_speed_before_training_l3240_324032


namespace NUMINAMATH_CALUDE_yevgeniy_age_unique_l3240_324052

def birth_year (y : ℕ) := 1900 + y

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- Define the condition from the problem
def condition (y : ℕ) : Prop :=
  y ≥ 0 ∧ y < 100 ∧ (2011 - birth_year y = sum_of_digits (birth_year y))

-- The theorem to prove
theorem yevgeniy_age_unique :
  ∃! y : ℕ, condition y ∧ (2014 - birth_year y = 23) :=
sorry

end NUMINAMATH_CALUDE_yevgeniy_age_unique_l3240_324052


namespace NUMINAMATH_CALUDE_round_trip_percentage_l3240_324049

/-- The percentage of passengers with round-trip tickets, given the conditions -/
theorem round_trip_percentage (total_passengers : ℝ) 
  (h1 : 0 < total_passengers)
  (h2 : (0.2 : ℝ) * total_passengers = 
        (0.8 : ℝ) * (round_trip_passengers : ℝ)) : 
  round_trip_passengers / total_passengers = 0.25 := by
  sorry

#check round_trip_percentage

end NUMINAMATH_CALUDE_round_trip_percentage_l3240_324049


namespace NUMINAMATH_CALUDE_unique_satisfying_polynomial_l3240_324005

/-- A quadratic polynomial with real coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The roots of a quadratic polynomial -/
def roots (p : QuadraticPolynomial) : Set ℝ :=
  {r : ℝ | p.a * r^2 + p.b * r + p.c = 0}

/-- The coefficients of a quadratic polynomial -/
def coefficients (p : QuadraticPolynomial) : Set ℝ :=
  {p.a, p.b, p.c}

/-- Predicate for a polynomial satisfying the problem conditions -/
def satisfies_conditions (p : QuadraticPolynomial) : Prop :=
  roots p = coefficients p ∧
  (p.a < 0 ∨ p.b < 0 ∨ p.c < 0)

/-- The main theorem stating that exactly one quadratic polynomial satisfies the conditions -/
theorem unique_satisfying_polynomial :
  ∃! p : QuadraticPolynomial, satisfies_conditions p :=
sorry

end NUMINAMATH_CALUDE_unique_satisfying_polynomial_l3240_324005


namespace NUMINAMATH_CALUDE_power_seven_twelve_mod_hundred_l3240_324018

theorem power_seven_twelve_mod_hundred : 7^12 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_twelve_mod_hundred_l3240_324018


namespace NUMINAMATH_CALUDE_quadratic_equation_nonnegative_integer_solutions_l3240_324089

theorem quadratic_equation_nonnegative_integer_solutions :
  ∃! (x : ℕ), x^2 + x - 6 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_nonnegative_integer_solutions_l3240_324089


namespace NUMINAMATH_CALUDE_triangle_problem_l3240_324003

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if √3 * sin(C) + c * cos(A) = a + b, c = 2, and the area is √3,
    then C = π/3 and the perimeter is 6. -/
theorem triangle_problem (a b c A B C : ℝ) : 
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  Real.sqrt 3 * Real.sin C + c * Real.cos A = a + b →
  c = 2 →
  (1/2) * a * b * Real.sin C = Real.sqrt 3 →
  C = π/3 ∧ a + b + c = 6 := by sorry

end NUMINAMATH_CALUDE_triangle_problem_l3240_324003


namespace NUMINAMATH_CALUDE_dress_price_difference_l3240_324053

theorem dress_price_difference (original_price : ℝ) : 
  (0.85 * original_price = 85) →
  (original_price - (85 + 0.25 * 85) = -6.25) := by
sorry

end NUMINAMATH_CALUDE_dress_price_difference_l3240_324053


namespace NUMINAMATH_CALUDE_combination_not_equal_permutation_div_n_factorial_l3240_324008

/-- The number of combinations of n things taken m at a time -/
def C (n m : ℕ) : ℕ := sorry

/-- The number of permutations of n things taken m at a time -/
def A (n m : ℕ) : ℕ := sorry

theorem combination_not_equal_permutation_div_n_factorial (n m : ℕ) :
  C n m ≠ A n m / n! :=
sorry

end NUMINAMATH_CALUDE_combination_not_equal_permutation_div_n_factorial_l3240_324008


namespace NUMINAMATH_CALUDE_total_money_divided_l3240_324033

/-- Proves that the total amount of money divided is 1600, given the specified conditions. -/
theorem total_money_divided (x : ℝ) (T : ℝ) : 
  x + (T - x) = T →  -- The money is divided into two parts
  0.06 * x + 0.05 * (T - x) = 85 →  -- The whole annual interest from both parts is 85
  T - x = 1100 →  -- 1100 was lent at approximately 5%
  T = 1600 := by
  sorry

end NUMINAMATH_CALUDE_total_money_divided_l3240_324033


namespace NUMINAMATH_CALUDE_books_read_during_trip_l3240_324029

-- Define the travel distance in miles
def travel_distance : ℕ := 6760

-- Define the reading rate in miles per book
def miles_per_book : ℕ := 450

-- Theorem to prove
theorem books_read_during_trip : travel_distance / miles_per_book = 15 := by
  sorry

end NUMINAMATH_CALUDE_books_read_during_trip_l3240_324029


namespace NUMINAMATH_CALUDE_sock_time_correct_l3240_324014

/-- Represents the time (in hours) to knit each sock -/
def sock_time : ℝ := 1.5

/-- Represents the number of grandchildren -/
def num_grandchildren : ℕ := 3

/-- Time (in hours) to knit a hat -/
def hat_time : ℝ := 2

/-- Time (in hours) to knit a scarf -/
def scarf_time : ℝ := 3

/-- Time (in hours) to knit a sweater -/
def sweater_time : ℝ := 6

/-- Time (in hours) to knit each mitten -/
def mitten_time : ℝ := 1

/-- Total time (in hours) to knit all outfits -/
def total_time : ℝ := 48

/-- Theorem stating that the calculated sock_time satisfies the given conditions -/
theorem sock_time_correct : 
  num_grandchildren * (hat_time + scarf_time + sweater_time + 2 * mitten_time + 2 * sock_time) = total_time := by
  sorry

end NUMINAMATH_CALUDE_sock_time_correct_l3240_324014


namespace NUMINAMATH_CALUDE_shirts_sold_l3240_324070

/-- Proves that the number of shirts sold is 4 given the conditions of the problem -/
theorem shirts_sold (total_money : ℕ) (num_dresses : ℕ) (price_dress : ℕ) (price_shirt : ℕ) :
  total_money = 69 →
  num_dresses = 7 →
  price_dress = 7 →
  price_shirt = 5 →
  (total_money - num_dresses * price_dress) / price_shirt = 4 := by
  sorry

end NUMINAMATH_CALUDE_shirts_sold_l3240_324070


namespace NUMINAMATH_CALUDE_sum_a_b_equals_one_l3240_324023

theorem sum_a_b_equals_one (a b : ℝ) (h : Real.sqrt (a - b - 3) + |2 * a - 4| = 0) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_one_l3240_324023


namespace NUMINAMATH_CALUDE_parking_cost_excess_hours_l3240_324071

theorem parking_cost_excess_hours (base_cost : ℝ) (avg_cost : ℝ) (excess_cost : ℝ) : 
  base_cost = 10 →
  avg_cost = 2.4722222222222223 →
  (base_cost + 7 * excess_cost) / 9 = avg_cost →
  excess_cost = 1.75 := by
sorry

end NUMINAMATH_CALUDE_parking_cost_excess_hours_l3240_324071


namespace NUMINAMATH_CALUDE_time_to_hear_second_blast_l3240_324012

/-- The time taken for a man to hear a second blast, given specific conditions -/
theorem time_to_hear_second_blast 
  (speed_of_sound : ℝ) 
  (time_between_blasts : ℝ) 
  (distance_at_second_blast : ℝ) 
  (h1 : speed_of_sound = 330)
  (h2 : time_between_blasts = 30 * 60)
  (h3 : distance_at_second_blast = 4950) :
  speed_of_sound * (time_between_blasts + distance_at_second_blast / speed_of_sound) = 1815 * speed_of_sound :=
by sorry

end NUMINAMATH_CALUDE_time_to_hear_second_blast_l3240_324012


namespace NUMINAMATH_CALUDE_exam_mean_score_l3240_324063

/-- Given an exam where a score of 86 is 7 standard deviations below the mean,
    and a score of 90 is 3 standard deviations above the mean,
    prove that the mean score is 88.8 -/
theorem exam_mean_score (μ σ : ℝ) 
    (h1 : 86 = μ - 7 * σ) 
    (h2 : 90 = μ + 3 * σ) : 
  μ = 88.8 := by
sorry

end NUMINAMATH_CALUDE_exam_mean_score_l3240_324063


namespace NUMINAMATH_CALUDE_optimal_price_and_profit_l3240_324044

/-- Represents the monthly sales quantity as a function of price -/
def sales_quantity (x : ℝ) : ℝ := -10000 * x + 80000

/-- Represents the monthly profit as a function of price -/
def monthly_profit (x : ℝ) : ℝ := (x - 4) * (sales_quantity x)

theorem optimal_price_and_profit :
  let price_1 : ℝ := 5
  let quantity_1 : ℝ := 30000
  let price_2 : ℝ := 6
  let quantity_2 : ℝ := 20000
  let unit_cost : ℝ := 4
  
  -- The sales quantity function is correct
  (∀ x, sales_quantity x = -10000 * x + 80000) ∧
  
  -- The function satisfies the given points
  (sales_quantity price_1 = quantity_1) ∧
  (sales_quantity price_2 = quantity_2) ∧
  
  -- The optimal price is 6
  (∀ x, monthly_profit x ≤ monthly_profit 6) ∧
  
  -- The maximum monthly profit is 40000
  (monthly_profit 6 = 40000) := by
    sorry

end NUMINAMATH_CALUDE_optimal_price_and_profit_l3240_324044


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3240_324015

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, -x^2 + 2*x + 3 ≤ a^2 - 3*a) ↔ 
  (a ≤ -1 ∨ a ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3240_324015


namespace NUMINAMATH_CALUDE_triangle_area_l3240_324046

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side a
  c : ℝ  -- Side c

-- Define the conditions of the problem
def problem_triangle : Triangle where
  A := sorry
  B := sorry
  C := sorry
  a := 2
  c := 5

-- Define the arithmetic sequence property
def is_arithmetic_sequence (t : Triangle) : Prop :=
  t.A + t.C = 2 * t.B

-- Define the angle sum property
def angle_sum (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi

-- Theorem statement
theorem triangle_area (t : Triangle) 
  (h1 : is_arithmetic_sequence t) 
  (h2 : angle_sum t) 
  (h3 : t.a = 2) 
  (h4 : t.c = 5) : 
  (1/2 : ℝ) * t.a * t.c * Real.sin t.B = (5 * Real.sqrt 3) / 2 := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_triangle_area_l3240_324046
