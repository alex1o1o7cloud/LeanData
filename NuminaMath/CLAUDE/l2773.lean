import Mathlib

namespace NUMINAMATH_CALUDE_three_pipes_fill_time_l2773_277373

-- Define the tank volume and pipe rates
variable (T : ℝ) -- Tank volume
variable (X Y Z : ℝ) -- Filling rates of pipes X, Y, and Z

-- Define the conditions
axiom fill_XY : T = 3 * (X + Y)
axiom fill_XZ : T = 6 * (X + Z)
axiom fill_YZ : T = 4.5 * (Y + Z)

-- Define the theorem
theorem three_pipes_fill_time : 
  T / (X + Y + Z) = 36 / 11 := by sorry

end NUMINAMATH_CALUDE_three_pipes_fill_time_l2773_277373


namespace NUMINAMATH_CALUDE_sqrt_seven_minus_a_l2773_277323

theorem sqrt_seven_minus_a (a : ℝ) : a = -1 → Real.sqrt (7 - a) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_minus_a_l2773_277323


namespace NUMINAMATH_CALUDE_triangle_cosine_sum_less_than_two_l2773_277386

theorem triangle_cosine_sum_less_than_two (α β γ : Real) 
  (h_triangle : α + β + γ = Real.pi) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) : 
  Real.cos α + Real.cos β + Real.cos γ < 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_sum_less_than_two_l2773_277386


namespace NUMINAMATH_CALUDE_total_litter_weight_l2773_277385

/-- The amount of litter collected by Gina and her neighborhood --/
def litterCollection (gina_bags : ℕ) (neighborhood_multiplier : ℕ) (weight_per_bag : ℕ) : ℕ :=
  let total_bags := gina_bags + gina_bags * neighborhood_multiplier
  total_bags * weight_per_bag

/-- Theorem stating the total weight of litter collected --/
theorem total_litter_weight :
  litterCollection 2 82 4 = 664 := by
  sorry

end NUMINAMATH_CALUDE_total_litter_weight_l2773_277385


namespace NUMINAMATH_CALUDE_enclosed_area_equals_eight_thirds_l2773_277338

-- Define the functions f and g
def f (x : ℝ) : ℝ := -2 * x^2 + 7 * x - 6
def g (x : ℝ) : ℝ := -x

-- Define the theorem
theorem enclosed_area_equals_eight_thirds :
  ∃ (a b : ℝ), a < b ∧
  (∫ x in a..b, f x - g x) = 8/3 :=
sorry

end NUMINAMATH_CALUDE_enclosed_area_equals_eight_thirds_l2773_277338


namespace NUMINAMATH_CALUDE_total_amount_paid_l2773_277327

def apple_quantity : ℕ := 8
def apple_rate : ℕ := 70
def mango_quantity : ℕ := 9
def mango_rate : ℕ := 70

theorem total_amount_paid : 
  apple_quantity * apple_rate + mango_quantity * mango_rate = 1190 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l2773_277327


namespace NUMINAMATH_CALUDE_integer_sum_problem_l2773_277329

theorem integer_sum_problem (x y : ℕ) (h1 : x > y) (h2 : x - y = 16) (h3 : x * y = 63) :
  x + y = 2 * Real.sqrt 127 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l2773_277329


namespace NUMINAMATH_CALUDE_shoe_color_probability_l2773_277324

theorem shoe_color_probability (n : ℕ) (k : ℕ) (p : ℕ) :
  n = 2 * p →
  k = 3 →
  p = 6 →
  (1 - (n * (n - 2) * (n - 4) / (k * (k - 1) * (k - 2))) / (n.choose k)) = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_shoe_color_probability_l2773_277324


namespace NUMINAMATH_CALUDE_sqrt_20_minus_1_range_l2773_277350

theorem sqrt_20_minus_1_range : 3 < Real.sqrt 20 - 1 ∧ Real.sqrt 20 - 1 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_20_minus_1_range_l2773_277350


namespace NUMINAMATH_CALUDE_inequality_proof_l2773_277326

theorem inequality_proof (x y : ℝ) (m n : ℤ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  (1 - x^n.toNat)^m.toNat + (1 - y^m.toNat)^n.toNat ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2773_277326


namespace NUMINAMATH_CALUDE_count_decimals_near_three_elevenths_l2773_277369

theorem count_decimals_near_three_elevenths :
  let lower_bound : ℚ := 2614 / 10000
  let upper_bound : ℚ := 2792 / 10000
  let count := (upper_bound * 10000).floor.toNat - (lower_bound * 10000).ceil.toNat + 1
  (∀ s : ℚ, lower_bound ≤ s → s ≤ upper_bound →
    (∃ w x y z : ℕ, w < 10 ∧ x < 10 ∧ y < 10 ∧ z < 10 ∧
      s = (w * 1000 + x * 100 + y * 10 + z) / 10000) →
    (∀ n d : ℕ, n ≤ 3 → 0 < d → |s - n / d| ≥ |s - 3 / 11|)) →
  count = 179 := by
sorry

end NUMINAMATH_CALUDE_count_decimals_near_three_elevenths_l2773_277369


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2773_277372

/-- A geometric sequence with first term 2 and satisfying a₄a₆ = 4a₇² has a₃ = 1 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) (h1 : a 1 = 2) 
  (h2 : a 4 * a 6 = 4 * (a 7)^2) (h3 : ∀ n : ℕ, n ≥ 1 → ∃ q : ℝ, a (n + 1) = a n * q) : 
  a 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2773_277372


namespace NUMINAMATH_CALUDE_stating_min_drops_required_stating_drops_18_sufficient_min_drops_is_18_l2773_277315

/-- Represents the number of floors in the building -/
def num_floors : ℕ := 163

/-- Represents the number of test phones available -/
def num_phones : ℕ := 2

/-- Represents a strategy for dropping phones -/
structure DropStrategy where
  num_drops : ℕ
  can_determine_all_cases : Bool

/-- 
Theorem stating that 18 drops is the minimum number required 
to determine the breaking floor or conclude the phone is unbreakable
-/
theorem min_drops_required : 
  ∀ (s : DropStrategy), s.can_determine_all_cases → s.num_drops ≥ 18 := by
  sorry

/-- 
Theorem stating that 18 drops is sufficient to determine 
the breaking floor or conclude the phone is unbreakable
-/
theorem drops_18_sufficient : 
  ∃ (s : DropStrategy), s.can_determine_all_cases ∧ s.num_drops = 18 := by
  sorry

/-- 
Main theorem combining the above results to prove that 18 
is the minimum number of drops required
-/
theorem min_drops_is_18 : 
  (∃ (s : DropStrategy), s.can_determine_all_cases ∧ 
    (∀ (t : DropStrategy), t.can_determine_all_cases → s.num_drops ≤ t.num_drops)) ∧
  (∃ (s : DropStrategy), s.can_determine_all_cases ∧ s.num_drops = 18) := by
  sorry

end NUMINAMATH_CALUDE_stating_min_drops_required_stating_drops_18_sufficient_min_drops_is_18_l2773_277315


namespace NUMINAMATH_CALUDE_sqrt_equation_solvability_l2773_277392

theorem sqrt_equation_solvability (a : ℝ) :
  (∃ x : ℝ, Real.sqrt x - Real.sqrt (x - a) = 2) ↔ a ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solvability_l2773_277392


namespace NUMINAMATH_CALUDE_vectors_not_coplanar_l2773_277312

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given that {a, b, c} form a basis for a space V, prove that {a + b, a - b, c} are not coplanar -/
theorem vectors_not_coplanar (a b c : V) 
  (h : LinearIndependent ℝ ![a, b, c]) :
  ¬ (∃ (x y z : ℝ), x • (a + b) + y • (a - b) + z • c = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_vectors_not_coplanar_l2773_277312


namespace NUMINAMATH_CALUDE_factorization_proof_l2773_277337

theorem factorization_proof (a b : ℝ) : a * b^2 - 9 * a = a * (b + 3) * (b - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2773_277337


namespace NUMINAMATH_CALUDE_speaker_combinations_l2773_277349

/-- Represents the number of representatives for each company -/
def company_reps : List ℕ := [2, 1, 1, 1, 1]

/-- The total number of companies -/
def num_companies : ℕ := company_reps.length

/-- The number of speakers required -/
def num_speakers : ℕ := 3

/-- Calculates the number of ways to choose speakers from different companies -/
def choose_speakers (reps : List ℕ) (k : ℕ) : ℕ := sorry

theorem speaker_combinations :
  choose_speakers company_reps num_speakers = 16 := by sorry

end NUMINAMATH_CALUDE_speaker_combinations_l2773_277349


namespace NUMINAMATH_CALUDE_number_separation_l2773_277343

theorem number_separation (a b : ℝ) (h1 : a = 50) (h2 : 0.40 * a = 0.625 * b + 10) : a + b = 66 := by
  sorry

end NUMINAMATH_CALUDE_number_separation_l2773_277343


namespace NUMINAMATH_CALUDE_car_endpoint_locus_l2773_277321

-- Define the car's properties
structure Car where
  r₁ : ℝ
  r₂ : ℝ
  start : ℝ × ℝ
  s : ℝ
  α : ℝ

-- Define the circles W₁ and W₂
def W₁ (car : Car) (α₂ : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (center : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = (2*(car.r₁ - car.r₂)*Real.sin α₂)^2}

def W₂ (car : Car) (α₁ : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (center : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = (2*(car.r₁ - car.r₂)*Real.sin α₁)^2}

-- State the theorem
theorem car_endpoint_locus (car : Car) (α₁ α₂ : ℝ) 
  (h₁ : car.r₁ > car.r₂)
  (h₂ : car.α < 2 * Real.pi)
  (h₃ : α₁ + α₂ = car.α)
  (h₄ : car.r₁ * α₁ + car.r₂ * α₂ = car.s) :
  ∃ (endpoints : Set (ℝ × ℝ)), endpoints = W₁ car α₂ ∩ W₂ car α₁ := by
  sorry

end NUMINAMATH_CALUDE_car_endpoint_locus_l2773_277321


namespace NUMINAMATH_CALUDE_acute_angle_relationship_l2773_277381

theorem acute_angle_relationship (α β : Real) : 
  0 < α ∧ α < π / 2 →
  0 < β ∧ β < π / 2 →
  2 * Real.sin α = Real.sin α * Real.cos β + Real.cos α * Real.sin β →
  α < β := by
sorry

end NUMINAMATH_CALUDE_acute_angle_relationship_l2773_277381


namespace NUMINAMATH_CALUDE_triangle_construction_pieces_l2773_277351

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Calculates the sum of the first n natural numbers -/
def triangularNumber (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Properties of the triangle construction -/
structure TriangleConstruction where
  rodRows : ℕ
  firstRowRods : ℕ
  rodIncrease : ℕ
  connectorRows : ℕ

/-- Theorem statement for the triangle construction problem -/
theorem triangle_construction_pieces 
  (t : TriangleConstruction) 
  (h1 : t.rodRows = 10)
  (h2 : t.firstRowRods = 4)
  (h3 : t.rodIncrease = 4)
  (h4 : t.connectorRows = t.rodRows + 1) :
  arithmeticSum t.firstRowRods t.rodIncrease t.rodRows + triangularNumber t.connectorRows = 286 := by
  sorry

end NUMINAMATH_CALUDE_triangle_construction_pieces_l2773_277351


namespace NUMINAMATH_CALUDE_total_amount_correct_l2773_277340

/-- The total amount earned from selling notebooks -/
def total_amount (a b : ℝ) : ℝ :=
  70 * (1 + 0.2) * a + 30 * (a - b)

/-- Proof that the total amount is correct -/
theorem total_amount_correct (a b : ℝ) :
  let total_notebooks : ℕ := 100
  let first_batch : ℕ := 70
  let price_increase : ℝ := 0.2
  total_amount a b = first_batch * (1 + price_increase) * a + (total_notebooks - first_batch) * (a - b) :=
by sorry

end NUMINAMATH_CALUDE_total_amount_correct_l2773_277340


namespace NUMINAMATH_CALUDE_rectangle_sequence_area_stage_6_l2773_277335

/-- Calculates the area of a rectangle sequence up to a given stage -/
def rectangleSequenceArea (stage : ℕ) : ℕ :=
  let baseWidth := 2
  let length := 3
  List.range stage |>.map (fun i => (baseWidth + i) * length) |>.sum

/-- The area of the rectangle sequence at Stage 6 is 81 square inches -/
theorem rectangle_sequence_area_stage_6 :
  rectangleSequenceArea 6 = 81 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_sequence_area_stage_6_l2773_277335


namespace NUMINAMATH_CALUDE_homework_problem_l2773_277330

theorem homework_problem (total_problems : ℕ) (finished_problems : ℕ) (remaining_pages : ℕ) 
  (x y : ℕ) (h1 : total_problems = 450) (h2 : finished_problems = 185) (h3 : remaining_pages = 15) :
  ∃ (odd_pages even_pages : ℕ), 
    odd_pages + even_pages = remaining_pages ∧ 
    odd_pages * x + even_pages * y = total_problems - finished_problems :=
by sorry

end NUMINAMATH_CALUDE_homework_problem_l2773_277330


namespace NUMINAMATH_CALUDE_exists_number_satisfying_equation_l2773_277387

theorem exists_number_satisfying_equation : ∃ x : ℝ, (x * 7) / (10 * 17) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_exists_number_satisfying_equation_l2773_277387


namespace NUMINAMATH_CALUDE_sum_of_max_min_l2773_277383

theorem sum_of_max_min (a b c d : ℝ) (ha : a = 0.11) (hb : b = 0.98) (hc : c = 3/4) (hd : d = 2/3) :
  (max a (max b (max c d))) + (min a (min b (min c d))) = 1.09 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_l2773_277383


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2773_277328

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360 * (2 * Real.pi * r₁)) = (48 / 360 * (2 * Real.pi * r₂)) →
  (Real.pi * r₁^2) / (Real.pi * r₂^2) = 16 / 25 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2773_277328


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l2773_277314

def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 + 2 * a 2 = 4 →
  a 4 ^ 2 = 4 * a 3 * a 7 →
  a 5 = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l2773_277314


namespace NUMINAMATH_CALUDE_prob_same_length_is_17_35_l2773_277397

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of segments (sides and diagonals) in a regular hexagon -/
def total_segments : ℕ := num_sides + num_diagonals

/-- The number of pairs of segments with the same length -/
def same_length_pairs : ℕ := (num_sides.choose 2) + (num_diagonals.choose 2)

/-- The total number of possible pairs of segments -/
def total_pairs : ℕ := total_segments.choose 2

/-- The probability of selecting two segments of the same length -/
def prob_same_length : ℚ := same_length_pairs / total_pairs

theorem prob_same_length_is_17_35 : prob_same_length = 17 / 35 := by sorry

end NUMINAMATH_CALUDE_prob_same_length_is_17_35_l2773_277397


namespace NUMINAMATH_CALUDE_inscribed_parallelogram_exists_l2773_277361

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (P : Point)
  (Q : Point)
  (R : Point)
  (S : Point)

/-- Checks if a point lies on a line -/
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Checks if a parallelogram is inscribed in a quadrilateral -/
def Parallelogram.inscribed_in (p : Parallelogram) (q : Quadrilateral) : Prop :=
  (p.P.on_line (Line.mk q.A.x q.A.y (-1)) ∨ p.P.on_line (Line.mk q.A.x q.B.y (-1)) ∨
   p.P.on_line (Line.mk q.B.x q.B.y (-1)) ∨ p.P.on_line (Line.mk q.C.x q.C.y (-1))) ∧
  (p.Q.on_line (Line.mk q.A.x q.A.y (-1)) ∨ p.Q.on_line (Line.mk q.A.x q.B.y (-1)) ∨
   p.Q.on_line (Line.mk q.B.x q.B.y (-1)) ∨ p.Q.on_line (Line.mk q.C.x q.C.y (-1))) ∧
  (p.R.on_line (Line.mk q.B.x q.B.y (-1)) ∨ p.R.on_line (Line.mk q.B.x q.C.y (-1)) ∨
   p.R.on_line (Line.mk q.C.x q.C.y (-1)) ∨ p.R.on_line (Line.mk q.D.x q.D.y (-1))) ∧
  (p.S.on_line (Line.mk q.A.x q.A.y (-1)) ∨ p.S.on_line (Line.mk q.C.x q.D.y (-1)) ∨
   p.S.on_line (Line.mk q.D.x q.D.y (-1)) ∨ p.S.on_line (Line.mk q.A.x q.D.y (-1)))

/-- Checks if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The main theorem -/
theorem inscribed_parallelogram_exists (ABCD : Quadrilateral) 
  (E : Point) (F : Point) (BF CE : Line) :
  E.on_line (Line.mk ABCD.A.x ABCD.B.y (-1)) →
  F.on_line (Line.mk ABCD.C.x ABCD.D.y (-1)) →
  ∃ (PQRS : Parallelogram),
    PQRS.inscribed_in ABCD ∧
    Line.parallel (Line.mk PQRS.P.x PQRS.Q.y (-1)) BF ∧
    Line.parallel (Line.mk PQRS.Q.x PQRS.R.y (-1)) CE :=
  sorry

end NUMINAMATH_CALUDE_inscribed_parallelogram_exists_l2773_277361


namespace NUMINAMATH_CALUDE_geometric_arithmetic_progression_sum_l2773_277309

/-- Given a, b, c in geometric progression, a, m, b in arithmetic progression,
    and b, n, c in arithmetic progression, prove that m/a + n/c = 2 -/
theorem geometric_arithmetic_progression_sum (a b c m n : ℝ) 
  (h_geom : b/a = c/b)  -- a, b, c in geometric progression
  (h_arith1 : 2*m = a + b)  -- a, m, b in arithmetic progression
  (h_arith2 : 2*n = b + c)  -- b, n, c in arithmetic progression
  : m/a + n/c = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_progression_sum_l2773_277309


namespace NUMINAMATH_CALUDE_average_score_is_94_l2773_277382

/-- The average math test score of Clyde's four children -/
def average_score (june_score patty_score josh_score henry_score : ℕ) : ℚ :=
  (june_score + patty_score + josh_score + henry_score : ℚ) / 4

/-- Theorem stating that the average math test score of Clyde's four children is 94 -/
theorem average_score_is_94 :
  average_score 97 85 100 94 = 94 := by sorry

end NUMINAMATH_CALUDE_average_score_is_94_l2773_277382


namespace NUMINAMATH_CALUDE_sequence_representation_l2773_277377

theorem sequence_representation (a : ℕ → ℝ) 
  (h0 : a 0 = 4)
  (h1 : a 1 = 22)
  (h_rec : ∀ n : ℕ, n ≥ 2 → a n - 6 * a (n - 1) + a (n - 2) = 0) :
  ∃ x y : ℕ → ℕ, ∀ n : ℕ, a n = (y n ^ 2 + 7) / (x n - y n) := by
sorry

end NUMINAMATH_CALUDE_sequence_representation_l2773_277377


namespace NUMINAMATH_CALUDE_appropriate_grouping_l2773_277325

theorem appropriate_grouping : 
  (43 + 27) + ((-78) + (-52)) = 43 + (-78) + 27 + (-52) := by
  sorry

end NUMINAMATH_CALUDE_appropriate_grouping_l2773_277325


namespace NUMINAMATH_CALUDE_profit_calculation_l2773_277362

/-- Calculates the total profit of a business given the investments and one partner's share of the profit -/
def calculate_total_profit (investment_A investment_B investment_C share_A : ℕ) : ℕ :=
  let ratio_A := investment_A / (Nat.gcd investment_A (Nat.gcd investment_B investment_C))
  let ratio_B := investment_B / (Nat.gcd investment_A (Nat.gcd investment_B investment_C))
  let ratio_C := investment_C / (Nat.gcd investment_A (Nat.gcd investment_B investment_C))
  let total_ratio := ratio_A + ratio_B + ratio_C
  (share_A * total_ratio) / ratio_A

theorem profit_calculation (investment_A investment_B investment_C share_A : ℕ) 
  (h1 : investment_A = 6300)
  (h2 : investment_B = 4200)
  (h3 : investment_C = 10500)
  (h4 : share_A = 3900) :
  calculate_total_profit investment_A investment_B investment_C share_A = 13000 := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l2773_277362


namespace NUMINAMATH_CALUDE_lower_limit_of_b_l2773_277384

theorem lower_limit_of_b (a b : ℤ) (h1 : 8 < a ∧ a < 15) (h2 : b < 21) 
  (h3 : (14 : ℚ) / b - (9 : ℚ) / b = (155 : ℚ) / 100) : 4 ≤ b := by
  sorry

end NUMINAMATH_CALUDE_lower_limit_of_b_l2773_277384


namespace NUMINAMATH_CALUDE_f_neg_three_gt_f_neg_pi_l2773_277342

/-- A function f satisfying the given condition -/
def StrictlyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, (x₁ - x₂) * (f x₁ - f x₂) > 0

/-- Theorem stating that f(-3) > f(-π) given the condition -/
theorem f_neg_three_gt_f_neg_pi (f : ℝ → ℝ) (h : StrictlyIncreasing f) :
  f (-3) > f (-Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_f_neg_three_gt_f_neg_pi_l2773_277342


namespace NUMINAMATH_CALUDE_picture_frame_length_l2773_277355

/-- Given a rectangular frame with perimeter 30 cm and width 10 cm, its length is 5 cm. -/
theorem picture_frame_length (perimeter width : ℝ) (h1 : perimeter = 30) (h2 : width = 10) :
  let length := (perimeter - 2 * width) / 2
  length = 5 := by sorry

end NUMINAMATH_CALUDE_picture_frame_length_l2773_277355


namespace NUMINAMATH_CALUDE_intersection_distance_l2773_277365

/-- A cube with vertices at (0,0,0), (6,0,0), (6,6,0), (0,6,0), (0,0,6), (6,0,6), (6,6,6), and (0,6,6) -/
def cube : Set (Fin 3 → ℝ) :=
  {v | ∀ i, v i ∈ ({0, 6} : Set ℝ)}

/-- The plane cutting the cube -/
def plane (x y z : ℝ) : Prop :=
  -3 * x + 10 * y + 4 * z = 30

/-- The plane cuts the edges of the cube at these points -/
axiom plane_cuts : plane 0 3 0 ∧ plane 6 0 3 ∧ plane 2 6 6

/-- The intersection point on the edge from (0,0,0) to (0,0,6) -/
def U : Fin 3 → ℝ := ![0, 0, 3]

/-- The intersection point on the edge from (6,6,0) to (6,6,6) -/
def V : Fin 3 → ℝ := ![6, 6, 3]

/-- The theorem to be proved -/
theorem intersection_distance : 
  U ∈ cube ∧ V ∈ cube ∧ plane (U 0) (U 1) (U 2) ∧ plane (V 0) (V 1) (V 2) →
  Real.sqrt (((U 0 - V 0)^2 + (U 1 - V 1)^2 + (U 2 - V 2)^2) : ℝ) = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l2773_277365


namespace NUMINAMATH_CALUDE_line_of_sight_not_blocked_l2773_277347

/-- The curve C: y = 2x^2 -/
def C : ℝ → ℝ := λ x ↦ 2 * x^2

/-- Point A: (0, -2) -/
def A : ℝ × ℝ := (0, -2)

/-- Point B: (3, a), where a is a parameter -/
def B (a : ℝ) : ℝ × ℝ := (3, a)

/-- The line of sight from A to B(a) is not blocked by C if and only if a < 10 -/
theorem line_of_sight_not_blocked (a : ℝ) : 
  (∀ x ∈ Set.Icc A.1 (B a).1, (B a).2 - A.2 > (C x - A.2) * ((B a).1 - A.1) / (x - A.1)) ↔ 
  a < 10 :=
sorry

end NUMINAMATH_CALUDE_line_of_sight_not_blocked_l2773_277347


namespace NUMINAMATH_CALUDE_x_coordinate_C_l2773_277354

/-- Parabola function -/
def parabola (x : ℝ) : ℝ := x^2

/-- Triangle ABC with vertices on parabola y = x^2 -/
structure TriangleABC where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h_A : A.2 = parabola A.1
  h_B : B.2 = parabola B.1
  h_C : C.2 = parabola C.1
  h_A_origin : A = (0, 0)
  h_B_coords : B = (-3, 9)
  h_C_positive : C.1 > 0
  h_BC_parallel : B.2 = C.2
  h_area : (1/2) * |C.1 + 3| * C.2 = 45

/-- The x-coordinate of vertex C is 7 -/
theorem x_coordinate_C (t : TriangleABC) : t.C.1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_coordinate_C_l2773_277354


namespace NUMINAMATH_CALUDE_quadratic_intercepts_l2773_277390

/-- Given a quadratic function y = x^2 + bx - 3 that passes through the point (3,0),
    prove that b = -2 and the other x-intercept is at (-1,0) -/
theorem quadratic_intercepts (b : ℝ) : 
  (3^2 + 3*b - 3 = 0) → 
  (b = -2 ∧ (-1)^2 + (-1)*b - 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intercepts_l2773_277390


namespace NUMINAMATH_CALUDE_unique_intersection_point_l2773_277389

/-- Two equations y = x^2 and y = 2x + k intersect at exactly one point if and only if k = 0 -/
theorem unique_intersection_point (k : ℝ) : 
  (∃! x : ℝ, x^2 = 2*x + k) ↔ k = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l2773_277389


namespace NUMINAMATH_CALUDE_least_common_multiple_of_pack_sizes_l2773_277399

theorem least_common_multiple_of_pack_sizes (tulip_pack_size daffodil_pack_size : ℕ) 
  (h1 : tulip_pack_size = 15) 
  (h2 : daffodil_pack_size = 16) : 
  Nat.lcm tulip_pack_size daffodil_pack_size = 240 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_of_pack_sizes_l2773_277399


namespace NUMINAMATH_CALUDE_mechanic_job_hours_l2773_277366

theorem mechanic_job_hours (hourly_rate parts_cost total_bill : ℕ) : 
  hourly_rate = 45 → parts_cost = 225 → total_bill = 450 → 
  ∃ hours : ℕ, hours * hourly_rate + parts_cost = total_bill ∧ hours = 5 := by
  sorry

end NUMINAMATH_CALUDE_mechanic_job_hours_l2773_277366


namespace NUMINAMATH_CALUDE_company_survey_l2773_277395

theorem company_survey (total employees_with_tool employees_with_training employees_with_both : ℕ)
  (h_total : total = 150)
  (h_tool : employees_with_tool = 90)
  (h_training : employees_with_training = 60)
  (h_both : employees_with_both = 30) :
  (↑(total - (employees_with_tool + employees_with_training - employees_with_both)) / ↑total) * 100 = 20 :=
by sorry

end NUMINAMATH_CALUDE_company_survey_l2773_277395


namespace NUMINAMATH_CALUDE_exists_valid_arrangement_l2773_277305

/-- Represents a connection between two circle indices -/
def Connection := Fin 5 × Fin 5

/-- Checks if two circles are connected -/
def is_connected (connections : List Connection) (i j : Fin 5) : Prop :=
  (i, j) ∈ connections ∨ (j, i) ∈ connections

/-- The theorem stating the existence of a valid arrangement -/
theorem exists_valid_arrangement : ∃ (numbers : Fin 5 → ℕ+) (connections : List Connection),
  (∀ i j : Fin 5, is_connected connections i j →
    (numbers i).val / (numbers j).val = 3 ∨ (numbers i).val / (numbers j).val = 9) ∧
  (∀ i j : Fin 5, ¬is_connected connections i j →
    (numbers i).val / (numbers j).val ≠ 3 ∧ (numbers i).val / (numbers j).val ≠ 9) :=
by sorry


end NUMINAMATH_CALUDE_exists_valid_arrangement_l2773_277305


namespace NUMINAMATH_CALUDE_larger_number_is_42_l2773_277368

theorem larger_number_is_42 (x y : ℝ) (sum_eq : x + y = 77) (ratio_eq : 5 * x = 6 * y) :
  max x y = 42 := by
sorry

end NUMINAMATH_CALUDE_larger_number_is_42_l2773_277368


namespace NUMINAMATH_CALUDE_marias_number_l2773_277367

theorem marias_number : ∃ x : ℚ, (((3 * x - 6) * 5) / 2 = 94) ∧ (x = 218 / 15) := by
  sorry

end NUMINAMATH_CALUDE_marias_number_l2773_277367


namespace NUMINAMATH_CALUDE_soccer_league_games_l2773_277301

/-- Calculates the total number of games in a soccer league --/
def total_games (n : ℕ) (k : ℕ) : ℕ :=
  -- Regular season games
  n * (n - 1) +
  -- Playoff games (single elimination format)
  (k - 1)

/-- Theorem stating the total number of games in the soccer league --/
theorem soccer_league_games :
  total_games 20 8 = 767 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l2773_277301


namespace NUMINAMATH_CALUDE_water_consumption_l2773_277398

theorem water_consumption (morning_amount : ℝ) (afternoon_multiplier : ℝ) : 
  morning_amount = 1.5 → 
  afternoon_multiplier = 3 → 
  morning_amount + (afternoon_multiplier * morning_amount) = 6 := by
  sorry

end NUMINAMATH_CALUDE_water_consumption_l2773_277398


namespace NUMINAMATH_CALUDE_product_mod_800_l2773_277370

theorem product_mod_800 : (2431 * 1587) % 800 = 397 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_800_l2773_277370


namespace NUMINAMATH_CALUDE_alpha_tan_beta_gt_beta_tan_alpha_l2773_277357

theorem alpha_tan_beta_gt_beta_tan_alpha (α β : Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) : 
  α * Real.tan β > β * Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_alpha_tan_beta_gt_beta_tan_alpha_l2773_277357


namespace NUMINAMATH_CALUDE_median_average_ratio_l2773_277306

theorem median_average_ratio (a b c : ℤ) : 
  a < b → b < c → a = 0 → (a + b + c) / 3 = 4 * b → c / b = 11 := by
  sorry

end NUMINAMATH_CALUDE_median_average_ratio_l2773_277306


namespace NUMINAMATH_CALUDE_power_of_product_l2773_277359

theorem power_of_product (a b : ℝ) : (a^2 * b)^3 = a^6 * b^3 := by sorry

end NUMINAMATH_CALUDE_power_of_product_l2773_277359


namespace NUMINAMATH_CALUDE_biology_enrollment_percentage_l2773_277310

theorem biology_enrollment_percentage (total_students : ℕ) (not_enrolled : ℕ) :
  total_students = 840 →
  not_enrolled = 546 →
  (((total_students - not_enrolled : ℚ) / total_students) * 100 : ℚ) = 35 := by
  sorry

end NUMINAMATH_CALUDE_biology_enrollment_percentage_l2773_277310


namespace NUMINAMATH_CALUDE_pythagorean_triple_identity_l2773_277302

theorem pythagorean_triple_identity (n : ℕ+) :
  (2 * n + 1) ^ 2 + (2 * n ^ 2 + 2 * n) ^ 2 = (2 * n ^ 2 + 2 * n + 1) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_identity_l2773_277302


namespace NUMINAMATH_CALUDE_common_tangents_count_l2773_277380

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 9
def circle2 (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 4

-- Define a function to count common tangent lines
noncomputable def count_common_tangents (c1 c2 : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem common_tangents_count :
  count_common_tangents circle1 circle2 = 3 := by sorry

end NUMINAMATH_CALUDE_common_tangents_count_l2773_277380


namespace NUMINAMATH_CALUDE_cubic_function_derivative_condition_l2773_277332

/-- Given a function f(x) = x^3 - mx + 3, if f'(1) = 0, then m = 3 -/
theorem cubic_function_derivative_condition (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 - m*x + 3
  (∀ x, (deriv f) x = 3*x^2 - m) → (deriv f) 1 = 0 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_derivative_condition_l2773_277332


namespace NUMINAMATH_CALUDE_a_investment_is_800_l2773_277356

/-- Represents the investment and profit scenario of three business partners -/
structure BusinessScenario where
  a_investment : ℝ
  b_investment : ℝ
  c_investment : ℝ
  investment_period : ℝ
  total_profit : ℝ
  c_profit_share : ℝ

/-- The business scenario with given conditions -/
def given_scenario : BusinessScenario :=
  { a_investment := 0,  -- Unknown, to be solved
    b_investment := 1000,
    c_investment := 1200,
    investment_period := 2,
    total_profit := 1000,
    c_profit_share := 400 }

/-- Theorem stating that a's investment in the given scenario is 800 -/
theorem a_investment_is_800 (scenario : BusinessScenario) 
  (h1 : scenario = given_scenario) :
  scenario.a_investment = 800 := by
  sorry


end NUMINAMATH_CALUDE_a_investment_is_800_l2773_277356


namespace NUMINAMATH_CALUDE_no_lower_grade_possible_l2773_277352

/-- Represents Lisa's quiz performance and goal --/
structure QuizPerformance where
  total_quizzes : ℕ
  goal_percentage : ℚ
  completed_quizzes : ℕ
  as_earned : ℕ

/-- Theorem stating that Lisa cannot earn a grade lower than A on any remaining quiz --/
theorem no_lower_grade_possible (perf : QuizPerformance) 
  (h1 : perf.total_quizzes = 60)
  (h2 : perf.goal_percentage = 85 / 100)
  (h3 : perf.completed_quizzes = 35)
  (h4 : perf.as_earned = 25) :
  (perf.total_quizzes - perf.completed_quizzes : ℚ) - 
  (↑⌈perf.goal_percentage * perf.total_quizzes⌉ - perf.as_earned) ≤ 0 := by
  sorry

#eval ⌈(85 : ℚ) / 100 * 60⌉ -- Expected output: 51

end NUMINAMATH_CALUDE_no_lower_grade_possible_l2773_277352


namespace NUMINAMATH_CALUDE_water_remaining_l2773_277364

theorem water_remaining (poured_out : ℚ) (h : poured_out = 45 / 100) :
  1 - poured_out = 55 / 100 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_l2773_277364


namespace NUMINAMATH_CALUDE_circle_area_through_points_l2773_277358

/-- The area of a circle with center P and passing through Q is 149π -/
theorem circle_area_through_points (P Q : ℝ × ℝ) : 
  P = (-2, 3) → Q = (8, -4) → 
  let r := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  π * r^2 = 149 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_through_points_l2773_277358


namespace NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_l2773_277322

theorem sum_of_multiples_of_6_and_9 (a b : ℤ) (ha : 6 ∣ a) (hb : 9 ∣ b) : 3 ∣ (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_of_6_and_9_l2773_277322


namespace NUMINAMATH_CALUDE_largest_inscribed_square_l2773_277396

theorem largest_inscribed_square (outer_square_side : ℝ) (triangle_side : ℝ) 
  (h1 : outer_square_side = 8)
  (h2 : triangle_side = outer_square_side)
  (h3 : 0 < outer_square_side) :
  let triangle_height : ℝ := triangle_side * (Real.sqrt 3) / 2
  let center_to_midpoint : ℝ := triangle_height / 2
  let inscribed_square_side : ℝ := 2 * center_to_midpoint
  inscribed_square_side = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribed_square_l2773_277396


namespace NUMINAMATH_CALUDE_sequential_search_element_count_l2773_277374

/-- Represents a sequential search in an unordered array -/
structure SequentialSearch where
  n : ℕ  -- number of elements in the array
  avg_comparisons : ℕ  -- average number of comparisons

/-- 
  Theorem: If the average number of comparisons in a sequential search 
  of an unordered array is 100, and the searched element is not in the array, 
  then the number of elements in the array is 200.
-/
theorem sequential_search_element_count 
  (search : SequentialSearch) 
  (h1 : search.avg_comparisons = 100) 
  (h2 : search.avg_comparisons = search.n / 2) : 
  search.n = 200 := by
  sorry

#check sequential_search_element_count

end NUMINAMATH_CALUDE_sequential_search_element_count_l2773_277374


namespace NUMINAMATH_CALUDE_max_min_sum_of_2x_minus_3y_l2773_277313

theorem max_min_sum_of_2x_minus_3y : 
  ∀ x y : ℝ, 3 ≤ x → x ≤ 5 → 4 ≤ y → y ≤ 6 → 
  (∃ (max min : ℝ), 
    (∀ z : ℝ, z = 2*x - 3*y → z ≤ max) ∧
    (∃ x' y' : ℝ, 3 ≤ x' ∧ x' ≤ 5 ∧ 4 ≤ y' ∧ y' ≤ 6 ∧ 2*x' - 3*y' = max) ∧
    (∀ z : ℝ, z = 2*x - 3*y → min ≤ z) ∧
    (∃ x' y' : ℝ, 3 ≤ x' ∧ x' ≤ 5 ∧ 4 ≤ y' ∧ y' ≤ 6 ∧ 2*x' - 3*y' = min) ∧
    max + min = -14) :=
by sorry

end NUMINAMATH_CALUDE_max_min_sum_of_2x_minus_3y_l2773_277313


namespace NUMINAMATH_CALUDE_school_girls_count_l2773_277308

theorem school_girls_count (total_students sample_size : ℕ) 
  (h_total : total_students = 1600)
  (h_sample : sample_size = 200)
  (h_stratified_sample : ∃ (girls_sampled boys_sampled : ℕ), 
    girls_sampled + boys_sampled = sample_size ∧ 
    girls_sampled + 20 = boys_sampled) :
  ∃ (school_girls : ℕ), 
    school_girls * sample_size = 720 * total_students ∧
    school_girls ≤ total_students :=
by sorry

end NUMINAMATH_CALUDE_school_girls_count_l2773_277308


namespace NUMINAMATH_CALUDE_standard_deviation_is_2_l2773_277331

def data : List ℝ := [51, 54, 55, 57, 53]

theorem standard_deviation_is_2 :
  let mean := (data.sum) / (data.length : ℝ)
  let variance := (data.map (λ x => (x - mean) ^ 2)).sum / (data.length : ℝ)
  Real.sqrt variance = 2 := by sorry

end NUMINAMATH_CALUDE_standard_deviation_is_2_l2773_277331


namespace NUMINAMATH_CALUDE_girls_joined_team_l2773_277300

/-- Proves that 7 girls joined the track team given the initial and final conditions --/
theorem girls_joined_team (initial_girls : ℕ) (initial_boys : ℕ) (boys_quit : ℕ) (final_total : ℕ) : 
  initial_girls = 18 → initial_boys = 15 → boys_quit = 4 → final_total = 36 →
  final_total - (initial_girls + (initial_boys - boys_quit)) = 7 := by
sorry

end NUMINAMATH_CALUDE_girls_joined_team_l2773_277300


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2773_277316

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 8 = 0 ∧ x₂^2 + m*x₂ - 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2773_277316


namespace NUMINAMATH_CALUDE_fundraiser_goal_is_750_l2773_277307

/-- Represents the fundraiser goal calculation --/
def fundraiser_goal (bronze_families : ℕ) (silver_families : ℕ) (gold_families : ℕ) 
  (bronze_donation : ℕ) (silver_donation : ℕ) (gold_donation : ℕ) 
  (final_day_goal : ℕ) : ℕ :=
  bronze_families * bronze_donation + 
  silver_families * silver_donation + 
  gold_families * gold_donation + 
  final_day_goal

/-- Theorem stating that the fundraiser goal is $750 --/
theorem fundraiser_goal_is_750 : 
  fundraiser_goal 10 7 1 25 50 100 50 = 750 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_goal_is_750_l2773_277307


namespace NUMINAMATH_CALUDE_bobby_shoe_cost_l2773_277344

theorem bobby_shoe_cost (mold_cost labor_rate hours discount_rate : ℝ) : 
  mold_cost = 250 →
  labor_rate = 75 →
  hours = 8 →
  discount_rate = 0.8 →
  mold_cost + (labor_rate * hours * discount_rate) = 730 := by
sorry

end NUMINAMATH_CALUDE_bobby_shoe_cost_l2773_277344


namespace NUMINAMATH_CALUDE_complex_division_result_l2773_277388

theorem complex_division_result : (3 - Complex.I) / Complex.I = -1 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_result_l2773_277388


namespace NUMINAMATH_CALUDE_jordans_income_l2773_277353

/-- Represents the state income tax calculation and Jordan's specific case -/
theorem jordans_income (q : ℝ) : 
  ∃ (I : ℝ),
    I > 35000 ∧
    0.01 * q * 35000 + 0.01 * (q + 3) * (I - 35000) = (0.01 * q + 0.004) * I ∧
    I = 40000 := by
  sorry

end NUMINAMATH_CALUDE_jordans_income_l2773_277353


namespace NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l2773_277378

/-- A function f is increasing on an interval [a, b) if for any x, y in [a, b) with x < y, f(x) < f(y) -/
def IsIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y < b → f x < f y

/-- The quadratic function f(x) = ax^2 + 2x - 2a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 2 * a

theorem range_of_a_for_increasing_f :
  {a : ℝ | IsIncreasing (f a) (-1) 2} = Set.Icc (-1/2) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_increasing_f_l2773_277378


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l2773_277348

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 6 > 0}
def B : Set ℝ := {x | x - 1 > 0}

-- Define the complement of A in ℝ
def C_R_A : Set ℝ := (Set.univ : Set ℝ) \ A

-- State the theorem
theorem complement_A_intersect_B : C_R_A ∩ B = Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l2773_277348


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2773_277391

theorem expression_simplification_and_evaluation :
  let x : ℝ := 3
  let expr := (1 / (x - 2) - 1 / (x + 1)) / (3 / (x^2 - 1))
  expr = 2 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2773_277391


namespace NUMINAMATH_CALUDE_square_equation_solution_l2773_277303

theorem square_equation_solution (x : ℝ) : (x - 1)^2 = 4 → x = 3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l2773_277303


namespace NUMINAMATH_CALUDE_expected_condition_sufferers_l2773_277341

theorem expected_condition_sufferers (total_sample : ℕ) (condition_rate : ℚ) : 
  total_sample = 450 → condition_rate = 1/3 → 
  (condition_rate * total_sample : ℚ) = 150 := by
sorry

end NUMINAMATH_CALUDE_expected_condition_sufferers_l2773_277341


namespace NUMINAMATH_CALUDE_amount_for_c_l2773_277371

def total_amount : ℕ := 2000
def ratio_b : ℕ := 4
def ratio_c : ℕ := 16

theorem amount_for_c (total : ℕ) (rb : ℕ) (rc : ℕ) (h1 : total = total_amount) (h2 : rb = ratio_b) (h3 : rc = ratio_c) :
  (rc * total) / (rb + rc) = 1600 := by
  sorry

end NUMINAMATH_CALUDE_amount_for_c_l2773_277371


namespace NUMINAMATH_CALUDE_stratified_sampling_is_appropriate_l2773_277311

/-- Represents a stratum in a population --/
structure Stratum where
  size : ℕ
  characteristic : ℝ

/-- Represents a population with three strata --/
structure Population where
  strata : Fin 3 → Stratum
  total_size : ℕ
  avg_characteristic : ℝ

/-- Represents a sampling method --/
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling
  | SystematicSampling

/-- Determines if a sampling method is appropriate for a given population and sample size --/
def is_appropriate_sampling_method (pop : Population) (sample_size : ℕ) (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.StratifiedSampling => true
  | _ => false

/-- Theorem stating that stratified sampling is the appropriate method for the given scenario --/
theorem stratified_sampling_is_appropriate (pop : Population) (sample_size : ℕ) :
  is_appropriate_sampling_method pop sample_size SamplingMethod.StratifiedSampling :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_is_appropriate_l2773_277311


namespace NUMINAMATH_CALUDE_abs_S_value_l2773_277360

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- The complex number S -/
def S : ℂ := (1 + 2*i)^12 - (1 - i)^12

/-- Theorem stating the absolute value of S -/
theorem abs_S_value : Complex.abs S = 15689 := by sorry

end NUMINAMATH_CALUDE_abs_S_value_l2773_277360


namespace NUMINAMATH_CALUDE_sum_of_three_element_subset_sums_l2773_277376

def A : Finset ℕ := Finset.range 10

def three_element_subsets (s : Finset ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ subset => subset.card = 3)

def subset_sum (subset : Finset ℕ) : ℕ :=
  subset.sum id

theorem sum_of_three_element_subset_sums : 
  (three_element_subsets A).sum subset_sum = 1980 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_element_subset_sums_l2773_277376


namespace NUMINAMATH_CALUDE_roots_properties_l2773_277394

theorem roots_properties (r s t : ℝ) : 
  (∀ x : ℝ, x * (x - 2) * (3 * x - 7) = 2 ↔ x = r ∨ x = s ∨ x = t) →
  (r > 0 ∧ s > 0 ∧ t > 0) ∧
  (Real.arctan r + Real.arctan s + Real.arctan t = 3 * π / 4) := by
sorry

end NUMINAMATH_CALUDE_roots_properties_l2773_277394


namespace NUMINAMATH_CALUDE_unique_arrangement_l2773_277346

def is_valid_arrangement (A B C D E F : ℕ) : Prop :=
  A ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  B ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  C ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  D ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  E ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  F ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
  D ≠ E ∧ D ≠ F ∧
  E ≠ F ∧
  A + D + E = 15 ∧
  7 + C + E = 15 ∧
  9 + C + A = 15 ∧
  A + 8 + F = 15 ∧
  7 + D + F = 15 ∧
  9 + D + B = 15

theorem unique_arrangement :
  ∀ A B C D E F : ℕ,
  is_valid_arrangement A B C D E F →
  A = 4 ∧ B = 1 ∧ C = 2 ∧ D = 5 ∧ E = 6 ∧ F = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_arrangement_l2773_277346


namespace NUMINAMATH_CALUDE_non_working_video_games_l2773_277333

theorem non_working_video_games (total : ℕ) (price : ℕ) (earnings : ℕ) : 
  total = 10 → price = 6 → earnings = 12 → total - (earnings / price) = 8 := by
  sorry

end NUMINAMATH_CALUDE_non_working_video_games_l2773_277333


namespace NUMINAMATH_CALUDE_first_group_factory_count_l2773_277317

theorem first_group_factory_count (total : ℕ) (second_group : ℕ) (remaining : ℕ) 
  (h1 : total = 169) 
  (h2 : second_group = 52) 
  (h3 : remaining = 48) : 
  total - second_group - remaining = 69 := by
  sorry

end NUMINAMATH_CALUDE_first_group_factory_count_l2773_277317


namespace NUMINAMATH_CALUDE_opposite_sides_line_constant_range_l2773_277320

/-- Given two points on opposite sides of a line, prove the range of the constant term -/
theorem opposite_sides_line_constant_range :
  ∀ (a : ℝ),
  (((3 * 2 - 2 * 1 + a) * (3 * (-2) - 2 * 3 + a) < 0) ↔ (-4 < a ∧ a < 12)) := by
  sorry

end NUMINAMATH_CALUDE_opposite_sides_line_constant_range_l2773_277320


namespace NUMINAMATH_CALUDE_rosas_initial_flowers_l2773_277339

/-- The problem of finding Rosa's initial number of flowers -/
theorem rosas_initial_flowers :
  ∀ (initial_flowers additional_flowers total_flowers : ℕ),
    additional_flowers = 23 →
    total_flowers = 90 →
    total_flowers = initial_flowers + additional_flowers →
    initial_flowers = 67 := by
  sorry

end NUMINAMATH_CALUDE_rosas_initial_flowers_l2773_277339


namespace NUMINAMATH_CALUDE_existence_of_non_coprime_pair_l2773_277336

theorem existence_of_non_coprime_pair :
  ∃ m : ℤ, (Nat.gcd (100 + 101 * m).natAbs (101 - 100 * m).natAbs) ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_non_coprime_pair_l2773_277336


namespace NUMINAMATH_CALUDE_power_multiplication_l2773_277319

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2773_277319


namespace NUMINAMATH_CALUDE_intersection_distance_product_l2773_277393

/-- Given an ellipse and a hyperbola sharing the same foci, the product of distances
    from their intersection point to the foci is equal to the difference of their
    respective parameters. -/
theorem intersection_distance_product (a b m n : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  a > b ∧ b > 0 ∧ m > 0 ∧ n > 0 →
  (P.1^2 / a + P.2^2 / b = 1) →
  (P.1^2 / m - P.2^2 / n = 1) →
  (∀ Q : ℝ × ℝ, Q.1^2 / a + Q.2^2 / b = 1 → dist Q F₁ + dist Q F₂ = 2 * Real.sqrt a) →
  (∀ R : ℝ × ℝ, R.1^2 / m - R.2^2 / n = 1 → |dist R F₁ - dist R F₂| = 2 * Real.sqrt m) →
  dist P F₁ * dist P F₂ = a - m :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_product_l2773_277393


namespace NUMINAMATH_CALUDE_dinosaur_egg_theft_l2773_277345

theorem dinosaur_egg_theft (total_eggs : ℕ) (claimed_max : ℕ) : 
  total_eggs = 20 → 
  claimed_max = 7 → 
  ¬(∃ (a b : ℕ), 
    a + b + claimed_max = total_eggs ∧ 
    a ≠ b ∧ 
    a ≠ claimed_max ∧ 
    b ≠ claimed_max ∧
    a < claimed_max ∧ 
    b < claimed_max) := by
  sorry

end NUMINAMATH_CALUDE_dinosaur_egg_theft_l2773_277345


namespace NUMINAMATH_CALUDE_second_day_percentage_l2773_277334

def puzzle_pieces : ℕ := 1000
def first_day_percentage : ℚ := 10 / 100
def third_day_percentage : ℚ := 30 / 100
def pieces_left_after_third_day : ℕ := 504

theorem second_day_percentage :
  ∃ (p : ℚ),
    p > 0 ∧
    p < 1 ∧
    (puzzle_pieces * (1 - first_day_percentage) * (1 - p) * (1 - third_day_percentage) : ℚ) =
      pieces_left_after_third_day ∧
    p = 20 / 100 := by
  sorry

end NUMINAMATH_CALUDE_second_day_percentage_l2773_277334


namespace NUMINAMATH_CALUDE_cube_face_sum_l2773_277379

/-- Represents the numbers on the faces of a cube -/
structure CubeFaces where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+

/-- The sum of vertex products for a given set of cube faces -/
def vertexProductSum (faces : CubeFaces) : ℕ :=
  faces.a * faces.b * faces.c +
  faces.a * faces.e * faces.c +
  faces.a * faces.b * faces.f +
  faces.a * faces.e * faces.f +
  faces.d * faces.b * faces.c +
  faces.d * faces.e * faces.c +
  faces.d * faces.b * faces.f +
  faces.d * faces.e * faces.f

/-- The sum of all face values -/
def faceSum (faces : CubeFaces) : ℕ :=
  faces.a + faces.b + faces.c + faces.d + faces.e + faces.f

theorem cube_face_sum (faces : CubeFaces) :
  vertexProductSum faces = 1008 → faceSum faces = 173 := by
  sorry


end NUMINAMATH_CALUDE_cube_face_sum_l2773_277379


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2773_277304

theorem arithmetic_sequence_fifth_term 
  (x y : ℝ) 
  (seq : ℕ → ℝ)
  (h1 : seq 0 = x + 2*y)
  (h2 : seq 1 = x - 2*y)
  (h3 : seq 2 = x^2 * y)
  (h4 : seq 3 = x / (2*y))
  (h_arith : ∀ n, seq (n+1) - seq n = seq 1 - seq 0)
  (hy : y = 1)
  (hx : x = 20) :
  seq 4 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2773_277304


namespace NUMINAMATH_CALUDE_sum_of_squares_l2773_277375

theorem sum_of_squares (x y z : ℝ) 
  (eq1 : x^2 + 3*y = 10)
  (eq2 : y^2 + 5*z = -10)
  (eq3 : z^2 + 7*x = -21) : 
  x^2 + y^2 + z^2 = 83/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2773_277375


namespace NUMINAMATH_CALUDE_share_difference_l2773_277318

/-- Represents the share ratio for each person --/
structure ShareRatio :=
  (faruk : ℕ)
  (vasim : ℕ)
  (ranjith : ℕ)
  (kavita : ℕ)
  (neel : ℕ)

/-- Represents the distribution problem --/
structure DistributionProblem :=
  (ratio : ShareRatio)
  (vasim_share : ℕ)
  (x : ℕ+)
  (y : ℕ+)

def total_ratio (r : ShareRatio) : ℕ :=
  r.faruk + r.vasim + r.ranjith + r.kavita + r.neel

def total_amount (p : DistributionProblem) : ℕ :=
  p.vasim_share * (p.x + p.y)

theorem share_difference (p : DistributionProblem) 
  (h1 : p.ratio = ⟨3, 5, 7, 9, 11⟩)
  (h2 : p.vasim_share = 1500)
  (h3 : total_amount p = total_ratio p.ratio * (p.vasim_share / p.ratio.vasim)) :
  p.ratio.ranjith * (p.vasim_share / p.ratio.vasim) - 
  p.ratio.faruk * (p.vasim_share / p.ratio.vasim) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_share_difference_l2773_277318


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_231_4620_l2773_277363

theorem gcd_lcm_sum_231_4620 : Nat.gcd 231 4620 + Nat.lcm 231 4620 = 4851 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_231_4620_l2773_277363
