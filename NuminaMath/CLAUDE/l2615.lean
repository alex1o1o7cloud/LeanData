import Mathlib

namespace NUMINAMATH_CALUDE_spring_decrease_percentage_l2615_261544

theorem spring_decrease_percentage (initial_increase : ℝ) (total_change : ℝ) : 
  initial_increase = 0.05 →
  total_change = -0.1495 →
  ∃ spring_decrease : ℝ, 
    (1 + initial_increase) * (1 - spring_decrease) = 1 + total_change ∧
    spring_decrease = 0.19 :=
by sorry

end NUMINAMATH_CALUDE_spring_decrease_percentage_l2615_261544


namespace NUMINAMATH_CALUDE_booknote_unique_letters_l2615_261583

def word : String := "booknote"

def letter_set : Finset Char := word.toList.toFinset

theorem booknote_unique_letters : Finset.card letter_set = 6 := by
  sorry

end NUMINAMATH_CALUDE_booknote_unique_letters_l2615_261583


namespace NUMINAMATH_CALUDE_geometryville_schools_l2615_261556

theorem geometryville_schools (n : ℕ) : 
  n > 0 → 
  let total_students := 4 * n
  let andreas_rank := (12 * n + 1) / 4
  andreas_rank > total_students / 2 →
  andreas_rank ≤ 3 * total_students / 4 →
  (∃ (teammate_rank : ℕ), 
    teammate_rank ≤ total_students / 2 ∧ 
    teammate_rank < andreas_rank) →
  (∃ (bottom_teammates : Fin 2 → ℕ), 
    ∀ i, bottom_teammates i > total_students / 2 ∧ 
         bottom_teammates i < andreas_rank) →
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_geometryville_schools_l2615_261556


namespace NUMINAMATH_CALUDE_experiment_sequences_l2615_261516

/-- The number of procedures in the experiment -/
def num_procedures : ℕ := 5

/-- Represents the possible positions for procedure A -/
inductive ProcedureAPosition
| First
| Last

/-- Represents a pair of adjacent procedures (C and D) -/
structure AdjacentPair where
  first : Fin num_procedures
  second : Fin num_procedures
  adjacent : first.val + 1 = second.val

/-- The total number of possible sequences in the experiment -/
def num_sequences : ℕ := 24

/-- Theorem stating the number of possible sequences in the experiment -/
theorem experiment_sequences :
  ∀ (a_pos : ProcedureAPosition) (cd_pair : AdjacentPair),
  num_sequences = 24 :=
sorry

end NUMINAMATH_CALUDE_experiment_sequences_l2615_261516


namespace NUMINAMATH_CALUDE_sector_max_area_l2615_261561

/-- Given a sector with circumference 20cm, its area is maximized when the radius is 5cm -/
theorem sector_max_area (R : ℝ) : 
  let circumference := 20
  let arc_length := circumference - 2 * R
  let area := (1 / 2) * arc_length * R
  (∀ r : ℝ, area ≤ ((1 / 2) * (circumference - 2 * r) * r)) → R = 5 := by
sorry

end NUMINAMATH_CALUDE_sector_max_area_l2615_261561


namespace NUMINAMATH_CALUDE_f_negative_a_eq_zero_l2615_261551

noncomputable def f (x : ℝ) : ℝ := x^3 * (Real.exp x + Real.exp (-x)) + 2

theorem f_negative_a_eq_zero (a : ℝ) (h : f a = 4) : f (-a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_a_eq_zero_l2615_261551


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l2615_261578

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by its slope and y-intercept --/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Check if a line is tangent to a circle --/
def isTangent (l : Line) (c : Circle) : Prop :=
  let (x, y) := c.center
  (l.slope * x - y + l.yIntercept)^2 = c.radius^2 * (l.slope^2 + 1)

theorem tangent_line_y_intercept :
  ∃ (l : Line),
    isTangent l { center := (2, 0), radius := 2 } ∧
    isTangent l { center := (5, 0), radius := 1 } ∧
    l.yIntercept = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l2615_261578


namespace NUMINAMATH_CALUDE_student_arrangements_l2615_261540

def num_male_students : ℕ := 4
def num_female_students : ℕ := 3
def total_students : ℕ := num_male_students + num_female_students

def arrangements_female_together : ℕ := sorry

def arrangements_no_adjacent_females : ℕ := sorry

def arrangements_ordered_females : ℕ := sorry

theorem student_arrangements :
  (arrangements_female_together = 720) ∧
  (arrangements_no_adjacent_females = 1440) ∧
  (arrangements_ordered_females = 840) := by sorry

end NUMINAMATH_CALUDE_student_arrangements_l2615_261540


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l2615_261577

theorem hemisphere_surface_area (base_area : Real) (h : base_area = 225 * Real.pi) :
  let radius : Real := (base_area / Real.pi).sqrt
  let curved_surface_area : Real := 2 * Real.pi * radius^2
  let total_surface_area : Real := curved_surface_area + base_area
  total_surface_area = 675 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l2615_261577


namespace NUMINAMATH_CALUDE_only_one_claim_impossible_l2615_261581

-- Define the possible ring scores
def RingScores : List Nat := [1, 3, 5, 7, 9]

-- Define a structure for each person's claim
structure Claim where
  shots : Nat
  hits : Nat
  total_score : Nat

-- Define the claims
def claim_A : Claim := { shots := 5, hits := 5, total_score := 35 }
def claim_B : Claim := { shots := 6, hits := 6, total_score := 36 }
def claim_C : Claim := { shots := 3, hits := 3, total_score := 24 }
def claim_D : Claim := { shots := 4, hits := 3, total_score := 21 }

-- Function to check if a claim is possible
def is_claim_possible (c : Claim) : Prop :=
  ∃ (scores : List Nat),
    scores.length = c.hits ∧
    scores.all (· ∈ RingScores) ∧
    scores.sum = c.total_score

-- Theorem stating that only one claim is impossible
theorem only_one_claim_impossible :
  is_claim_possible claim_A ∧
  is_claim_possible claim_B ∧
  ¬is_claim_possible claim_C ∧
  is_claim_possible claim_D :=
sorry

end NUMINAMATH_CALUDE_only_one_claim_impossible_l2615_261581


namespace NUMINAMATH_CALUDE_expression_value_l2615_261584

theorem expression_value : (1/3 * 9 * 1/27 * 81 * 1/243 * 729)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2615_261584


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2615_261565

theorem sqrt_equation_solution :
  ∃ x : ℝ, x ≥ 0 ∧ Real.sqrt x + Real.sqrt (x + 1) - Real.sqrt (x + 2) = 0 ∧ x = -1 + (2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2615_261565


namespace NUMINAMATH_CALUDE_divisibility_implication_l2615_261524

theorem divisibility_implication (a b m n : ℕ) 
  (h1 : a > 1) 
  (h2 : Nat.gcd a b = 1) 
  (h3 : (a^n + b^n) ∣ (a^m + b^m)) : 
  n ∣ m := by sorry

end NUMINAMATH_CALUDE_divisibility_implication_l2615_261524


namespace NUMINAMATH_CALUDE_sum_inequality_l2615_261500

theorem sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_sum : 1 / (a^3 + 1) + 1 / (b^3 + 1) + 1 / (c^3 + 1) + 1 / (d^3 + 1) = 2) :
  (1 - a) / (a^2 - a + 1) + (1 - b) / (b^2 - b + 1) + 
  (1 - c) / (c^2 - c + 1) + (1 - d) / (d^2 - d + 1) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l2615_261500


namespace NUMINAMATH_CALUDE_absolute_value_expression_l2615_261574

theorem absolute_value_expression : |-2| * (|-25| - |5|) = 40 := by sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l2615_261574


namespace NUMINAMATH_CALUDE_planter_cost_theorem_l2615_261520

/-- Represents the cost and quantity of a type of plant in a planter --/
structure PlantInfo where
  quantity : ℕ
  price : ℚ

/-- Calculates the total cost for a rectangle-shaped pool's corner planters --/
def total_cost (palm_fern : PlantInfo) (creeping_jenny : PlantInfo) (geranium : PlantInfo) : ℚ :=
  let cost_per_pot := palm_fern.quantity * palm_fern.price + 
                      creeping_jenny.quantity * creeping_jenny.price + 
                      geranium.quantity * geranium.price
  4 * cost_per_pot

/-- Theorem stating the total cost for the planters --/
theorem planter_cost_theorem (palm_fern : PlantInfo) (creeping_jenny : PlantInfo) (geranium : PlantInfo)
  (h1 : palm_fern.quantity = 1)
  (h2 : palm_fern.price = 15)
  (h3 : creeping_jenny.quantity = 4)
  (h4 : creeping_jenny.price = 4)
  (h5 : geranium.quantity = 4)
  (h6 : geranium.price = 7/2) :
  total_cost palm_fern creeping_jenny geranium = 180 := by
  sorry

end NUMINAMATH_CALUDE_planter_cost_theorem_l2615_261520


namespace NUMINAMATH_CALUDE_only_four_and_six_have_three_solutions_l2615_261567

def X : Finset ℕ := {1, 2, 5, 7, 11, 13, 16, 17}

def hasThreedifferentsolutions (k : ℕ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℕ), 
    x₁ ∈ X ∧ y₁ ∈ X ∧ x₂ ∈ X ∧ y₂ ∈ X ∧ x₃ ∈ X ∧ y₃ ∈ X ∧
    x₁ - y₁ = k ∧ x₂ - y₂ = k ∧ x₃ - y₃ = k ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₃, y₃)

theorem only_four_and_six_have_three_solutions :
  ∀ k : ℕ, k > 0 → (hasThreedifferentsolutions k ↔ k = 4 ∨ k = 6) := by sorry

end NUMINAMATH_CALUDE_only_four_and_six_have_three_solutions_l2615_261567


namespace NUMINAMATH_CALUDE_john_final_amount_l2615_261593

def calculate_final_amount (initial_amount : ℚ) (game_cost : ℚ) (candy_cost : ℚ) 
  (soda_cost : ℚ) (magazine_cost : ℚ) (coupon_value : ℚ) (discount_rate : ℚ) 
  (allowance : ℚ) : ℚ :=
  let discounted_soda_cost := soda_cost * (1 - discount_rate)
  let magazine_paid := magazine_cost - coupon_value
  let total_expenses := game_cost + candy_cost + discounted_soda_cost + magazine_paid
  let remaining_after_expenses := initial_amount - total_expenses
  remaining_after_expenses + allowance

theorem john_final_amount :
  calculate_final_amount 5 2 1 1.5 3 0.5 0.1 26 = 24.15 := by
  sorry

end NUMINAMATH_CALUDE_john_final_amount_l2615_261593


namespace NUMINAMATH_CALUDE_share_division_l2615_261541

/-- Given a total sum of 427 to be divided among three people A, B, and C,
    where 3 times A's share equals 4 times B's share equals 7 times C's share,
    C's share is 84. -/
theorem share_division (a b c : ℚ) : 
  a + b + c = 427 →
  3 * a = 4 * b →
  4 * b = 7 * c →
  c = 84 := by
  sorry

end NUMINAMATH_CALUDE_share_division_l2615_261541


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2615_261552

theorem functional_equation_solution 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) : 
  ∃! f : ℝ → ℝ, 
    (∀ x, x > 0 → f x > 0) ∧ 
    (∀ x, x > 0 → f (f x) + a * f x = b * (a + b) * x) ∧
    (∀ x, x > 0 → f x = b * x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2615_261552


namespace NUMINAMATH_CALUDE_min_value_a_l2615_261590

theorem min_value_a (a m n : ℕ) (h1 : a ≠ 0) (h2 : (2 : ℚ) / 10 * a = m ^ 2) (h3 : (5 : ℚ) / 10 * a = n ^ 3) :
  ∀ b : ℕ, b ≠ 0 ∧ (∃ p q : ℕ, (2 : ℚ) / 10 * b = p ^ 2 ∧ (5 : ℚ) / 10 * b = q ^ 3) → a ≤ b → a = 2000 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l2615_261590


namespace NUMINAMATH_CALUDE_box_length_proof_l2615_261596

/-- Proves that a rectangular box with given dimensions and fill rate has a specific length -/
theorem box_length_proof (fill_rate : ℝ) (width depth time : ℝ) (h1 : fill_rate = 4)
    (h2 : width = 6) (h3 : depth = 2) (h4 : time = 21) :
  (fill_rate * time) / (width * depth) = 7 := by
  sorry

end NUMINAMATH_CALUDE_box_length_proof_l2615_261596


namespace NUMINAMATH_CALUDE_twenty_is_forty_percent_of_fifty_l2615_261599

theorem twenty_is_forty_percent_of_fifty :
  ∀ x : ℝ, (20 : ℝ) / x = (40 : ℝ) / 100 → x = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_twenty_is_forty_percent_of_fifty_l2615_261599


namespace NUMINAMATH_CALUDE_min_sum_arc_lengths_l2615_261595

/-- A set of points on a circle consisting of n arcs -/
structure CircleSet (n : ℕ) where
  arcs : Fin n → Set ℝ
  sum_lengths : ℝ

/-- Rotation of a set of points on a circle -/
def rotate (α : ℝ) (F : Set ℝ) : Set ℝ := sorry

/-- Property that for any rotation, the rotated set intersects with the original set -/
def intersects_all_rotations (F : Set ℝ) : Prop :=
  ∀ α : ℝ, (rotate α F ∩ F).Nonempty

/-- Theorem stating the minimum sum of arc lengths -/
theorem min_sum_arc_lengths (n : ℕ) (F : CircleSet n) 
  (h : intersects_all_rotations (⋃ i, F.arcs i)) :
  F.sum_lengths ≥ 180 / n := sorry

end NUMINAMATH_CALUDE_min_sum_arc_lengths_l2615_261595


namespace NUMINAMATH_CALUDE_map_distance_theorem_l2615_261509

/-- Represents the scale of a map as a ratio -/
def MapScale : ℚ := 1 / 250000

/-- Converts kilometers to centimeters -/
def kmToCm (km : ℚ) : ℚ := km * 100000

/-- Calculates the distance on a map given the actual distance and map scale -/
def mapDistance (actualDistance : ℚ) (scale : ℚ) : ℚ :=
  actualDistance * scale

theorem map_distance_theorem (actualDistanceKm : ℚ) 
  (h : actualDistanceKm = 5) : 
  mapDistance (kmToCm actualDistanceKm) MapScale = 2 := by
  sorry

#check map_distance_theorem

end NUMINAMATH_CALUDE_map_distance_theorem_l2615_261509


namespace NUMINAMATH_CALUDE_square_fraction_count_l2615_261564

theorem square_fraction_count : 
  ∃! (S : Finset ℤ), 
    (∀ n ∈ S, 0 ≤ n ∧ n ≤ 23 ∧ ∃ k : ℤ, (n : ℚ) / (24 - n) = k^2) ∧ 
    Finset.card S = 2 :=
by sorry

end NUMINAMATH_CALUDE_square_fraction_count_l2615_261564


namespace NUMINAMATH_CALUDE_triangle_inequality_bound_l2615_261560

theorem triangle_inequality_bound (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : b = 2 * a) :
  (a^2 + b^2) / c^2 > 5/9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_bound_l2615_261560


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2615_261588

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 3 + 56 / 99) ∧ (x = 353 / 99) :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l2615_261588


namespace NUMINAMATH_CALUDE_power_equation_non_negative_l2615_261514

theorem power_equation_non_negative (a b c d : ℤ) 
  (h : (2 : ℝ)^a + (2 : ℝ)^b = (5 : ℝ)^c + (5 : ℝ)^d) : 
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d := by
  sorry

end NUMINAMATH_CALUDE_power_equation_non_negative_l2615_261514


namespace NUMINAMATH_CALUDE_annual_savings_l2615_261506

/-- Represents the parking garage rental rates and conditions -/
structure ParkingGarage where
  regular_peak_weekly : ℕ
  regular_nonpeak_weekly : ℕ
  regular_peak_monthly : ℕ
  regular_nonpeak_monthly : ℕ
  large_peak_weekly : ℕ
  large_nonpeak_weekly : ℕ
  large_peak_monthly : ℕ
  large_nonpeak_monthly : ℕ
  holiday_surcharge : ℕ
  nonpeak_weeks : ℕ
  peak_holiday_weeks : ℕ
  total_weeks : ℕ

/-- Calculates the annual cost of renting a large space weekly -/
def weekly_cost (pg : ParkingGarage) : ℕ :=
  pg.large_nonpeak_weekly * pg.nonpeak_weeks +
  pg.large_peak_weekly * (pg.total_weeks - pg.nonpeak_weeks - pg.peak_holiday_weeks) +
  (pg.large_peak_weekly + pg.holiday_surcharge) * pg.peak_holiday_weeks

/-- Calculates the annual cost of renting a large space monthly -/
def monthly_cost (pg : ParkingGarage) : ℕ :=
  pg.large_nonpeak_monthly * (pg.nonpeak_weeks / 4) +
  pg.large_peak_monthly * ((pg.total_weeks - pg.nonpeak_weeks) / 4)

/-- Theorem: The annual savings from renting monthly instead of weekly is $124 -/
theorem annual_savings (pg : ParkingGarage) : weekly_cost pg - monthly_cost pg = 124 :=
  by
    have h1 : pg.regular_peak_weekly = 10 := by sorry
    have h2 : pg.regular_nonpeak_weekly = 8 := by sorry
    have h3 : pg.regular_peak_monthly = 40 := by sorry
    have h4 : pg.regular_nonpeak_monthly = 35 := by sorry
    have h5 : pg.large_peak_weekly = 12 := by sorry
    have h6 : pg.large_nonpeak_weekly = 10 := by sorry
    have h7 : pg.large_peak_monthly = 48 := by sorry
    have h8 : pg.large_nonpeak_monthly = 42 := by sorry
    have h9 : pg.holiday_surcharge = 2 := by sorry
    have h10 : pg.nonpeak_weeks = 16 := by sorry
    have h11 : pg.peak_holiday_weeks = 6 := by sorry
    have h12 : pg.total_weeks = 52 := by sorry
    sorry

end NUMINAMATH_CALUDE_annual_savings_l2615_261506


namespace NUMINAMATH_CALUDE_dagger_example_l2615_261594

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * (2 * q / n)

-- Theorem statement
theorem dagger_example : dagger (5/9) (7/6) = 140/3 := by
  sorry

end NUMINAMATH_CALUDE_dagger_example_l2615_261594


namespace NUMINAMATH_CALUDE_positive_number_relationship_l2615_261535

theorem positive_number_relationship (n : ℕ) (a b : ℝ) 
  (h_n : n ≥ 2) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0) 
  (h_a_eq : a^n = a + 1) 
  (h_b_eq : b^(2*n) = b + 3*a) : 
  a > b ∧ b > 1 := by
sorry

end NUMINAMATH_CALUDE_positive_number_relationship_l2615_261535


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2615_261575

theorem regular_polygon_sides (exterior_angle : ℝ) (h : exterior_angle = 45) :
  (360 / exterior_angle : ℝ) = 8 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2615_261575


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2615_261597

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a) 
  (h_a1 : a 1 = -1) 
  (h_sum : a 2 + a 3 = -2) :
  ∃ q : ℝ, (q = -2 ∨ q = 1) ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2615_261597


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2615_261589

theorem diophantine_equation_solutions :
  ∀ x y z w : ℕ,
  2^x * 3^y - 5^z * 7^w = 1 ↔
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ w = 0) ∨
  (x = 3 ∧ y = 0 ∧ z = 0 ∧ w = 1) ∨
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ w = 0) ∨
  (x = 2 ∧ y = 2 ∧ z = 1 ∧ w = 1) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2615_261589


namespace NUMINAMATH_CALUDE_bike_shop_profit_l2615_261522

/-- The profit calculation for Jim's bike shop -/
theorem bike_shop_profit (x : ℝ) 
  (h1 : x > 0) -- Charge for fixing bike tires is positive
  (h2 : 300 * x + 600 + 2000 - (300 * 5 + 100 + 4000) = 3000) -- Profit equation
  : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_bike_shop_profit_l2615_261522


namespace NUMINAMATH_CALUDE_equations_solution_set_l2615_261533

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(0,0,0,0), (2,2,2,2), (1,5,2,3), (5,1,2,3), (1,5,3,2), (5,1,3,2),
   (2,3,1,5), (2,3,5,1), (3,2,1,5), (3,2,5,1)}

def satisfies_equations (x y z t : ℕ) : Prop :=
  x + y = z + t ∧ z + t = x * y

theorem equations_solution_set :
  ∀ x y z t : ℕ, satisfies_equations x y z t ↔ (x, y, z, t) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_equations_solution_set_l2615_261533


namespace NUMINAMATH_CALUDE_inverse_proposition_l2615_261586

theorem inverse_proposition :
  (∀ x a b : ℝ, x ≥ a^2 + b^2 → x ≥ 2*a*b) →
  (∀ x a b : ℝ, x ≥ 2*a*b → x ≥ a^2 + b^2) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proposition_l2615_261586


namespace NUMINAMATH_CALUDE_equivalent_representations_l2615_261566

theorem equivalent_representations : ∀ (a b c d e : ℚ),
  (a = 15 ∧ b = 20 ∧ c = 6 ∧ d = 8 ∧ e = 75) →
  (a / b = c / d) ∧
  (a / b = 3 / 4) ∧
  (a / b = 0.75) ∧
  (a / b = e / 100) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_representations_l2615_261566


namespace NUMINAMATH_CALUDE_expression_equals_nine_l2615_261518

theorem expression_equals_nine : 3 * 3 - 3 + 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_nine_l2615_261518


namespace NUMINAMATH_CALUDE_negation_existential_proposition_l2615_261571

theorem negation_existential_proposition :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x > x - 2) ↔ (∀ x : ℝ, x > 0 → Real.log x ≤ x - 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_existential_proposition_l2615_261571


namespace NUMINAMATH_CALUDE_monthly_income_calculation_l2615_261510

theorem monthly_income_calculation (income : ℝ) : 
  (income / 2 - 20 = 100) → income = 240 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_calculation_l2615_261510


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2615_261519

theorem arithmetic_expression_equality : 10 - 9^2 + 8 * 7 + 6^2 - 5 * 4 + 3 - 2^3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2615_261519


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2615_261536

theorem quadratic_one_solution (n : ℝ) : 
  (∃! x : ℝ, 16 * x^2 + n * x + 4 = 0) ↔ (n = 16 ∨ n = -16) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2615_261536


namespace NUMINAMATH_CALUDE_system_solution_l2615_261585

theorem system_solution (x y z : ℝ) : 
  (x * (y^2 + z) = z * (z + x*y)) ∧ 
  (y * (z^2 + x) = x * (x + y*z)) ∧ 
  (z * (x^2 + y) = y * (y + x*z)) → 
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 0 ∧ y = 0 ∧ z = 0)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2615_261585


namespace NUMINAMATH_CALUDE_product_equals_zero_l2615_261538

theorem product_equals_zero (a : ℤ) (h : a = 3) : 
  (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l2615_261538


namespace NUMINAMATH_CALUDE_function_bound_l2615_261553

def function_property (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 1) - f x = 2 * x + 1

def bounded_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 0 1 → |f x| ≤ 1

theorem function_bound (f : ℝ → ℝ) 
  (h1 : function_property f) 
  (h2 : bounded_on_unit_interval f) : 
  ∀ x : ℝ, |f x| ≤ 2 + x^2 := by
  sorry

end NUMINAMATH_CALUDE_function_bound_l2615_261553


namespace NUMINAMATH_CALUDE_highest_points_is_38_l2615_261537

/-- The TRISQUARE game awards points for triangles and squares --/
structure TRISQUARE where
  small_triangles : ℕ
  large_triangles : ℕ
  small_squares : ℕ
  large_squares : ℕ
  triangle_points : ℕ
  square_points : ℕ

/-- Calculate the total points for a TRISQUARE game --/
def total_points (game : TRISQUARE) : ℕ :=
  (game.small_triangles + game.large_triangles) * game.triangle_points +
  (game.small_squares + game.large_squares) * game.square_points

/-- Theorem: The highest number of points achievable in the given TRISQUARE game is 38 --/
theorem highest_points_is_38 (game : TRISQUARE) 
  (h1 : game.small_triangles = 4)
  (h2 : game.large_triangles = 2)
  (h3 : game.small_squares = 4)
  (h4 : game.large_squares = 1)
  (h5 : game.triangle_points = 3)
  (h6 : game.square_points = 4) :
  total_points game = 38 := by
  sorry

#check highest_points_is_38

end NUMINAMATH_CALUDE_highest_points_is_38_l2615_261537


namespace NUMINAMATH_CALUDE_painted_equals_unpainted_l2615_261563

/-- Represents a cube with edge length n, painted on two adjacent faces and sliced into unit cubes -/
structure PaintedCube where
  n : ℕ
  n_gt_two : n > 2

/-- The number of smaller cubes with exactly two faces painted -/
def two_faces_painted (c : PaintedCube) : ℕ := c.n - 2

/-- The number of smaller cubes completely without paint -/
def unpainted (c : PaintedCube) : ℕ := (c.n - 2)^3

/-- Theorem stating that the number of cubes with two faces painted equals the number of unpainted cubes if and only if n = 3 -/
theorem painted_equals_unpainted (c : PaintedCube) : 
  two_faces_painted c = unpainted c ↔ c.n = 3 := by
  sorry

end NUMINAMATH_CALUDE_painted_equals_unpainted_l2615_261563


namespace NUMINAMATH_CALUDE_courtney_marbles_count_l2615_261562

/-- The number of marbles in Courtney's first jar -/
def first_jar : ℕ := 80

/-- The number of marbles in Courtney's second jar -/
def second_jar : ℕ := 2 * first_jar

/-- The number of marbles in Courtney's third jar -/
def third_jar : ℕ := first_jar / 4

/-- The total number of marbles Courtney has -/
def total_marbles : ℕ := first_jar + second_jar + third_jar

theorem courtney_marbles_count : total_marbles = 260 := by
  sorry

end NUMINAMATH_CALUDE_courtney_marbles_count_l2615_261562


namespace NUMINAMATH_CALUDE_xyz_product_zero_l2615_261507

theorem xyz_product_zero (x y z : ℝ) 
  (eq1 : x + 1/y = 1) 
  (eq2 : y + 1/z = 1) 
  (eq3 : z + 1/x = 1) : 
  x * y * z = 0 := by
  sorry

end NUMINAMATH_CALUDE_xyz_product_zero_l2615_261507


namespace NUMINAMATH_CALUDE_pond_a_twice_pond_b_total_frogs_is_48_l2615_261528

/-- The number of frogs in Pond A -/
def frogs_in_pond_a : ℕ := 32

/-- The number of frogs in Pond B -/
def frogs_in_pond_b : ℕ := frogs_in_pond_a / 2

/-- Pond A has twice as many frogs as Pond B -/
theorem pond_a_twice_pond_b : frogs_in_pond_a = 2 * frogs_in_pond_b := by sorry

/-- The total number of frogs in both ponds -/
def total_frogs : ℕ := frogs_in_pond_a + frogs_in_pond_b

/-- Theorem: The total number of frogs in both ponds is 48 -/
theorem total_frogs_is_48 : total_frogs = 48 := by sorry

end NUMINAMATH_CALUDE_pond_a_twice_pond_b_total_frogs_is_48_l2615_261528


namespace NUMINAMATH_CALUDE_game_ends_with_two_l2615_261546

/-- Represents the state of the game board -/
structure GameBoard where
  ones : ℕ
  twos : ℕ

/-- Represents a move in the game -/
inductive Move
  | EraseOnes
  | EraseTwos
  | EraseOneTwo

/-- Applies a move to the game board -/
def applyMove (board : GameBoard) (move : Move) : GameBoard :=
  match move with
  | Move.EraseOnes => { ones := board.ones - 2, twos := board.twos + 1 }
  | Move.EraseTwos => { ones := board.ones, twos := board.twos - 1 }
  | Move.EraseOneTwo => { ones := board.ones - 1, twos := board.twos }

/-- The initial state of the game board -/
def initialBoard : GameBoard := { ones := 10, twos := 10 }

/-- Predicate to check if the game is over -/
def gameOver (board : GameBoard) : Prop :=
  board.ones + board.twos = 1

/-- Theorem stating that the game always ends with a two -/
theorem game_ends_with_two :
  ∀ (sequence : List Move),
    let finalBoard := sequence.foldl applyMove initialBoard
    gameOver finalBoard → finalBoard.twos = 1 :=
  sorry

end NUMINAMATH_CALUDE_game_ends_with_two_l2615_261546


namespace NUMINAMATH_CALUDE_school_pizza_profit_l2615_261549

theorem school_pizza_profit :
  let num_pizzas : ℕ := 55
  let pizza_cost : ℚ := 685 / 100
  let slices_per_pizza : ℕ := 8
  let slice_price : ℚ := 1
  let total_revenue : ℚ := num_pizzas * slices_per_pizza * slice_price
  let total_cost : ℚ := num_pizzas * pizza_cost
  let profit : ℚ := total_revenue - total_cost
  profit = 6325 / 100 := by
  sorry

end NUMINAMATH_CALUDE_school_pizza_profit_l2615_261549


namespace NUMINAMATH_CALUDE_books_read_l2615_261532

/-- The number of books read in the 'crazy silly school' series -/
theorem books_read (total_books : ℕ) (unread_books : ℕ) (h1 : total_books = 20) (h2 : unread_books = 5) :
  total_books - unread_books = 15 := by
  sorry

end NUMINAMATH_CALUDE_books_read_l2615_261532


namespace NUMINAMATH_CALUDE_expression_evaluation_l2615_261501

theorem expression_evaluation : 5 * 402 + 4 * 402 + 3 * 402 + 401 = 5225 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2615_261501


namespace NUMINAMATH_CALUDE_nelly_earnings_per_night_l2615_261504

/-- Calculates Nelly's earnings per night babysitting given the pizza party conditions -/
theorem nelly_earnings_per_night (total_people : ℕ) (pizza_cost : ℚ) (people_per_pizza : ℕ) (babysitting_nights : ℕ) : 
  total_people = 15 →
  pizza_cost = 12 →
  people_per_pizza = 3 →
  babysitting_nights = 15 →
  (total_people : ℚ) / (people_per_pizza : ℚ) * pizza_cost / (babysitting_nights : ℚ) = 4 := by
  sorry

#check nelly_earnings_per_night

end NUMINAMATH_CALUDE_nelly_earnings_per_night_l2615_261504


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2615_261512

theorem decimal_sum_to_fraction :
  (0.1 : ℚ) + 0.02 + 0.003 + 0.0004 + 0.00005 + 0.000006 + 0.0000007 = 1234567 / 10000000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2615_261512


namespace NUMINAMATH_CALUDE_meiosis_fertilization_result_l2615_261531

/-- Represents a genetic combination -/
structure GeneticCombination where
  -- Add necessary fields

/-- Represents a gamete -/
structure Gamete where
  -- Add necessary fields

/-- Represents an organism -/
structure Organism where
  genetic_combination : GeneticCombination

/-- Meiosis process -/
def meiosis (parent : Organism) : List Gamete :=
  sorry

/-- Fertilization process -/
def fertilization (gamete1 gamete2 : Gamete) : Organism :=
  sorry

/-- Predicate to check if two genetic combinations are different -/
def are_different (gc1 gc2 : GeneticCombination) : Prop :=
  sorry

theorem meiosis_fertilization_result 
  (parent1 parent2 : Organism) : 
  ∃ (offspring : Organism), 
    (∃ (g1 : Gamete) (g2 : Gamete), 
      g1 ∈ meiosis parent1 ∧ 
      g2 ∈ meiosis parent2 ∧ 
      offspring = fertilization g1 g2) ∧
    are_different offspring.genetic_combination parent1.genetic_combination ∧
    are_different offspring.genetic_combination parent2.genetic_combination :=
  sorry

end NUMINAMATH_CALUDE_meiosis_fertilization_result_l2615_261531


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_150_l2615_261543

theorem closest_integer_to_cube_root_150 :
  ∀ n : ℤ, |n^3 - 150| ≥ |5^3 - 150| := by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_150_l2615_261543


namespace NUMINAMATH_CALUDE_number_difference_l2615_261557

theorem number_difference (x y : ℤ) : 
  x + y = 50 → 
  y = 19 → 
  x < 2 * y → 
  2 * y - x = 7 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l2615_261557


namespace NUMINAMATH_CALUDE_farmer_cow_division_l2615_261587

theorem farmer_cow_division (herd : ℕ) : 
  (herd / 3 : ℕ) + (herd / 6 : ℕ) + (herd / 8 : ℕ) + 9 = herd → herd = 24 := by
  sorry

end NUMINAMATH_CALUDE_farmer_cow_division_l2615_261587


namespace NUMINAMATH_CALUDE_ways_to_pay_100_l2615_261547

/-- Represents the available coin denominations -/
def CoinDenominations : List Nat := [1, 2, 10, 20, 50]

/-- Calculates the number of ways to pay a given amount using the available coin denominations -/
def waysToPayAmount (amount : Nat) : Nat :=
  sorry -- Implementation details omitted

/-- Theorem stating that there are 784 ways to pay 100 using the given coin denominations -/
theorem ways_to_pay_100 : waysToPayAmount 100 = 784 := by
  sorry

end NUMINAMATH_CALUDE_ways_to_pay_100_l2615_261547


namespace NUMINAMATH_CALUDE_min_rectangles_for_problem_figure_l2615_261534

/-- Represents a corner in the figure -/
inductive Corner
| Type1
| Type2

/-- Represents a set of three Type2 corners -/
structure CornerSet :=
  (corners : Fin 3 → Corner)
  (all_type2 : ∀ i, corners i = Corner.Type2)

/-- The figure with its corner structure -/
structure Figure :=
  (total_corners : Nat)
  (type1_corners : Nat)
  (type2_corners : Nat)
  (corner_sets : Nat)
  (valid_total : total_corners = type1_corners + type2_corners)
  (valid_type2 : type2_corners = 3 * corner_sets)

/-- The minimum number of rectangles needed to cover the figure -/
def min_rectangles (f : Figure) : Nat :=
  f.type1_corners + f.corner_sets

/-- The specific figure from the problem -/
def problem_figure : Figure :=
  { total_corners := 24
  , type1_corners := 12
  , type2_corners := 12
  , corner_sets := 4
  , valid_total := by rfl
  , valid_type2 := by rfl }

theorem min_rectangles_for_problem_figure :
  min_rectangles problem_figure = 12 := by sorry

end NUMINAMATH_CALUDE_min_rectangles_for_problem_figure_l2615_261534


namespace NUMINAMATH_CALUDE_calvin_prevents_hobbes_win_l2615_261569

/-- Represents a position on the integer lattice -/
structure Position where
  x : ℤ
  y : ℤ

/-- The game state -/
structure GameState where
  position : Position
  chosenIntegers : Set ℤ

/-- Calvin's strategy function -/
def calvinsStrategy (state : GameState) : Position := sorry

/-- Theorem stating Calvin can always prevent Hobbes from winning -/
theorem calvin_prevents_hobbes_win :
  ∀ (state : GameState),
  let newPos := calvinsStrategy state
  ∀ a b : ℤ,
    a ∉ state.chosenIntegers →
    b ∉ state.chosenIntegers →
    a ≠ (newPos.x - state.position.x) →
    b ≠ (newPos.y - state.position.y) →
    Position.mk (newPos.x + a) (newPos.y + b) ≠ Position.mk 0 0 :=
by sorry

end NUMINAMATH_CALUDE_calvin_prevents_hobbes_win_l2615_261569


namespace NUMINAMATH_CALUDE_chipmunks_went_away_count_l2615_261513

/-- Represents the chipmunk population in a forest --/
structure ChipmunkForest where
  originalFamilies : ℕ
  remainingFamilies : ℕ
  avgMembersRemaining : ℕ
  avgMembersLeft : ℕ

/-- Calculates the number of chipmunks that went away --/
def chipmunksWentAway (forest : ChipmunkForest) : ℕ :=
  (forest.originalFamilies - forest.remainingFamilies) * forest.avgMembersLeft

/-- Theorem stating the number of chipmunks that went away --/
theorem chipmunks_went_away_count (forest : ChipmunkForest) 
  (h1 : forest.originalFamilies = 86)
  (h2 : forest.remainingFamilies = 21)
  (h3 : forest.avgMembersRemaining = 15)
  (h4 : forest.avgMembersLeft = 18) :
  chipmunksWentAway forest = 1170 := by
  sorry

#eval chipmunksWentAway { originalFamilies := 86, remainingFamilies := 21, avgMembersRemaining := 15, avgMembersLeft := 18 }

end NUMINAMATH_CALUDE_chipmunks_went_away_count_l2615_261513


namespace NUMINAMATH_CALUDE_jelly_bean_ratio_l2615_261542

/-- Given 1200 total jelly beans divided between two jars X and Y, where jar X has 800 jelly beans,
    prove that the ratio of jelly beans in jar X to jar Y is 2:1. -/
theorem jelly_bean_ratio :
  let total_beans : ℕ := 1200
  let jar_x : ℕ := 800
  let jar_y : ℕ := total_beans - jar_x
  (jar_x : ℚ) / jar_y = 2 := by sorry

end NUMINAMATH_CALUDE_jelly_bean_ratio_l2615_261542


namespace NUMINAMATH_CALUDE_fib_999_1001_minus_1000_squared_l2615_261527

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem stating that F₉₉₉ * F₁₀₀₁ - F₁₀₀₀² = 1 for the Fibonacci sequence -/
theorem fib_999_1001_minus_1000_squared :
  fib 999 * fib 1001 - fib 1000 * fib 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fib_999_1001_minus_1000_squared_l2615_261527


namespace NUMINAMATH_CALUDE_no_prime_sum_53_l2615_261579

theorem no_prime_sum_53 : ¬ ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p + q = 53 := by sorry

end NUMINAMATH_CALUDE_no_prime_sum_53_l2615_261579


namespace NUMINAMATH_CALUDE_prime_equation_solution_l2615_261550

/-- Given that x and y are prime numbers, prove that x^y - y^x = xy^2 - 19 if and only if (x, y) = (2, 3) or (x, y) = (2, 7) -/
theorem prime_equation_solution (x y : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) :
  x^y - y^x = x*y^2 - 19 ↔ (x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 7) := by
  sorry

end NUMINAMATH_CALUDE_prime_equation_solution_l2615_261550


namespace NUMINAMATH_CALUDE_sandy_correct_sums_l2615_261523

theorem sandy_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (correct_marks : ℕ) 
  (incorrect_marks : ℕ) 
  (h1 : total_sums = 30)
  (h2 : total_marks = 45)
  (h3 : correct_marks = 3)
  (h4 : incorrect_marks = 2) :
  ∃ (correct_sums : ℕ), 
    correct_sums * correct_marks - (total_sums - correct_sums) * incorrect_marks = total_marks ∧
    correct_sums = 21 :=
by sorry

end NUMINAMATH_CALUDE_sandy_correct_sums_l2615_261523


namespace NUMINAMATH_CALUDE_increasing_function_condition_l2615_261598

/-- The function f(x) = lg(x^2 - mx - m) is increasing on (1, +∞) iff m ≤ 1/2 -/
theorem increasing_function_condition (m : ℝ) :
  (∀ x > 1, StrictMono (fun x => Real.log (x^2 - m*x - m))) ↔ m ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l2615_261598


namespace NUMINAMATH_CALUDE_bees_count_second_day_l2615_261559

theorem bees_count_second_day (first_day_count : ℕ) (second_day_multiplier : ℕ) :
  first_day_count = 144 →
  second_day_multiplier = 3 →
  first_day_count * second_day_multiplier = 432 :=
by
  sorry

end NUMINAMATH_CALUDE_bees_count_second_day_l2615_261559


namespace NUMINAMATH_CALUDE_csc_negative_330_degrees_l2615_261526

-- Define the cosecant function
noncomputable def csc (θ : Real) : Real := 1 / Real.sin θ

-- State the theorem
theorem csc_negative_330_degrees : csc ((-330 : Real) * Real.pi / 180) = 2 := by
  sorry

end NUMINAMATH_CALUDE_csc_negative_330_degrees_l2615_261526


namespace NUMINAMATH_CALUDE_not_both_perfect_cubes_l2615_261502

theorem not_both_perfect_cubes (n : ℕ) : 
  ¬(∃ a b : ℕ, (n + 2 = a^3) ∧ (n^2 + n + 1 = b^3)) := by
  sorry

end NUMINAMATH_CALUDE_not_both_perfect_cubes_l2615_261502


namespace NUMINAMATH_CALUDE_f_properties_l2615_261525

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem statement
theorem f_properties :
  -- 1. f(x) is increasing on (-∞, -1) and (1, +∞)
  (∀ x y, (x < y ∧ ((x < -1 ∧ y < -1) ∨ (x > 1 ∧ y > 1))) → f x < f y) ∧
  -- 2. f(x) is decreasing on (-1, 1)
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  -- 3. The maximum value of f(x) on [-3, 2] is 2
  (∀ x, -3 ≤ x ∧ x ≤ 2 → f x ≤ 2) ∧
  (∃ x, -3 ≤ x ∧ x ≤ 2 ∧ f x = 2) ∧
  -- 4. The minimum value of f(x) on [-3, 2] is -18
  (∀ x, -3 ≤ x ∧ x ≤ 2 → f x ≥ -18) ∧
  (∃ x, -3 ≤ x ∧ x ≤ 2 ∧ f x = -18) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2615_261525


namespace NUMINAMATH_CALUDE_supermarket_spending_l2615_261592

/-- Represents the total amount spent at the supermarket -/
def total_spent : ℝ := 120

/-- Represents the amount spent on candy -/
def candy_spent : ℝ := 8

/-- Theorem stating the total amount spent at the supermarket -/
theorem supermarket_spending :
  (1/2 + 1/3 + 1/10) * total_spent + candy_spent = total_spent :=
by sorry

end NUMINAMATH_CALUDE_supermarket_spending_l2615_261592


namespace NUMINAMATH_CALUDE_michaels_coins_value_l2615_261572

theorem michaels_coins_value (p n : ℕ) : 
  p + n = 15 ∧ 
  n + 2 = 2 * (p - 2) →
  p * 1 + n * 5 = 47 :=
by sorry

end NUMINAMATH_CALUDE_michaels_coins_value_l2615_261572


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2615_261554

/-- Given a complex number z satisfying z(1+i) = 2, prove that z has a positive real part and a negative imaginary part. -/
theorem z_in_fourth_quadrant (z : ℂ) (h : z * (1 + Complex.I) = 2) :
  0 < z.re ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l2615_261554


namespace NUMINAMATH_CALUDE_max_clowns_proof_l2615_261517

/-- The number of distinct colors available -/
def num_colors : ℕ := 12

/-- The minimum number of colors each clown must use -/
def min_colors_per_clown : ℕ := 5

/-- The maximum number of clowns that can use any particular color -/
def max_clowns_per_color : ℕ := 20

/-- The set of all possible color combinations for clowns -/
def color_combinations : Finset (Finset (Fin num_colors)) :=
  (Finset.powerset (Finset.univ : Finset (Fin num_colors))).filter (fun s => s.card ≥ min_colors_per_clown)

/-- The maximum number of clowns satisfying all conditions -/
def max_clowns : ℕ := num_colors * max_clowns_per_color

theorem max_clowns_proof :
  (∀ s : Finset (Fin num_colors), s ∈ color_combinations → s.card ≥ min_colors_per_clown) ∧
  (∀ c : Fin num_colors, (color_combinations.filter (fun s => c ∈ s)).card ≤ max_clowns_per_color) →
  color_combinations.card ≥ max_clowns ∧
  max_clowns = 240 := by
  sorry

end NUMINAMATH_CALUDE_max_clowns_proof_l2615_261517


namespace NUMINAMATH_CALUDE_range_of_a_l2615_261505

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  (f = λ x => x * |x^2 - a|) →
  (∃ x ∈ Set.Icc 1 2, f x < 2) →
  -1 < a ∧ a < 5 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2615_261505


namespace NUMINAMATH_CALUDE_gcd_sum_ten_l2615_261521

theorem gcd_sum_ten (n : ℕ) : 
  (Nat.gcd 6 n + Nat.gcd 8 (2 * n) = 10) ↔ 
  (∃ t : ℕ, n = 12 * t + 4 ∨ n = 12 * t + 6 ∨ n = 12 * t + 8) :=
sorry

end NUMINAMATH_CALUDE_gcd_sum_ten_l2615_261521


namespace NUMINAMATH_CALUDE_upper_bound_of_prime_set_l2615_261508

theorem upper_bound_of_prime_set (A : Set ℕ) : 
  (∀ x ∈ A, Nat.Prime x) →   -- A contains only prime numbers
  (∃ a ∈ A, a > 62) →        -- Lower bound is greater than 62
  (∀ a ∈ A, a > 62) →        -- All elements are greater than 62
  (∃ max min : ℕ, max ∈ A ∧ min ∈ A ∧ max - min = 16 ∧
    ∀ a ∈ A, min ≤ a ∧ a ≤ max) →  -- Range of A is 16
  (∃ x ∈ A, ∀ y ∈ A, y ≤ x) →  -- A has a maximum element
  (∃ x ∈ A, x = 83 ∧ ∀ y ∈ A, y ≤ x) :=  -- The upper bound (maximum) is 83
by sorry

end NUMINAMATH_CALUDE_upper_bound_of_prime_set_l2615_261508


namespace NUMINAMATH_CALUDE_r_daily_earnings_l2615_261529

/-- Represents the daily earnings of individuals p, q, and r -/
structure Earnings where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The conditions given in the problem -/
def problem_conditions (e : Earnings) : Prop :=
  9 * (e.p + e.q + e.r) = 1620 ∧
  5 * (e.p + e.r) = 600 ∧
  7 * (e.q + e.r) = 910

/-- The theorem stating that given the problem conditions, r's daily earnings are 70 -/
theorem r_daily_earnings (e : Earnings) : 
  problem_conditions e → e.r = 70 := by
  sorry

#check r_daily_earnings

end NUMINAMATH_CALUDE_r_daily_earnings_l2615_261529


namespace NUMINAMATH_CALUDE_division_remainder_proof_l2615_261558

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 729 → divisor = 38 → quotient = 19 → 
  dividend = divisor * quotient + remainder → remainder = 7 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l2615_261558


namespace NUMINAMATH_CALUDE_problem_solution_l2615_261573

theorem problem_solution (m n : ℕ) (x : ℝ) 
  (h1 : 2^m = 8)
  (h2 : 2^n = 32)
  (h3 : x = 2^m - 1) :
  (2^(2*m + n - 4) = 128) ∧ 
  (1 + 4^(m+1) = 4*x^2 + 8*x + 5) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2615_261573


namespace NUMINAMATH_CALUDE_transformed_sine_sum_l2615_261568

theorem transformed_sine_sum (ω A a φ : ℝ) (hω : ω > 0) (hA : A > 0) (ha : a > 0) (hφ : 0 < φ ∧ φ < π)
  (h : ∀ x, A * Real.sin (ω * x - φ) + a = 3 * Real.sin (2 * x - π / 6) + 1) :
  A + a + ω + φ = 16 / 3 + 11 * π / 12 := by
  sorry

end NUMINAMATH_CALUDE_transformed_sine_sum_l2615_261568


namespace NUMINAMATH_CALUDE_dolphins_score_l2615_261591

theorem dolphins_score (total_score winning_margin : ℕ) : 
  total_score = 48 → winning_margin = 20 → 
  ∃ (sharks_score dolphins_score : ℕ), 
    sharks_score + dolphins_score = total_score ∧ 
    sharks_score = dolphins_score + winning_margin ∧
    dolphins_score = 14 := by
  sorry

end NUMINAMATH_CALUDE_dolphins_score_l2615_261591


namespace NUMINAMATH_CALUDE_no_equal_digit_sum_decomposition_l2615_261580

def digit_sum (n : ℕ) : ℕ := sorry

theorem no_equal_digit_sum_decomposition :
  ¬ ∃ (B C : ℕ), B + C = 999999999 ∧ digit_sum B = digit_sum C := by sorry

end NUMINAMATH_CALUDE_no_equal_digit_sum_decomposition_l2615_261580


namespace NUMINAMATH_CALUDE_relay_race_time_l2615_261570

/-- The relay race problem -/
theorem relay_race_time (athlete1 athlete2 athlete3 athlete4 total : ℕ) : 
  athlete1 = 55 →
  athlete2 = athlete1 + 10 →
  athlete3 = athlete2 - 15 →
  athlete4 = athlete1 - 25 →
  total = athlete1 + athlete2 + athlete3 + athlete4 →
  total = 200 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_time_l2615_261570


namespace NUMINAMATH_CALUDE_rhombus_count_in_divided_equilateral_triangle_l2615_261530

/-- Given an equilateral triangle ABC with each side divided into n equal parts,
    and parallel lines drawn through each division point to form a grid of smaller
    equilateral triangles, the number of rhombuses with side length 1/n in this grid
    is equal to 3 * C(n,2), where C(n,2) is the binomial coefficient. -/
theorem rhombus_count_in_divided_equilateral_triangle (n : ℕ) :
  let num_rhombuses := 3 * (n.choose 2)
  num_rhombuses = 3 * (n * (n - 1)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_count_in_divided_equilateral_triangle_l2615_261530


namespace NUMINAMATH_CALUDE_clock_cost_price_l2615_261539

theorem clock_cost_price (total_clocks : ℕ) (clocks_sold_10_percent : ℕ) (clocks_sold_20_percent : ℕ)
  (profit_10_percent : ℝ) (profit_20_percent : ℝ) (uniform_profit : ℝ) (price_difference : ℝ) :
  total_clocks = 90 →
  clocks_sold_10_percent = 40 →
  clocks_sold_20_percent = 50 →
  profit_10_percent = 0.1 →
  profit_20_percent = 0.2 →
  uniform_profit = 0.15 →
  price_difference = 40 →
  ∃ (cost_price : ℝ),
    cost_price * (clocks_sold_10_percent * (1 + profit_10_percent) + 
      clocks_sold_20_percent * (1 + profit_20_percent)) - 
    cost_price * total_clocks * (1 + uniform_profit) = price_difference ∧
    cost_price = 80 :=
by sorry


end NUMINAMATH_CALUDE_clock_cost_price_l2615_261539


namespace NUMINAMATH_CALUDE_woodburning_cost_l2615_261511

def woodburning_problem (num_sold : ℕ) (price_per_item : ℚ) (profit : ℚ) : Prop :=
  let total_revenue := num_sold * price_per_item
  let cost_of_wood := total_revenue - profit
  cost_of_wood = 100

theorem woodburning_cost :
  woodburning_problem 20 15 200 := by
  sorry

end NUMINAMATH_CALUDE_woodburning_cost_l2615_261511


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2615_261576

/-- The sum of the lengths of all sides of a rectangle with sides 9 cm and 11 cm is 40 cm. -/
theorem rectangle_perimeter (length width : ℝ) (h1 : length = 9) (h2 : width = 11) :
  2 * (length + width) = 40 := by
  sorry

#check rectangle_perimeter

end NUMINAMATH_CALUDE_rectangle_perimeter_l2615_261576


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_19_l2615_261545

/-- Triangle PQR with given properties -/
structure Triangle where
  /-- Length of side PQ -/
  PQ : ℝ
  /-- Length of side QR -/
  QR : ℝ
  /-- Length of side PR -/
  PR : ℝ
  /-- Angle PQR equals angle PRQ -/
  angle_equality : PQ = PR

/-- The perimeter of a triangle is the sum of its side lengths -/
def perimeter (t : Triangle) : ℝ := t.PQ + t.QR + t.PR

/-- Theorem: The perimeter of the given triangle is 19 -/
theorem triangle_perimeter_is_19 (t : Triangle) 
  (h1 : t.QR = 5) 
  (h2 : t.PR = 7) : 
  perimeter t = 19 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_is_19_l2615_261545


namespace NUMINAMATH_CALUDE_function_satisfying_condition_is_zero_function_l2615_261555

theorem function_satisfying_condition_is_zero_function 
  (f : ℝ → ℝ) (h : ∀ x y : ℝ, f x + f y = f (f x * f y)) : 
  ∀ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_satisfying_condition_is_zero_function_l2615_261555


namespace NUMINAMATH_CALUDE_triangle_existence_l2615_261582

theorem triangle_existence (y : ℕ+) : 
  (y + 1 + 6 > y^2 + 2*y + 3) ∧ 
  (y + 1 + (y^2 + 2*y + 3) > 6) ∧ 
  (6 + (y^2 + 2*y + 3) > y + 1) ↔ 
  y = 2 := by sorry

end NUMINAMATH_CALUDE_triangle_existence_l2615_261582


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_two_l2615_261515

theorem least_subtraction_for_divisibility_by_two (n : ℕ) (h : n = 9671) :
  ∃ (k : ℕ), k = 1 ∧ 
  (∀ (m : ℕ), m < k → ¬(∃ (q : ℕ), n - m = 2 * q)) ∧
  (∃ (q : ℕ), n - k = 2 * q) :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_two_l2615_261515


namespace NUMINAMATH_CALUDE_dice_probability_l2615_261503

/-- The number of sides on each die -/
def num_sides : ℕ := 12

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability of rolling a 'low' number (1-8) -/
def prob_low : ℚ := 2/3

/-- The probability of rolling a 'mid' or 'high' number (9-12) -/
def prob_mid_high : ℚ := 1/3

/-- The number of ways to choose 2 dice out of 5 -/
def choose_two_from_five : ℕ := 10

/-- The probability of the desired outcome -/
theorem dice_probability : 
  (choose_two_from_five : ℚ) * prob_low^2 * prob_mid_high^3 = 40/243 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l2615_261503


namespace NUMINAMATH_CALUDE_susan_walk_distance_l2615_261548

theorem susan_walk_distance (total_distance : ℝ) (erin_susan_diff : ℝ) (daniel_susan_ratio : ℝ) :
  total_distance = 32 ∧
  erin_susan_diff = 3 ∧
  daniel_susan_ratio = 2 →
  ∃ susan_distance : ℝ,
    susan_distance + (susan_distance - erin_susan_diff) + (daniel_susan_ratio * susan_distance) = total_distance ∧
    susan_distance = 8.75 := by
  sorry

end NUMINAMATH_CALUDE_susan_walk_distance_l2615_261548
