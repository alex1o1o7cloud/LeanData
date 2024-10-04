import Mathlib

namespace jill_sod_area_needed_l487_487047

def plot_width : ℕ := 200
def plot_length : ℕ := 50
def sidewalk_width : ℕ := 3
def sidewalk_length : ℕ := 50
def flower_bed1_depth : ℕ := 4
def flower_bed1_length : ℕ := 25
def flower_bed1_count : ℕ := 2
def flower_bed2_width : ℕ := 10
def flower_bed2_length : ℕ := 12
def flower_bed3_width : ℕ := 7
def flower_bed3_length : ℕ := 8

theorem jill_sod_area_needed :
  (plot_width * plot_length) - 
  (sidewalk_width * sidewalk_length + 
   flower_bed1_depth * flower_bed1_length * flower_bed1_count + 
   flower_bed2_width * flower_bed2_length + 
   flower_bed3_width * flower_bed3_length) = 9474 :=
by
  sorry

end jill_sod_area_needed_l487_487047


namespace raine_steps_l487_487111

theorem raine_steps (steps_per_trip : ℕ) (num_days : ℕ) (total_steps : ℕ) : 
  steps_per_trip = 150 → 
  num_days = 5 → 
  total_steps = steps_per_trip * 2 * num_days → 
  total_steps = 1500 := 
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end raine_steps_l487_487111


namespace max_intersections_two_circles_three_lines_l487_487572

theorem max_intersections_two_circles_three_lines :
  ∀ (C1 C2 : ℝ × ℝ × ℝ) (L1 L2 L3 : ℝ × ℝ × ℝ), 
  C1 ≠ C2 → L1 ≠ L2 → L2 ≠ L3 → L1 ≠ L3 →
  ∃ (P : ℕ), P = 17 :=
by 
  sorry

end max_intersections_two_circles_three_lines_l487_487572


namespace sum_reciprocals_factors_12_l487_487630

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487630


namespace sum_of_reciprocals_factors_12_l487_487783

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487783


namespace sum_reciprocals_of_factors_12_l487_487852

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487852


namespace projection_matrix_correct_l487_487066

noncomputable def Q_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  !![
    [7/9, 1/9, -2/9],
    [1/9, 11/18, 1/9],
    [-2/9, 1/9, 7/9]
  ]

def normal_vector : Vector (Fin 3) ℝ :=
  !![
    [2],
    [-1],
    [2]
  ]

def projection_on_plane (v : Vector (Fin 3) ℝ) : Vector (Fin 3) ℝ :=
  Q_matrix ⬝ v

theorem projection_matrix_correct (v : Vector (Fin 3) ℝ) :
  projection_on_plane v = Q_matrix ⬝ v :=
sorry

end projection_matrix_correct_l487_487066


namespace ellipse_constants_sum_l487_487133

/-- Given the center of the ellipse at (h, k) = (3, -5),
    the semi-major axis a = 7,
    and the semi-minor axis b = 4,
    prove that h + k + a + b = 9. -/
theorem ellipse_constants_sum :
  let h := 3
  let k := -5
  let a := 7
  let b := 4
  h + k + a + b = 9 :=
by
  let h := 3
  let k := -5
  let a := 7
  let b := 4
  sorry

end ellipse_constants_sum_l487_487133


namespace sum_of_reciprocals_factors_12_l487_487831

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487831


namespace find_point_M_l487_487034

-- Definitions according to the conditions
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def distance_3d (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

-- Given points A and B
def A : Point3D := ⟨4, 3, 2⟩
def B : Point3D := ⟨2, 5, 4⟩

-- The point M on the y-axis
def M (y : ℝ) : Point3D := ⟨0, y, 0⟩

-- The theorem to be proved
theorem find_point_M :
  ∃ y : ℝ, distance_3d (M y) A = distance_3d (M y) B ∧ M y = ⟨0, 4, 0⟩ :=
sorry

end find_point_M_l487_487034


namespace sum_reciprocals_factors_12_l487_487891

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487891


namespace sum_reciprocals_factors_12_l487_487617

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487617


namespace impossible_to_achieve_12_percent_return_l487_487223

-- Define the stock parameters and their individual returns
def stock_A_price : ℝ := 52
def stock_A_dividend_rate : ℝ := 0.09
def stock_A_transaction_fee_rate : ℝ := 0.02

def stock_B_price : ℝ := 80
def stock_B_dividend_rate : ℝ := 0.07
def stock_B_transaction_fee_rate : ℝ := 0.015

def stock_C_price : ℝ := 40
def stock_C_dividend_rate : ℝ := 0.10
def stock_C_transaction_fee_rate : ℝ := 0.01

def tax_rate : ℝ := 0.10
def desired_return : ℝ := 0.12

theorem impossible_to_achieve_12_percent_return :
  false :=
sorry

end impossible_to_achieve_12_percent_return_l487_487223


namespace sum_of_reciprocals_of_factors_of_12_l487_487672

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487672


namespace detect_two_counterfeits_detect_three_counterfeits_l487_487437

-- Definitions for counterfeit detection conditions and problem
def distinct_natural_denominations (banknotes : List ℕ) : Prop :=
  (banknotes.nodup)

def sum_of_genuine (detector : List ℕ → ℕ) (genuine_notes : List ℕ) : ℕ :=
  detector genuine_notes

-- Prove for N=2
theorem detect_two_counterfeits (banknotes : List ℕ) (detector : List ℕ → ℕ) (N : ℕ) 
  (h₁ : distinct_natural_denominations banknotes) (h₂ : N = 2) :
  ∃ (S₀ S₁ : List ℕ), detector S₀ = detector banknotes ∧ detector S₁ = detector S₀ - (sum_of_genuine detector S₀ - detector banknotes) :=
sorry

-- Prove for N=3
theorem detect_three_counterfeits (banknotes : List ℕ) (detector : List ℕ → ℕ) (N : ℕ) 
  (h₁ : distinct_natural_denominations banknotes) (h₂ : N = 3) :
  ∃ (S₀ S₁ S₂ : List ℕ), detector S₀ = detector banknotes ∧ detector S₁ = detector S₀ - (sum_of_genuine detector S₀ - detector banknotes) ∧ detector S₂ = detector S₁ - (sum_of_genuine detector S₁ - detector banknotes) :=
sorry

end detect_two_counterfeits_detect_three_counterfeits_l487_487437


namespace sum_of_reciprocals_of_factors_of_12_l487_487797

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487797


namespace sum_of_reciprocals_factors_12_l487_487718

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487718


namespace total_weight_of_container_l487_487206

variables {Weight : Type} [LinearOrderedField Weight]

def weight_of_copper : Weight := 90
def weight_of_steel (w_copper : Weight) := w_copper + 20
def weight_of_tin (w_steel : Weight) := w_steel / 2
def weight_of_aluminum (w_tin w_copper : Weight) := w_tin + 10

theorem total_weight_of_container (w_copper w_steel w_tin w_aluminum : Weight)
  (hc : w_copper = weight_of_copper)
  (hs : w_steel = weight_of_steel w_copper)
  (ht : w_tin = weight_of_tin w_steel)
  (ha : w_aluminum = weight_of_aluminum w_tin w_copper) :
  10 * w_steel + 15 * w_tin + 12 * w_copper + 8 * w_aluminum = 3525 :=
begin
  sorry
end

end total_weight_of_container_l487_487206


namespace sum_reciprocals_of_factors_of_12_l487_487966

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487966


namespace sum_reciprocals_factors_12_l487_487879

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487879


namespace total_square_miles_of_plains_l487_487147

-- Defining conditions
def region_east_of_b : ℕ := 200
def region_east_of_a : ℕ := region_east_of_b - 50

-- To test this statement in Lean 4
theorem total_square_miles_of_plains : region_east_of_a + region_east_of_b = 350 := by
  sorry

end total_square_miles_of_plains_l487_487147


namespace sub_complex_eq_l487_487165

theorem sub_complex_eq (a b : ℂ) (h1 : a = 5 - 3 * complex.I) (h2 : b = 4 + 3 * complex.I) : a - 3 * b = -7 - 12 * complex.I :=
by
  sorry

end sub_complex_eq_l487_487165


namespace dividend_in_terms_of_a_l487_487022

variable (a Q R D : ℕ)

-- Given conditions as hypotheses
def condition1 : Prop := D = 25 * Q
def condition2 : Prop := D = 7 * R
def condition3 : Prop := Q - R = 15
def condition4 : Prop := R = 3 * a

-- Prove that the dividend given these conditions equals the expected expression
theorem dividend_in_terms_of_a (a : ℕ) (Q : ℕ) (R : ℕ) (D : ℕ) :
  condition1 D Q → condition2 D R → condition3 Q R → condition4 R a →
  (D * Q + R) = 225 * a^2 + 1128 * a + 5625 :=
by
  intro h1 h2 h3 h4
  sorry

end dividend_in_terms_of_a_l487_487022


namespace circle_tangent_l487_487322

/-- Given a circle (x-1)^2 + (y-1)^2 = 4 and points A(a,0), B(-a,0) with a > 0,
  if there exists only one point P on the circle such that ∠APB = 90°, then a = 2 - sqrt 2 or a = 2 + sqrt 2. -/
theorem circle_tangent (a : ℝ) (h : a > 0)
  (P : ℝ × ℝ) (hP : (P.1 - 1)^2 + (P.2 - 1)^2 = 4)
  (h_angle : ∃ P : ℝ × ℝ, ((P.1 - 1)^2 + (P.2 - 1)^2 = 4) ∧
   (P ∈ Set.circle (1,1) 2) ∧ ∃ APB : ℝ, ∠ (a, 0) P (-a, 0) =  90) :
  a = 2 - Real.sqrt 2 ∨ a = 2 + Real.sqrt 2 := sorry

end circle_tangent_l487_487322


namespace Einstein_sold_25_cans_of_soda_l487_487279

def sell_snacks_proof : Prop :=
  let pizza_price := 12
  let fries_price := 0.30
  let soda_price := 2
  let goal := 500
  let pizza_boxes := 15
  let fries_packs := 40
  let still_needed := 258
  let earned_from_pizza := pizza_boxes * pizza_price
  let earned_from_fries := fries_packs * fries_price
  let total_earned := earned_from_pizza + earned_from_fries
  let total_have := goal - still_needed
  let earned_from_soda := total_have - total_earned
  let cans_of_soda_sold := earned_from_soda / soda_price
  cans_of_soda_sold = 25

theorem Einstein_sold_25_cans_of_soda : sell_snacks_proof := by
  sorry

end Einstein_sold_25_cans_of_soda_l487_487279


namespace sum_reciprocals_12_l487_487916

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487916


namespace sum_reciprocals_factors_of_12_l487_487674

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487674


namespace cyclist_total_heartbeats_l487_487217

theorem cyclist_total_heartbeats
  (heart_rate : ℕ := 120) -- beats per minute
  (race_distance : ℕ := 50) -- miles
  (pace : ℕ := 4) -- minutes per mile
  : (race_distance * pace) * heart_rate = 24000 := by
  sorry

end cyclist_total_heartbeats_l487_487217


namespace sum_reciprocals_factors_12_l487_487945

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487945


namespace sum_reciprocals_factors_12_l487_487624

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487624


namespace range_of_a_l487_487008

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x → a * exp (a * exp x + a) ≥ log (exp x + 1)) → 
  a ≥ 1 / Real.exp 1 :=
by
  sorry

end range_of_a_l487_487008


namespace building_c_floors_l487_487255

/-
  Building A has 4 floors.
  Building B has 9 more floors than Building A.
  Building C has six less than five times as many floors as Building B.
  Prove that Building C has 59 floors.
-/

theorem building_c_floors :
  let F_A := 4 in
  let F_B := F_A + 9 in
  let F_C := 5 * F_B - 6 in
  F_C = 59 :=
by
  let F_A := 4
  let F_B := F_A + 9
  let F_C := 5 * F_B - 6
  show F_C = 59
  sorry

end building_c_floors_l487_487255


namespace sum_reciprocals_factors_12_l487_487615

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487615


namespace sqrt_gt_sufficient_for_exp_gt_l487_487342

-- Given conditions
variable (a b : ℝ)

-- Definitions used directly from the conditions
def sqrt_gt (a b : ℝ) := sqrt a > sqrt b
def exp_gt (a b : ℝ) := exp a > exp b

-- Prove the relationship under given conditions
theorem sqrt_gt_sufficient_for_exp_gt (h : sqrt_gt a b) : exp_gt a b :=
by sorry

-- Show the sufficiency but not necessity
example : (sqrt_gt a b → exp_gt a b) ∧ ¬ (exp_gt a b → sqrt_gt a b) :=
by
  constructor
  . exact sqrt_gt_sufficient_for_exp_gt
  . sorry

end sqrt_gt_sufficient_for_exp_gt_l487_487342


namespace cubic_root_identity_l487_487072

theorem cubic_root_identity (x : ℝ) (r s : ℤ)
  (h1 : x = (r + Real.sqrt s : ℝ))
  (h2 : (r : ℝ) + Real.sqrt s = 21) :
  (Real.cbrt x + Real.cbrt (16 - x)) = 2 := sorry

end cubic_root_identity_l487_487072


namespace mika_stickers_l487_487094

def s1 : ℝ := 20.5
def s2 : ℝ := 26.3
def s3 : ℝ := 19.75
def s4 : ℝ := 6.25
def s5 : ℝ := 57.65
def s6 : ℝ := 15.8

theorem mika_stickers 
  (M : ℝ)
  (hM : M = s1 + s2 + s3 + s4 + s5 + s6) 
  : M = 146.25 :=
sorry

end mika_stickers_l487_487094


namespace DaisyDistance_l487_487489

def speedOfSound := 1100 -- feet per second
def timeDelay := 12 -- seconds
def feetPerMile := 5280 -- feet

def CalculateDistance (speed : ℕ) (time : ℕ) : ℕ := speed * time
def ConvertToMiles (distanceFeet : ℕ) (feetPerMile : ℕ) : ℚ := distanceFeet / feetPerMile.toRat
def RoundToQuarterMiles (distanceMiles : ℚ) : ℚ := (4 * distanceMiles).round() / 4

theorem DaisyDistance :
  let distanceFeet := CalculateDistance speedOfSound timeDelay
  let distanceMiles := ConvertToMiles distanceFeet feetPerMile
  RoundToQuarterMiles distanceMiles = 2.5 :=
by 
  sorry

end DaisyDistance_l487_487489


namespace sum_reciprocals_12_l487_487923

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487923


namespace sum_reciprocals_factors_of_12_l487_487692

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487692


namespace sum_of_reciprocals_of_factors_of_12_l487_487809

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487809


namespace digit_9_appearances_l487_487394

theorem digit_9_appearances (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 700) : 
  let digit_count := (List.range' 1 700).map (λ x, x.toString.count '9') in
  digit_count.sum = 140 :=
by
  sorry

end digit_9_appearances_l487_487394


namespace integral_sin_eq_zero_l487_487550

-- Define the function and the interval
def f (x : Real) : Real := sin x
def a : Real := 0
def b : Real := 2 * Real.pi

-- State the theorem without proving it
theorem integral_sin_eq_zero : ∫ x in a..b, f x = 0 :=
by
  sorry

end integral_sin_eq_zero_l487_487550


namespace sequence_equality_l487_487430

-- Define sequence a₁ = -2 and a_{n+1} = aₙ + n * 2^n
def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = -2 ∧ ∀ n > 0, a (n + 1) = a n + n * 2^n

-- Define the target expression for aₙ
def target_expression (n : ℕ) : ℤ :=
  (n - 2) * 2^n

theorem sequence_equality (a : ℕ → ℤ) (n : ℕ) (h : sequence a) : a n = target_expression n :=
by sorry

end sequence_equality_l487_487430


namespace find_m_l487_487325

variable (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ)
variable (q : ℝ)
variable (m : ℕ)

axiom geom_seq : ∀ n, a(n+1) = q * a n
axiom sum_geom_seq : ∀ n, S n = ∑ i in range (n+1), a i
axiom cond1 : ∀ n, 2 * a n + 2 * a (n + 2) + 5 * S n = 5 * S (n + 1)
axiom a1_value : q > 1
axiom bn_relation : ∀ n, b n = a n * |(Real.sin ((↑n + 1) * Real.pi / 2))|
axiom sum_bn : ∑ i in range m, b i = 340

theorem find_m : m = 8 ∨ m = 9 := by
  sorry

end find_m_l487_487325


namespace sym_diff_A_B_l487_487298

def set_difference {α : Type} (M N : set α) : set α :=
{ x | x ∈ M ∧ x ∉ N }

def sym_diff {α : Type} (M N : set α) : set α :=
(set_difference M N) ∪ (set_difference N M)

def A := { x : ℝ | x ≥ -9 / 4 }
def B := { x : ℝ | x < 0 }

theorem sym_diff_A_B : sym_diff A B = { x | x ≥ 0 ∨ x < -9 / 4 } :=
by
  sorry

end sym_diff_A_B_l487_487298


namespace sum_of_reciprocals_factors_12_l487_487775

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487775


namespace sector_angle_l487_487327

theorem sector_angle
  (r : ℝ) (S_sector : ℝ)
  (hr : r = 2)
  (hS : S_sector = 4) :
  ∃ (α : ℝ), 1 / 2 * α * r^2 = S_sector ∧ α = 2 :=
by
  use 2
  split
  sorry

end sector_angle_l487_487327


namespace sum_reciprocals_factors_12_l487_487743

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487743


namespace not_always_colorable_l487_487277

noncomputable def even_faces (P : Polyhedron) : Prop :=
  ∀ f ∈ faces P, even (sides_of_face f)

noncomputable def valid_coloring (P : Polyhedron) (color : Edge → Color) : Prop :=
  ∀ f ∈ faces P, (∃ c₁ c₂, c₁ ≠ c₂ ∧ count_edges_with_color f c₁ color = count_edges_with_color f c₂ color)

theorem not_always_colorable (P : Polyhedron) (h : even_faces P) :
  ¬ ∃ color : Edge → Color, valid_coloring P color :=
 by {
  sorry
  }

end not_always_colorable_l487_487277


namespace function_unique_l487_487282

open Nat

theorem function_unique (f : ℕ → ℕ) :
  (∀ x y : ℕ, x > 0 → y > 0 → f(x) + y * f(f(x)) ≤ x * (1 + f(y))) →
  (∀ x : ℕ, f(x) = x) :=
by
  intro h x
  sorry

end function_unique_l487_487282


namespace maximum_z_l487_487270

theorem maximum_z (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x * y + y * z + z * x = 3) : z ≤ 13 / 3 :=
sorry

end maximum_z_l487_487270


namespace sum_reciprocals_of_factors_of_12_l487_487967

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487967


namespace sum_of_reciprocals_of_factors_of_12_l487_487811

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487811


namespace similarity_of_T2_ratio_of_similarity_l487_487464

noncomputable def circumcenter (A B C : Point) : Point := sorry

structure Quadrilateral :=
(A B C D : Point)
(h_convex : Convex A B C D)
(h_no_circle : ¬∃ P : Circle, Circle.Contains P A ∧ Circle.Contains P B ∧ Circle.Contains P C ∧ Circle.Contains P D)

def T (quad : Quadrilateral) : Quadrilateral :=
{ A := circumcenter quad.B quad.C quad.D,
  B := circumcenter quad.A quad.C quad.D,
  C := circumcenter quad.A quad.B quad.D,
  D := circumcenter quad.A quad.B quad.C,
  h_convex := sorry,
  h_no_circle := sorry }

def T2 (quad : Quadrilateral) : Quadrilateral := T (T quad)

theorem similarity_of_T2 (ABCD : Quadrilateral) : Similar ABCD (T2 ABCD) := sorry

theorem ratio_of_similarity (ABCD : Quadrilateral) :
  similarity_ratio ABCD (T2 ABCD) = (sin (angle ABCD.A ABCD.C + angle ABCD.C ABCD.A))^2 / 
  (4 * sin (angle ABCD.A ABCD.B) * sin (angle ABCD.B ABCD.C) * sin (angle ABCD.C ABCD.D) * sin (angle ABCD.D ABCD.A)) := sorry

end similarity_of_T2_ratio_of_similarity_l487_487464


namespace sum_of_reciprocals_factors_12_l487_487829

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487829


namespace sum_reciprocals_factors_12_l487_487874

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487874


namespace inequality_proof_l487_487347

theorem inequality_proof (x y z : ℝ) (hx : -1 < x) (hy : -1 < y) (hz : -1 < z) :
    (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
sorry

end inequality_proof_l487_487347


namespace chameleons_never_same_color_l487_487490

theorem chameleons_never_same_color
  (g b c : ℕ) -- number of gray, brown, and crimson chameleons
  (h1 : g = 13) -- initial number of gray chameleons
  (h2 : b = 15) -- initial number of brown chameleons
  (h3 : c = 17) -- initial number of crimson chameleons
  (invariant : ∀ (g b c : ℕ),
    ((g + b + c = h1 + h2 + h3) → 
    (g - b) % 3 = (h1 - h2) % 3)) -- invariant condition
  : ¬ ∃ (g b c : ℕ), g + b + c = 45 ∧ (g = 0 ∨ b = 0 ∨ c = 0) :=
sorry

end chameleons_never_same_color_l487_487490


namespace sum_of_reciprocals_of_factors_of_12_l487_487763

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487763


namespace find_number_l487_487179

theorem find_number (x : ℝ) : (x / 2 = x - 5) → x = 10 :=
by
  intro h
  sorry

end find_number_l487_487179


namespace angle_difference_constant_l487_487076

open Real EuclideanSpace

-- Definitions based on conditions in Step a)
def triangle := {A B C : Point} (h : B ≠ C)
def isosceles (ABC : triangle) := dist ABC.ABC.B ABC.ABC.A = dist ABC.ABC.C ABC.ABC.A
def midpoint (p q : Point) := (p + q) / 2

def cyclic_quadrilateral (P Q R S : Point) :=
  let ⟨o, _, _, _⟩ := exists_circumcircle P Q R S in
  (dist P o = dist Q o ∧ dist Q o = dist R o ∧ dist R o = dist S o)

variables {A B C M X T Y : Point}
variables (h_iso : isosceles ⟨A, B, C, _⟩)
variables (h_mid_M : M = midpoint B C)
variables (X_on_arc_AM : X ∈ smaller_arc (circumcircle A B M))
variables (T_on_side_BM : same_side BM X T)
variables (T_on_opposite_halfplane_AM : opposite_halfplane X AM T)
variables (h_dist_TX_BX : dist T X = dist B X)
variables (h_right_angle_TMX : ∠TMX = π / 2)

-- The proof statement
theorem angle_difference_constant :
  ∠MTB - ∠CTM = ∠BAC / 2 :=
sorry

end angle_difference_constant_l487_487076


namespace area_of_rhombus_roots_eq_sqrt4_2_l487_487529

theorem area_of_rhombus_roots_eq_sqrt4_2 
  (a b c d : ℂ)
  (h_eq : ∀ z : ℂ, z^4 + 4*complex.I*z^3 + (2 + 2*complex.I)*z^2 + (7 + complex.I)*z + (6 - 3*complex.I) = 0 ↔ z = a ∨ z = b ∨ z = c ∨ z = d)
  (h_rhombus : a + b + c + d = -4*complex.I) :
  area_of_rhombus a b c d = complex.sqrt (complex.sqrt 2) :=
sorry

end area_of_rhombus_roots_eq_sqrt4_2_l487_487529


namespace expected_same_as_sixth_die_l487_487119

open ProbabilityTheory

/- Define the probability space -/
noncomputable theory
def die : Probₓ ℕ := uniform Icc 1 6

/- Define the expected_value function -/
def expected_value (n : ℕ) (p : Probₓ ℕ) : ℝ :=
  ∑ i in finₓRange n, (p i * i)

theorem expected_same_as_sixth_die:
  let X : Finₓ 6 → Probₓ ℕ := λ i, if i.val == 5 then uniform Icc 1 6 else dirac (1 / 6),
  expected_value 6 (X 0) + expected_value 6 (X 1) + expected_value 6 (X 2) +
  expected_value 6 (X 3) + expected_value 6 (X 4) + expected_value 6 (X 5) =
  11 / 6 :=
by
  sorry

end expected_same_as_sixth_die_l487_487119


namespace polygon_sides_l487_487007

theorem polygon_sides (x : ℝ) (hx : 0 < x) (h : x + 5 * x = 180) : 12 = 360 / x :=
by {
  -- Steps explaining: x should be the exterior angle then proof follows.
  sorry
}

end polygon_sides_l487_487007


namespace irreducibility_of_P_l487_487284

noncomputable def P (a : Fin n → ℝ) : ℝ :=
  ∑ i, (a i)^n - n * (∏ i, a i)

theorem irreducibility_of_P (n : ℕ) (hn : n ≥ 4) :
  irreducible (P a) := sorry

end irreducibility_of_P_l487_487284


namespace sum_reciprocals_factors_12_l487_487652

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487652


namespace sum_reciprocals_of_factors_of_12_l487_487964

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487964


namespace max_distance_origin_to_line_l487_487338

theorem max_distance_origin_to_line (a b c : ℝ) (h : a - b - c = 0) :
  let line := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0} in
  let origin := (0, 0 : ℝ × ℝ) in
  ∃ p ∈ line, ∀ q ∈ line, dist origin p = dist origin q := 
sorry

end max_distance_origin_to_line_l487_487338


namespace dot_product_eq_negative_29_l487_487380

def vector := ℝ × ℝ

variables (a b : vector)

theorem dot_product_eq_negative_29 
  (h1 : a + b = (2, -4))
  (h2 : 3 * a - b = (-10, 16)) :
  a.1 * b.1 + a.2 * b.2 = -29 :=
sorry

end dot_product_eq_negative_29_l487_487380


namespace sum_reciprocals_factors_12_l487_487740

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487740


namespace maximum_sets_l487_487417

-- define the initial conditions
def dinner_forks : Nat := 6
def knives : Nat := dinner_forks + 9
def soup_spoons : Nat := 2 * knives
def teaspoons : Nat := dinner_forks / 2
def dessert_forks : Nat := teaspoons / 3
def butter_knives : Nat := 2 * dessert_forks

def max_capacity_g : Nat := 20000

def weight_dinner_fork : Nat := 80
def weight_knife : Nat := 100
def weight_soup_spoon : Nat := 85
def weight_teaspoon : Nat := 50
def weight_dessert_fork : Nat := 70
def weight_butter_knife : Nat := 65

-- Calculate the total weight of the existing cutlery
def total_weight_existing : Nat := 
  (dinner_forks * weight_dinner_fork) + 
  (knives * weight_knife) + 
  (soup_spoons * weight_soup_spoon) + 
  (teaspoons * weight_teaspoon) + 
  (dessert_forks * weight_dessert_fork) + 
  (butter_knives * weight_butter_knife)

-- Calculate the weight of one 2-piece cutlery set (1 knife + 1 dinner fork)
def weight_set : Nat := weight_knife + weight_dinner_fork

-- The remaining capacity in the drawer
def remaining_capacity_g : Nat := max_capacity_g - total_weight_existing

-- The maximum number of 2-piece cutlery sets that can be added
def max_2_piece_sets : Nat := remaining_capacity_g / weight_set

-- Theorem: maximum number of 2-piece cutlery sets that can be added is 84
theorem maximum_sets : max_2_piece_sets = 84 :=
by
  sorry

end maximum_sets_l487_487417


namespace max_intersections_l487_487579

-- Define the number of circles and lines
def num_circles : ℕ := 2
def num_lines : ℕ := 3

-- Define the maximum number of intersection points of circles
def max_circle_intersections : ℕ := 2

-- Define the number of intersection points between each line and each circle
def max_line_circle_intersections : ℕ := 2

-- Define the number of intersection points among lines (using the combination formula)
def num_line_intersections : ℕ := (num_lines.choose 2)

-- Define the greatest number of points of intersection
def total_intersections : ℕ :=
  max_circle_intersections + (num_lines * num_circles * max_line_circle_intersections) + num_line_intersections

-- Prove the greatest number of points of intersection is 17
theorem max_intersections : total_intersections = 17 := by
  -- Calculating individual parts for clarity
  have h1: max_circle_intersections = 2 := rfl
  have h2: num_lines * num_circles * max_line_circle_intersections = 12 := by
    calc
      num_lines * num_circles * max_line_circle_intersections
        = 3 * 2 * 2 := by rw [num_lines, num_circles, max_line_circle_intersections]
        ... = 12 := by norm_num
  have h3: num_line_intersections = 3 := by
    calc
      num_line_intersections = (3.choose 2) := rfl
      ... = 3 := by norm_num

  -- Adding the parts to get the total intersections
  calc
    total_intersections
      = max_circle_intersections + (num_lines * num_circles * max_line_circle_intersections) + num_line_intersections := rfl
      ... = 2 + 12 + 3 := by rw [h1, h2, h3]
      ... = 17 := by norm_num

end max_intersections_l487_487579


namespace sum_of_reciprocals_of_factors_of_12_l487_487671

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487671


namespace total_diamonds_F15_l487_487262

-- Defining the sequence according to the conditions
def diamond_sequence (n : ℕ) : ℕ :=
  if n = 1 then 
    1
  else 
    diamond_sequence (n - 1) + 4 * n

-- The main statement we need to prove
theorem total_diamonds_F15 : diamond_sequence 15 = 480 := by
  sorry

end total_diamonds_F15_l487_487262


namespace max_sandwiches_optimal_max_revenue_optimal_min_time_optimal_l487_487520

-- Define the quantities required for each type of sandwich
def cheese_bread := 5
def cheese_butter := 1
def cheese_cheese := 2

def salami_bread := 5
def salami_butter := 5/3
def salami_salami := 1

-- Define available quantities
def available_bread := 200
def available_butter := 50
def available_cheese := 60
def available_salami := 20

-- Define prices
def cheese_price := 1.40
def salami_price := 1.90

-- Definition of maximum sandwiches given resource constraints
def max_sandwiches (x y : ℤ) : Prop :=
  cheese_bread * x + salami_bread * y ≤ available_bread ∧
  cheese_butter * x + salami_butter * y ≤ available_butter ∧
  cheese_cheese * x ≤ available_cheese ∧
  salami_salami * y ≤ available_salami

-- Prove the maximum number of sandwiches and corresponding distribution
theorem max_sandwiches_optimal : ∃ (x y : ℤ), max_sandwiches x y ∧ x + y = 40 ∧ x = 30 ∧ y = 10 :=
by
  sorry

-- Prove the maximum revenue given the prices and optimal distribution
theorem max_revenue_optimal : ∃ (x y : ℤ), max_sandwiches x y ∧ cheese_price * x + salami_price * y = 63.50 ∧ x = 25 ∧ y = 15 :=
by
  sorry

-- Prove the minimal preparation time for the maximum number of sandwiches
theorem min_time_optimal : ∃ (x y : ℤ), max_sandwiches x y ∧ x + 2 * y = 50 ∧ x = 30 ∧ y = 10 :=
by
  sorry

end max_sandwiches_optimal_max_revenue_optimal_min_time_optimal_l487_487520


namespace common_terms_count_common_elements_count_l487_487556

def sequence_1 (n : ℕ) := 2 * n - 1
def sequence_2 (m : ℕ) := 3 * m - 2

theorem common_terms_count :
  (∃ n : ℕ, sequence_1 n = x ∧ ∃ m : ℕ, sequence_2 m = x)
  → x ∈ finset.range 2018
  → x = 6 * k + 1 ∧ k ∈ finset.range 337 := sorry

theorem common_elements_count : 
  finset.univ.filter (λ x, ∃ n : ℕ, sequence_1 n = x)
    ∩ finset.univ.filter (λ x, ∃ m : ℕ, sequence_2 m = x) = 337 := sorry

end common_terms_count_common_elements_count_l487_487556


namespace mul_mod_correct_l487_487176

theorem mul_mod_correct :
  (2984 * 3998) % 1000 = 32 :=
by
  sorry

end mul_mod_correct_l487_487176


namespace max_points_of_intersection_l487_487586

-- Definitions from the conditions
def circles := 2
def lines := 3

-- Define the problem of the greatest intersection number
theorem max_points_of_intersection (c : ℕ) (l : ℕ) (h_c : c = circles) (h_l : l = lines) : 
  (2 + (l * 2 * c) + (l * (l - 1) / 2)) = 17 :=
by
  rw [h_c, h_l]
  -- We have 2 points from circle intersections
  -- 12 points from lines intersections with circles
  -- 3 points from lines intersections with lines
  -- Hence, 2 + 12 + 3 = 17
  exact Eq.refl 17

end max_points_of_intersection_l487_487586


namespace sum_reciprocal_factors_of_12_l487_487598

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487598


namespace sum_reciprocals_factors_12_l487_487614

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487614


namespace how_long_to_grow_more_l487_487104

def current_length : ℕ := 14
def length_to_donate : ℕ := 23
def desired_length_after_donation : ℕ := 12

theorem how_long_to_grow_more : 
  (desired_length_after_donation + length_to_donate - current_length) = 21 := 
by
  -- Leave the proof part for later
  sorry

end how_long_to_grow_more_l487_487104


namespace ratio_proof_l487_487999

noncomputable def area_section (S A : ℝ) (α : ℝ) : ℝ :=
  (1/2) * S * A^2 * sin α

noncomputable def surface_area_cone (A β : ℝ) : ℝ :=
  let R := A * cos β in
  π * R * A + π * R^2

theorem ratio_proof (S A α β : ℝ) (hA : A ≠ 0) (hβ : cos β ≠ 0) :
  (area_section S A α) / (surface_area_cone A β) = (sin α) / (4 * π * cos β * (cos (β / 2))^2) :=
by
  sorry

end ratio_proof_l487_487999


namespace range_of_a_l487_487406

theorem range_of_a (a : ℝ) :
  (a^2 > a + 6) → a ∈ (-6, -2) ∪ (3, +∞) :=
by
  intro h
  sorry

end range_of_a_l487_487406


namespace pentagon_centroid_ratio_l487_487454

theorem pentagon_centroid_ratio (A B C D E : Point) (ABCDEF_Convex : Convex A B C D E)
  (G_A : Point := centroid (triangle B C D E))
  (G_B : Point := centroid (triangle A C D E))
  (G_C : Point := centroid (triangle A B D E))
  (G_D : Point := centroid (triangle A B C E))
  (G_E : Point := centroid (triangle A B C D))
  :
  (\frac{area (pentagon G_A G_B G_C G_D G_E)}{area (pentagon A B C D E)} = \frac{1}{16}) :=
begin
  sorry,
end

end pentagon_centroid_ratio_l487_487454


namespace completing_square_transformation_l487_487982

theorem completing_square_transformation (x : ℝ) :
  x^2 - 2 * x - 5 = 0 -> (x - 1)^2 = 6 :=
by {
  sorry -- Proof to be completed
}

end completing_square_transformation_l487_487982


namespace sum_reciprocals_factors_12_l487_487696

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487696


namespace AC_div_AB_eq_cos_BAH_div_cos_CAH_l487_487251

theorem AC_div_AB_eq_cos_BAH_div_cos_CAH
  {A B C D E H : Type} [RealAffineSpace V P]
  (HA : AffineSubspace P)
  (ABD : ∈ HA) (CE : ∈ HA)
  (HBD : isAltitudeOfTriangle ABD BC D)
  (HCE : isAltitudeOfTriangle ABD CA E)
  (Hinter : intersection (altitude BD) (altitude CE) = some H) :
  (AC ⧸ AB) = (cos BA H ⧸ cos CA H) :=
  by sorry

end AC_div_AB_eq_cos_BAH_div_cos_CAH_l487_487251


namespace sum_reciprocals_factors_12_l487_487645

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487645


namespace S_gt_inverse_1988_cubed_l487_487164

theorem S_gt_inverse_1988_cubed (a b c d : ℕ) (hb: 0 < b) (hd: 0 < d) 
  (h1: a + c < 1988) (h2: 1 - (a / b) - (c / d) > 0) : 
  1 - (a / b) - (c / d) > 1 / (1988^3) := 
sorry

end S_gt_inverse_1988_cubed_l487_487164


namespace intersection_A_B_find_coefficients_a_b_l487_487084

open Set

variable {X : Type} (x : X)

def setA : Set ℝ := { x | x^2 < 9 }
def setB : Set ℝ := { x | (x - 2) * (x + 4) < 0 }
def A_inter_B : Set ℝ := { x | -3 < x ∧ x < 2 }
def A_union_B_solution_set : Set ℝ := { x | -4 < x ∧ x < 3 }

theorem intersection_A_B :
  A ∩ B = { x | -3 < x ∧ x < 2 } :=
sorry

theorem find_coefficients_a_b (a b : ℝ) :
  (∀ x, 2 * x^2 + a * x + b < 0 ↔ -4 < x ∧ x < 3) → 
  a = 2 ∧ b = -24 :=
sorry

end intersection_A_B_find_coefficients_a_b_l487_487084


namespace sum_reciprocals_of_factors_of_12_l487_487973

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487973


namespace minimum_value_of_y_l487_487539

def y (x : ℝ) : ℝ := cos (2 * x) - 6 * cos x + 6

theorem minimum_value_of_y :
  ∃ x : ℝ, y x = 1 :=
by
  sorry

end minimum_value_of_y_l487_487539


namespace sum_reciprocals_factors_12_l487_487650

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487650


namespace sum_of_reciprocals_factors_12_l487_487833

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487833


namespace sum_of_reciprocals_factors_12_l487_487787

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487787


namespace sum_reciprocal_factors_of_12_l487_487602

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487602


namespace value_of_40th_expression_l487_487156

-- Define the sequence
def minuend (n : ℕ) : ℕ := 100 - (n - 1)
def subtrahend (n : ℕ) : ℕ := n
def expression_value (n : ℕ) : ℕ := minuend n - subtrahend n

-- Theorem: The value of the 40th expression in the sequence is 21
theorem value_of_40th_expression : expression_value 40 = 21 := by
  show 100 - (40 - 1) - 40 = 21
  sorry

end value_of_40th_expression_l487_487156


namespace range_of_a_l487_487016

theorem range_of_a (x a : ℝ) (h₀ : x < 0) (h₁ : 2^x - a = 1 / (x - 1)) : 0 < a ∧ a < 2 :=
sorry

end range_of_a_l487_487016


namespace find_vector_b_l487_487071

/-- Let a and c be given vectors. Define the vector b such that a, b, 
and c are collinear, and b bisects the angle between a and c. -/
theorem find_vector_b :
  let a := (7, -4, -4 : ℝ × ℝ × ℝ)
  let c := (-2, -1, 2 : ℝ × ℝ × ℝ)
  ∃ b : ℝ × ℝ × ℝ,
    (∃ t : ℝ, b = ⟨7 - 9 * t, -4 + 3 * t, -4 + 6 * t⟩) ∧
    (cos_angle a b = cos_angle b c) ∧
    b = (1/4, -7/4, 1/2 : ℝ × ℝ × ℝ) :=
begin
  sorry
end

end find_vector_b_l487_487071


namespace imaginary_part_of_complex_expr_l487_487533

def complex_expr := 4 * complex.I / (1 - complex.I)
def result := complex.im complex_expr

theorem imaginary_part_of_complex_expr :
  result = 2 :=
by
  sorry

end imaginary_part_of_complex_expr_l487_487533


namespace projection_matrix_correct_l487_487064

-- Define the normal vector and the projection matrix.
def normal_vector : ℝ × ℝ × ℝ := (2, -1, 2)
def Q_matrix : ℝ → ℝ → ℝ → ℝ × ℝ × ℝ := 
  λ x y z, 
    ( (5 * x + 2 * y - 4 * z) / 9, 
      (2 * x + 10 * y - 2 * z) / 9, 
      (-4 * x + 2 * y + 5 * z) / 9 )

-- Theorem to prove the matrix Q correctly projects any vector onto the plane.
theorem projection_matrix_correct (v : ℝ × ℝ × ℝ) : 
    let ⟨x, y, z⟩ := v in 
    Q_matrix x y z = ((v.1) * (5/9) + (v.2) * (2/9) - (v.3) * (4/9), 
                      (v.1) * (2/9) + (v.2) * (10/9) - (v.3) * (2/9), 
                      (v.1) * (-4/9) + (v.2) * (2/9) + (v.3) * (5/9)) := 
by sorry

end projection_matrix_correct_l487_487064


namespace exchange_yen_for_yuan_l487_487038

-- Define the condition: 100 Japanese yen could be exchanged for 7.2 yuan
def exchange_rate : ℝ := 7.2
def yen_per_100_yuan : ℝ := 100

-- Define the amount in yuan we want to exchange
def yuan_amount : ℝ := 720

-- The mathematical assertion (proof problem)
theorem exchange_yen_for_yuan : 
  (yuan_amount / exchange_rate) * yen_per_100_yuan = 10000 :=
by
  sorry

end exchange_yen_for_yuan_l487_487038


namespace sum_of_reciprocals_factors_12_l487_487824

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487824


namespace sum_reciprocals_of_factors_of_12_l487_487958

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487958


namespace sum_of_reciprocals_factors_12_l487_487724

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487724


namespace simple_and_compound_interest_difference_l487_487181

theorem simple_and_compound_interest_difference (r : ℝ) :
  let P := 3600
  let t := 2
  let SI := P * r * t / 100
  let CI := P * (1 + r / 100)^t - P
  CI - SI = 225 → r = 25 := by
  intros
  sorry

end simple_and_compound_interest_difference_l487_487181


namespace Sue_made_22_buttons_l487_487446

def Mari_buttons : Nat := 8
def Kendra_buttons : Nat := 5 * Mari_buttons + 4
def Sue_buttons : Nat := Kendra_buttons / 2

theorem Sue_made_22_buttons : Sue_buttons = 22 :=
by
  -- proof to be added
  sorry

end Sue_made_22_buttons_l487_487446


namespace coeff_x4_term_in_expansion_l487_487521

open Nat

def binomial_expansion_coeff (n k : ℕ) (a b c : ℝ) : ℝ :=
  ∑ i in range (n + 1), choose n i * a ^ i * b ^ (n - i) * choose (n - i) k * c ^ ((n - i) - k)

theorem coeff_x4_term_in_expansion :
  binomial_expansion_coeff 6 4 (1 : ℤ) ((1 : ℤ)/(1 : ℤ)) (-1 : ℤ) = 21 :=
by
  sorry

end coeff_x4_term_in_expansion_l487_487521


namespace sum_reciprocal_factors_of_12_l487_487600

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487600


namespace scheduled_arrangements_correct_l487_487997

def total_scheduling_arrangements : Nat :=
  -- Given conditions
  let doctors : Finset Nat := {1, 2, 3, 4, 5, 6}
  let days : Finset Nat := {1, 2, 3}
  let pairs := (doctors.product doctors).filter (fun p => p.fst ≠ p.snd)
  
  let valid_schedules : Finset (Finset (Nat × Nat)) :=
    pairs.powerset.filter (fun s =>
      s.card = 3 ∧
      s.any (fun (d, a) => (a, d) ∈ pairs ∧ a = 1 ∧ d = 2) ∧ -- A not on 2nd day
      s.any (fun (d, b) => (b, d) ∈ pairs ∧ b = 2 ∧ d = 3) -- B not on 3rd day)
  
  valid_schedules.card

theorem scheduled_arrangements_correct : total_scheduling_arrangements = 42 := by
  sorry

end scheduled_arrangements_correct_l487_487997


namespace sum_reciprocal_factors_12_l487_487873

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487873


namespace max_intersections_l487_487581

-- Define the number of circles and lines
def num_circles : ℕ := 2
def num_lines : ℕ := 3

-- Define the maximum number of intersection points of circles
def max_circle_intersections : ℕ := 2

-- Define the number of intersection points between each line and each circle
def max_line_circle_intersections : ℕ := 2

-- Define the number of intersection points among lines (using the combination formula)
def num_line_intersections : ℕ := (num_lines.choose 2)

-- Define the greatest number of points of intersection
def total_intersections : ℕ :=
  max_circle_intersections + (num_lines * num_circles * max_line_circle_intersections) + num_line_intersections

-- Prove the greatest number of points of intersection is 17
theorem max_intersections : total_intersections = 17 := by
  -- Calculating individual parts for clarity
  have h1: max_circle_intersections = 2 := rfl
  have h2: num_lines * num_circles * max_line_circle_intersections = 12 := by
    calc
      num_lines * num_circles * max_line_circle_intersections
        = 3 * 2 * 2 := by rw [num_lines, num_circles, max_line_circle_intersections]
        ... = 12 := by norm_num
  have h3: num_line_intersections = 3 := by
    calc
      num_line_intersections = (3.choose 2) := rfl
      ... = 3 := by norm_num

  -- Adding the parts to get the total intersections
  calc
    total_intersections
      = max_circle_intersections + (num_lines * num_circles * max_line_circle_intersections) + num_line_intersections := rfl
      ... = 2 + 12 + 3 := by rw [h1, h2, h3]
      ... = 17 := by norm_num

end max_intersections_l487_487581


namespace sum_reciprocals_factors_12_l487_487713

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487713


namespace sum_reciprocal_factors_of_12_l487_487595

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487595


namespace even_three_digit_desc_order_count_l487_487387

/-
Problem: Prove that the number of even three-digit integers with digits in strictly decreasing order is 4.
-/

def is_digit (d : ℕ) : Prop := d ∈ {2, 4, 6, 8}

/-- Define a three-digit even integer with digits in strictly decreasing order. -/
def even_three_digit_desc_order_digits (a b c : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ a > b ∧ b > c

theorem even_three_digit_desc_order_count :
  ∃ n : ℕ, n = 4 ∧ (∀ a b c, even_three_digit_desc_order_digits a b c → true) := sorry

end even_three_digit_desc_order_count_l487_487387


namespace find_a_l487_487073

theorem find_a (a : ℤ) (h_range : 0 ≤ a ∧ a < 13) (h_div : (51 ^ 2022 + a) % 13 = 0) : a = 12 := 
by
  sorry

end find_a_l487_487073


namespace sum_reciprocals_factors_12_l487_487636

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487636


namespace quartic_polynomial_eval_l487_487403

noncomputable def f (x : ℝ) : ℝ := sorry  -- f is a monic quartic polynomial

theorem quartic_polynomial_eval (h_monic: true)
    (h1 : f (-1) = -1)
    (h2 : f 2 = -4)
    (h3 : f (-3) = -9)
    (h4 : f 4 = -16) : f 1 = 23 :=
sorry

end quartic_polynomial_eval_l487_487403


namespace sum_reciprocals_12_l487_487925

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487925


namespace pentagon_card_arrangement_l487_487274

def cards : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

noncomputable def sides (arrangement : List Nat) : List (Nat × Nat × Nat) :=
[[arrangement[0], arrangement[1], arrangement[4]],
 [arrangement[1], arrangement[2], arrangement[3]],
 [arrangement[2], arrangement[3], arrangement[4]],
 [arrangement[3], arrangement[4], arrangement[0]],
 [arrangement[4], arrangement[0], arrangement[1]]]

noncomputable def valid_arrangement (arrangement : List Nat) : Prop :=
  let side_sums := sides arrangement |>.map (λ (triplet : Nat × Nat × Nat) => triplet.fst + triplet.snd + triplet.trd)
  side_sums.all_eq (side_sums.head!)

noncomputable def total_solutions : Nat :=
  -- Placeholder for the number of unique valid arrangements
  sorry

theorem pentagon_card_arrangement : total_solutions = 6 := by
  sorry

end pentagon_card_arrangement_l487_487274


namespace find_m_eq_neg11_l487_487035

theorem find_m_eq_neg11 (m : ℝ) 
  (h_circle : ∀ (x y : ℝ), x^2 + y^2 + 8 * x - m + 1 = 0 → (x + 4)^2 + y^2 = m + 15) 
  (h_line : ∀ (x y : ℝ), x + sqrt 2 * y + 1 = 0) 
  (h_equilateral : ∀ (A B C : ℝ × ℝ), ∃ (A B : ℝ × ℝ), 
    A ≠ B ∧ 
    (A.1 ^ 2 + A.2 ^ 2 + 8 * A.1 - m + 1 = 0) ∧ 
    (B.1 ^ 2 + B.2 ^ 2 + 8 * B.1 - m + 1 = 0) ∧ 
    (A.1 + sqrt 2 * A.2 + 1 = 0) ∧ 
    (B.1 + sqrt 2 * B.2 + 1 = 0) ∧ 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 * m + 60)) 
  : m = -11 := 
sorry

end find_m_eq_neg11_l487_487035


namespace sum_reciprocals_12_l487_487932

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487932


namespace sum_reciprocals_factors_12_l487_487947

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487947


namespace dot_product_eq_neg29_l487_487378

-- Given definitions and conditions
variables (a b : ℝ × ℝ)

-- Theorem to prove the dot product condition.
theorem dot_product_eq_neg29 (h1 : a + b = (2, -4)) (h2 : 3 • a - b = (-10, 16)) :
  a.1 * b.1 + a.2 * b.2 = -29 :=
sorry

end dot_product_eq_neg29_l487_487378


namespace largest_in_set_when_a_is_neg3_l487_487312

-- Define the set expressions and state the problem
def expr_set (a : ℝ) : set ℝ := {-4 * a, 3 * a, 27 / a, a^3, 0}

theorem largest_in_set_when_a_is_neg3 : 
  let a := -3 in
  let s := expr_set a in
  (∃ (x : ℝ), x ∈ s ∧ x = 12) := 
begin
  -- This is where the proof would go.
  sorry
end

end largest_in_set_when_a_is_neg3_l487_487312


namespace sum_reciprocals_of_factors_of_12_l487_487955

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487955


namespace tangent_line_b_value_l487_487260

theorem tangent_line_b_value (C1 C2 : ℂ) (r1 r2 : ℝ) (hC1 : C1 = Complex.mk 1 3) (hC2 : C2 = Complex.mk 15 8) (hr1 : r1 = 3) (hr2 : r2 = 10) (m : ℝ) (hm : m > 0) :
  ∃ b : ℝ, tangent_line C1 r1 C2 r2 m b ∧ b = 148 / 19 :=
sorry

end tangent_line_b_value_l487_487260


namespace sum_reciprocals_factors_12_l487_487939

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487939


namespace sum_of_reciprocals_factors_12_l487_487714

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487714


namespace slope_of_line_l487_487294

def line_equation (x y : ℝ) : Prop := x / 4 + y / 3 = 1

theorem slope_of_line : ∀ (x y : ℝ), line_equation x y → (∃ m : ℝ, m = -3 / 4) :=
begin
  sorry
end

end slope_of_line_l487_487294


namespace f_bx_le_f_cx_l487_487313

theorem f_bx_le_f_cx
  (f := λ x : ℝ, x^2 - (1 : ℝ) * x + 3)
  (h₁ : f 0 = 3)
  (h₂ : ∀ x, f (1 + x) = f (1 - x)) :
  ∀ x, f (1^x) ≤ f (3^x) :=
by
  sorry

end f_bx_le_f_cx_l487_487313


namespace sum_reciprocals_of_factors_of_12_l487_487959

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487959


namespace percent_of_x_eq_21_percent_l487_487976

theorem percent_of_x_eq_21_percent (x : Real) : (0.21 * x = 0.30 * 0.70 * x) := by
  sorry

end percent_of_x_eq_21_percent_l487_487976


namespace length_inequality_l487_487302

noncomputable def l_a (A B C : ℝ) : ℝ := 
  sorry -- Definition according to the mathematical problem

noncomputable def l_b (A B C : ℝ) : ℝ := 
  sorry -- Definition according to the mathematical problem

noncomputable def l_c (A B C : ℝ) : ℝ := 
  sorry -- Definition according to the mathematical problem

noncomputable def perimeter (A B C : ℝ) : ℝ :=
  A + B + C

theorem length_inequality (A B C : ℝ) (hA : A > 0) (hB : B > 0) (hC : C > 0) :
  (l_a A B C * l_b A B C * l_c A B C) / (perimeter A B C)^3 ≤ 1 / 64 :=
by
  sorry

end length_inequality_l487_487302


namespace find_number_l487_487177

theorem find_number (x : ℝ) (h : x / 2 = x - 5) : x = 10 :=
by
  sorry

end find_number_l487_487177


namespace sum_of_reciprocals_factors_12_l487_487774

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487774


namespace isosceles_triangle_length_l487_487431

theorem isosceles_triangle_length (BC : ℕ) (area : ℕ) (h : ℕ)
  (isosceles : AB = AC)
  (BC_val : BC = 16)
  (area_val : area = 120)
  (height_val : h = (2 * area) / BC)
  (AB_square : ∀ BD AD : ℕ, BD = BC / 2 → AD = h → AB^2 = AD^2 + BD^2)
  : AB = 17 :=
by
  sorry

end isosceles_triangle_length_l487_487431


namespace boy_actual_height_is_236_l487_487517

def actual_height (n : ℕ) (incorrect_avg correct_avg wrong_height : ℕ) : ℕ :=
  let incorrect_total := n * incorrect_avg
  let correct_total := n * correct_avg
  let diff := incorrect_total - correct_total
  wrong_height + diff

theorem boy_actual_height_is_236 :
  ∀ (n incorrect_avg correct_avg wrong_height actual_height : ℕ),
  n = 35 → 
  incorrect_avg = 183 → 
  correct_avg = 181 → 
  wrong_height = 166 → 
  actual_height = wrong_height + (n * incorrect_avg - n * correct_avg) →
  actual_height = 236 :=
by
  intros n incorrect_avg correct_avg wrong_height actual_height hn hic hg hw ha
  rw [hn, hic, hg, hw] at ha
  -- At this point, we would normally proceed to prove the statement.
  -- However, as per the requirements, we just include "sorry" to skip the proof.
  sorry

end boy_actual_height_is_236_l487_487517


namespace max_points_of_intersection_l487_487585

-- Definitions from the conditions
def circles := 2
def lines := 3

-- Define the problem of the greatest intersection number
theorem max_points_of_intersection (c : ℕ) (l : ℕ) (h_c : c = circles) (h_l : l = lines) : 
  (2 + (l * 2 * c) + (l * (l - 1) / 2)) = 17 :=
by
  rw [h_c, h_l]
  -- We have 2 points from circle intersections
  -- 12 points from lines intersections with circles
  -- 3 points from lines intersections with lines
  -- Hence, 2 + 12 + 3 = 17
  exact Eq.refl 17

end max_points_of_intersection_l487_487585


namespace javier_daughter_cookie_count_l487_487442

theorem javier_daughter_cookie_count : 
  ∀ (total_cookies wife_share_percent eaten_cookies remaining_cookies : ℕ) (D : ℕ),
  total_cookies = 200 →
  wife_share_percent = 30 →
  eaten_cookies = 50 →
  remaining_cookies = total_cookies - (total_cookies * wife_share_percent / 100) - D →
  (remaining_cookies / 2) = eaten_cookies →
  D = 40 := 
by
  intros total_cookies wife_share_percent eaten_cookies remaining_cookies D
  assume h1 h2 h3 h4 h5
  have h6 : total_cookies - (total_cookies * wife_share_percent / 100) = 140, from sorry
  have h7 : 140 - D = 100, by linarith
  show D = 40, by linarith

end javier_daughter_cookie_count_l487_487442


namespace find_angle_l487_487295

theorem find_angle :
  ∃ θ : ℝ, θ > 0 ∧ θ ≤ 360 ∧ cos θ = cos 75 :=
sorry

end find_angle_l487_487295


namespace journey_time_l487_487157

variables (At : ℝ) (Br : ℝ) (Cr : ℝ) (d : ℝ) (t : ℝ)
variables (x : ℝ) (y : ℝ) (d1 : ℝ) (d2 : ℝ)

-- Conditions
def speeds := (At = 1) ∧ (Br = 2) ∧ (Cr = 8)
def distances := (d = 40)
def cart_travel :=
  t = x + d1 / Cr ∧
  t = y / Br + d2 / Cr ∧
  t = (d1 + 2 * (d - x - d2) + d2) / Cr
def simultaneous_arrival := speeds ∧ distances ∧ cart_travel

-- Goal
theorem journey_time (h : simultaneous_arrival):
  t = 10 + 5 / 41 :=
sorry  -- Proof is omitted

end journey_time_l487_487157


namespace distribute_computers_l487_487558

theorem distribute_computers : 
  let distribute_ways := (∑ (a b c : ℕ) in { 
    (2, 3, 4), 
    (2, 2, 5), 
    (3, 3, 3) }, if a + b + c = 9 then 1 else 0)
  in distribute_ways = 10 :=
by sorry

end distribute_computers_l487_487558


namespace total_sales_first_three_days_total_earnings_seven_days_l487_487194

def planned_daily_sales : Int := 100

def deviation : List Int := [4, -3, -5, 14, -8, 21, -6]

def selling_price_per_pound : Int := 8
def freight_cost_per_pound : Int := 3

-- Part (1): Proof statement for the total amount sold in the first three days
theorem total_sales_first_three_days :
  let monday_sales := planned_daily_sales + deviation.head!
  let tuesday_sales := planned_daily_sales + (deviation.drop 1).head!
  let wednesday_sales := planned_daily_sales + (deviation.drop 2).head!
  monday_sales + tuesday_sales + wednesday_sales = 296 := by
  sorry

-- Part (2): Proof statement for Xiaoming's total earnings for the seven days
theorem total_earnings_seven_days :
  let total_sales := (List.sum (deviation.map (λ x => planned_daily_sales + x)))
  total_sales * (selling_price_per_pound - freight_cost_per_pound) = 3585 := by
  sorry

end total_sales_first_three_days_total_earnings_seven_days_l487_487194


namespace sum_reciprocals_factors_12_l487_487748

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487748


namespace garden_area_l487_487027

noncomputable theory

-- Define the conditions
def AX : ℝ := 5
def BX : ℝ := 12
def AB : ℝ := 13

-- Define a theorem to prove the possible areas of the garden
theorem garden_area (X_on_diagonal : ∀ {ax bx ab : ℝ}, ax = AX → bx = BX → ab = AB → ax^2 + bx^2 = ab^2) :
  ∃ (A1 A2 : ℝ), A1 = 405.6 ∨ A2 = 70.42 :=
begin
  sorry
end

end garden_area_l487_487027


namespace find_dividend_and_divisor_l487_487161

theorem find_dividend_and_divisor (quotient : ℕ) (remainder : ℕ) (total : ℕ) (dividend divisor : ℕ) :
  quotient = 13 ∧ remainder = 6 ∧ total = 137 ∧ (dividend + divisor + quotient + remainder = total)
  ∧ dividend = 13 * divisor + remainder → 
  dividend = 110 ∧ divisor = 8 :=
by
  intro h
  sorry

end find_dividend_and_divisor_l487_487161


namespace sum_reciprocals_factors_of_12_l487_487691

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487691


namespace sum_of_reciprocals_of_factors_of_12_l487_487805

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487805


namespace sum_reciprocal_factors_12_l487_487863

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487863


namespace sum_of_reciprocals_factors_12_l487_487732

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487732


namespace sum_reciprocals_of_factors_12_l487_487850

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487850


namespace sum_reciprocal_factors_of_12_l487_487611

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487611


namespace similar_triangles_l487_487421

variable {A B C K L : Type} [triangle : Triangle A B C] [height1 : Height A K B C] [height2 : Height B L A C]

theorem similar_triangles 
(triangle_acute : AcuteTriangle A B C)
(height_AK : HeightFromVertexToOppositeSide A K B C)
(height_BL : HeightFromVertexToOppositeSide B L A C) :
  Similar (Triangle A K C) (Triangle B L C) :=
sorry

end similar_triangles_l487_487421


namespace correct_option_D_l487_487189

theorem correct_option_D : 
  (∀ (x : ℝ), x^2 + x^4 ≠ x^6) ∧
  ( ∃ (a b : ℝ), a = -2 ∧ b = 4 ∧ sqrt(b) ≠ a) ∧
  ( ∃ (c : ℝ), c = 16 ∧ sqrt(c) ≠ -4 ∧ sqrt(c) ≠ 4) ∧
  ( 3 * sqrt(3) - 2 * sqrt(3) = sqrt(3)) :=
by 
  sorry

end correct_option_D_l487_487189


namespace cos_coefficients_l487_487486

theorem cos_coefficients :
  ∃ m n p : ℤ,
  (cos 2 * α = 2 * cos α ^ 2 - 1) ∧
  (cos 4 * α = 8 * cos α ^ 4 - 8 * cos α ^ 2 + 1) ∧
  (cos 6 * α = 32 * cos α ^ 6 - 48 * cos α ^ 4 + 18 * cos α ^ 2 - 1) ∧
  (cos 8 * α = 128 * cos α ^ 8 - 256 * cos α ^ 6 + 160 * cos α ^ 4 - 32 * cos α ^ 2 + 1) ∧
  (cos 10 * α = m * cos α ^ 10 - 1280 * cos α ^ 8 + 1120 * cos α ^ 6 + n * cos α ^ 4 + p * cos α ^ 2 - 1) ∧
  (m - n + p = -2048) :=
sorry

end cos_coefficients_l487_487486


namespace model_tower_height_is_correct_l487_487420

def original_tower_height : ℝ := 50
def original_roof_radius : ℝ := 10
def original_roof_height : ℝ := 5
def original_roof_volume : ℝ := 523.6
def model_roof_volume : ℝ := 1

noncomputable def volume_ratio := original_roof_volume / model_roof_volume
noncomputable def scale_factor := real.cbrt(volume_ratio)
def original_tower_height_excluding_roof := original_tower_height - original_roof_height
noncomputable def model_tower_height_excluding_roof := original_tower_height_excluding_roof / scale_factor

theorem model_tower_height_is_correct :
  model_tower_height_excluding_roof ≈ 5.58 :=
sorry

end model_tower_height_is_correct_l487_487420


namespace profit_function_maximum_profit_l487_487512

def P (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 9 then
    1/2 * x^2 + 2 * x
  else if x ≥ 9 then
    11 * x + 100 / x - 53
  else
    0

def L (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 9 then
    10 * x - P x - 3
  else if x ≥ 9 then
    10 * x - P x - 3
  else
    0

theorem profit_function :
  (∀ x, (0 < x ∧ x < 9 → L x = -1/2 * x^2 + 8 * x - 3)) ∧
  (∀ x, (x ≥ 9 → L x = -x - 100 / x + 50)) :=
by
  intro x,
  simp [L, P],
  split,
  { intro hx,
    rw [if_pos hx, if_pos hx],
    sorry },
  { intro hx,
    rw [if_pos _ hx],
    sorry }

theorem maximum_profit :
  ∃ x ∈ Icc 0 10, ∀ y, L y ≤ L 10 ∧ L 10 = 30 :=
by
  use 10,
  split,
  { exact ⟨by norm_num, by norm_num⟩ },
  { intro y,
    sorry }

end profit_function_maximum_profit_l487_487512


namespace find_f_1000_l487_487074

noncomputable theory
open_locale classical

def f : ℕ+ → ℤ := sorry

axiom f_mul (x y : ℕ+) : f (x * y) = f x + f y
axiom f_10 : f 10 = 16
axiom f_40 : f 40 = 26
axiom f_8 : f 8 = 12

theorem find_f_1000 : f 1000 = 48 :=
by 
  -- Just indicating that proof should be provided here
  sorry

end find_f_1000_l487_487074


namespace sum_reciprocal_factors_of_12_l487_487603

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487603


namespace airplane_distance_difference_l487_487404

theorem airplane_distance_difference (a : ℕ) : 
  let against_wind_distance := (a - 20) * 3
  let with_wind_distance := (a + 20) * 4
  with_wind_distance - against_wind_distance = a + 140 :=
by
  sorry

end airplane_distance_difference_l487_487404


namespace solution_set_correct_l487_487547

theorem solution_set_correct (a b : ℝ) :
  (∀ x : ℝ, - 1 / 2 < x ∧ x < 1 / 3 → ax^2 + bx + 2 > 0) →
  (a - b = -10) :=
by
  sorry

end solution_set_correct_l487_487547


namespace divides_a_square_minus_a_and_a_cube_minus_a_l487_487465

theorem divides_a_square_minus_a_and_a_cube_minus_a (a : ℤ) : 
  (2 ∣ a^2 - a) ∧ (3 ∣ a^3 - a) :=
by
  sorry

end divides_a_square_minus_a_and_a_cube_minus_a_l487_487465


namespace sum_of_reciprocals_factors_12_l487_487782

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487782


namespace sum_reciprocals_factors_12_l487_487887

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487887


namespace systematic_sampling_sequence_l487_487271

theorem systematic_sampling_sequence :
  ∃ (s : Set ℕ), s = {3, 13, 23, 33, 43} ∧
  (∀ n, n ∈ s → n ≤ 50 ∧ ∃ k, k < 5 ∧ n = 3 + k * 10) :=
by
  sorry

end systematic_sampling_sequence_l487_487271


namespace polynomial_inequality_l487_487496

theorem polynomial_inequality (x : ℝ) (n : ℕ) (f : ℝ → ℝ) 
(hf_def: ∀ g : ℝ → ℝ, (∀ k : ℕ, g k = x ^ (2 * k) + if even k then x ^ (2 * k - 1) else -x ^ (2 * k - 1)) → f x = ∑ k in range n, g k + 1) 
(hx_nonneg: 0 ≤ x) : 
  f x > 1 / 2 := sorry

end polynomial_inequality_l487_487496


namespace problem1_problem2_l487_487326
open Real

-- Definition of the sequence {x_n}
def sequence (t : ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 1 else
  have ih : (∀ k < n, 0 ≤ sequence t k ∧ sequence t (k + 1) = (t + (sequence t k)^2) / 8) :=
    sorry,
  (t + (sequence t (n-1))^2) / 8

-- Proof Problem 1
theorem problem1 (t : ℝ) (n : ℕ) (h1 : 7 < t) (h2 : t ≤ 12) : 1 ≤ sequence t n ∧ sequence t n < sequence t (n+1) ∧ sequence t (n+1) < 2 :=
  sorry

-- Proof Problem 2
theorem problem2 (t : ℝ) (h : ∀ n : ℕ, sequence t n < 4) : t ≤ 16 :=
  sorry

end problem1_problem2_l487_487326


namespace goats_at_farm_l487_487555

theorem goats_at_farm (G C D P : ℕ) 
  (h1: C = 2 * G)
  (h2: D = (G + C) / 2)
  (h3: P = D / 3)
  (h4: G = P + 33) :
  G = 66 :=
by
  sorry

end goats_at_farm_l487_487555


namespace weight_of_one_bowling_ball_l487_487485

-- Define the weights of one bowling ball and one canoe
variables (b c : ℝ)

-- Given conditions as definitions in Lean 4
def condition1 : Prop := 9 * b = 6 * c
def condition2 : Prop := 5 * c = 120

-- The statement to prove
theorem weight_of_one_bowling_ball (h1 : condition1) (h2 : condition2) : b = 16 :=
begin
  sorry
end

end weight_of_one_bowling_ball_l487_487485


namespace sum_reciprocals_factors_of_12_l487_487678

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487678


namespace sum_reciprocals_factors_12_l487_487753

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487753


namespace single_winner_defeats_all_l487_487419

-- Definition: Player received an award if for any other player B,
-- the player either defeated B, or defeated someone who defeated B.

structure Tournament (Player : Type) :=
(matches : Player → Player → Prop) -- matches played between players
(defeated : Player → Player → Prop) -- player A defeated player B
(award : Player → Prop) -- player received an award

variables {Player : Type} (T : Tournament Player)

-- Condition: Each pair of competitors played exactly one match against each other.
axiom match_unique : ∀ (A B : Player), T.matches A B ↔ A ≠ B

-- Condition: Player A received an award if for any other participant B,
-- A either defeated B, or A defeated someone who defeated B.
axiom award_condition : ∀ (A B : Player), T.award A ↔ (∀ (B : Player), T.defeated A B ∨ ∃ C, T.defeated A C ∧ T.defeated C B)

-- Theorem: If exactly one player received an award, then that player defeated everyone else.
theorem single_winner_defeats_all :
  (∃! A : Player, T.award A) → 
  ∀ A B : Player, T.award A → A ≠ B → T.defeated A B := 
sorry

end single_winner_defeats_all_l487_487419


namespace angle_order_cosine_condition_l487_487017

theorem angle_order_cosine_condition (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A < B) (h3 : B < C) :
  ¬(A < B < C ↔ cos (2 * A) > cos (2 * B) > cos (2 * C)) ∧ (cos (2 * A) > cos (2 * B) > cos (2 * C) → A < B < C) :=
sorry

end angle_order_cosine_condition_l487_487017


namespace sum_of_reciprocals_of_factors_of_12_l487_487759

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487759


namespace complete_the_square_l487_487184

theorem complete_the_square (x : ℝ) : 
  x^2 - 2 * x - 5 = 0 ↔ (x - 1)^2 = 6 := 
by {
  -- This is where you would provide the proof
  sorry
}

end complete_the_square_l487_487184


namespace sufficient_condition_for_parallel_lines_l487_487459

-- Define the condition for lines to be parallel
def lines_parallel (a b c d e f : ℝ) : Prop :=
(∃ k : ℝ, a = k * c ∧ b = k * d)

-- Define the specific lines given in the problem
def line1 (a : ℝ) (x y : ℝ) : ℝ := a * x + y - 1
def line2 (a : ℝ) (x y : ℝ) : ℝ := x + a * y + 5

theorem sufficient_condition_for_parallel_lines (a : ℝ) :
  (lines_parallel (a) (1) (-1) (1) (-1) (1 + 5)) ↔ (a = -1) :=
sorry

end sufficient_condition_for_parallel_lines_l487_487459


namespace probability_asian_country_l487_487154

theorem probability_asian_country : 
  let cards := ["China", "USA", "UK", "South Korea"]
  let asian_countries := ["China", "South Korea"]
  let total_cards := 4
  let favorable_outcomes := 2
  (favorable_outcomes / total_cards : ℚ) = 1 / 2 := 
by 
  sorry

end probability_asian_country_l487_487154


namespace angle_of_inclination_l487_487290

noncomputable def y (x : ℝ) : ℝ := x^3 + x - 2

def point : ℝ × ℝ := (0, -2)

theorem angle_of_inclination : 
  let l := y; 
  ∃ α : ℝ, tan α = 1 ∧ α = Real.pi / 4 := 
  sorry

end angle_of_inclination_l487_487290


namespace max_moves_lt_n_cubed_over_4_l487_487056

universe u

section

variable {α : Type u} 

structure Config :=
  (points : set α)
  (segments : set (α × α))
  (no_collinear : ∀ (a b c : α), a ∈ points → b ∈ points → c ∈ points → (a, b) ∈ segments → (b, c) ∈ segments → (a, c) ∉ segments)
  (cycle : ∀ (a : α), a ∈ points → (∃! (b c : α), (a, b) ∈ segments ∧ (a, c) ∈ segments))

theorem max_moves_lt_n_cubed_over_4 {n : ℕ} (h : n ≥ 4) (cfg : Config) 
  (h_card : cfg.points.card = n) :
  ∃ (N : ℕ), N < n^3 / 4 ∧ 
  ∀ (moves : list (α × α × α × α)), (∀ (m : α × α × α × α), m ∈ moves → 
    let (a, b, c, d) := m in (a, b) ∈ cfg.segments ∧ (c, d) ∈ cfg.segments ∧ 
      shared_point (a, b) (c, d) ∧ no_overlap (a, c) (b, d)) → moves.length = N := 
sorry

end

end max_moves_lt_n_cubed_over_4_l487_487056


namespace complete_the_square_l487_487183

theorem complete_the_square (x : ℝ) : 
  x^2 - 2 * x - 5 = 0 ↔ (x - 1)^2 = 6 := 
by {
  -- This is where you would provide the proof
  sorry
}

end complete_the_square_l487_487183


namespace sum_reciprocals_factors_12_l487_487734

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487734


namespace min_difference_of_factorization_l487_487540

theorem min_difference_of_factorization :
  ∃ a1 a2 b1 b2 : ℕ, 
    0 < a1 ∧ 0 < a2 ∧ 0 < b1 ∧ 0 < b2 ∧
    a1 ≥ a2 ∧ b1 ≥ b2 ∧ 
    1729 = a1! * a2! / (b1! * b2!) ∧ 
    (a1 + a2 + b1 + b2 = 63) ∧ 
    (abs (a1 - b1) = 1) :=
begin
  sorry
end

end min_difference_of_factorization_l487_487540


namespace sum_reciprocals_12_l487_487926

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487926


namespace max_points_of_intersection_l487_487584

-- Definitions from the conditions
def circles := 2
def lines := 3

-- Define the problem of the greatest intersection number
theorem max_points_of_intersection (c : ℕ) (l : ℕ) (h_c : c = circles) (h_l : l = lines) : 
  (2 + (l * 2 * c) + (l * (l - 1) / 2)) = 17 :=
by
  rw [h_c, h_l]
  -- We have 2 points from circle intersections
  -- 12 points from lines intersections with circles
  -- 3 points from lines intersections with lines
  -- Hence, 2 + 12 + 3 = 17
  exact Eq.refl 17

end max_points_of_intersection_l487_487584


namespace sum_of_reciprocals_factors_12_l487_487785

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487785


namespace sum_reciprocal_factors_12_l487_487871

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487871


namespace sum_reciprocal_factors_12_l487_487868

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487868


namespace parabola_equation_and_triangle_area_l487_487371

theorem parabola_equation_and_triangle_area :
  (∃ p : ℝ, (∀ x y : ℝ, y^2 = 2*p*x ↔ x = 8 → y = -8) ∧ y^2 = 8*x) ∧
  (∀ (A B P : ℝ) (l₂ : ℝ → Prop), 
   (∃ x m : ℝ, l₂ x ↔ x = y + m ∧ m = 8) →
   |-abs (|A - O| - |P - B|) = 0 
   → area_triangle F A B = 24 * sqrt 5) := sorry

end parabola_equation_and_triangle_area_l487_487371


namespace inverse_of_parallel_lines_l487_487191

theorem inverse_of_parallel_lines 
  (P Q : Prop) 
  (parallel_impl_alt_angles : P → Q) :
  (Q → P) := 
by
  sorry

end inverse_of_parallel_lines_l487_487191


namespace solve_inequality_l487_487123

noncomputable def inequality_condition_1 (x : ℝ) : Prop := sin (π * x) > 0

noncomputable def inequality_condition_2 (x : ℝ) : Prop := sin (π * x) ≠ 1

noncomputable def inequality_condition_3 (x : ℝ) : Prop := x ≠ 3.5

theorem solve_inequality (x : ℝ) :
  inequality_condition_1 x →
  inequality_condition_2 x →
  inequality_condition_3 x →
  ( x ∈ (0, 0.5) ∨ x ∈ (0.5, 1) ∨ x ∈ (2, 2.5) ∨ x ∈ (4.5, 5) ∨ x ∈ (6, 6.5) ) :=
by sorry

end solve_inequality_l487_487123


namespace inequality_power_cubed_l487_487339

theorem inequality_power_cubed
  (x y a : ℝ)
  (h_condition : (0 < a ∧ a < 1) ∧ a ^ x < a ^ y) : x^3 > y^3 :=
by {
  sorry
}

end inequality_power_cubed_l487_487339


namespace find_k_l487_487009

-- Defining the quadratic equation with a known root x = 1
theorem find_k (k : ℝ) (h : (k-1) * 1^2 + k^2 - k = 0) : k = -1 := 
by {
  -- The proof goes here
  sorry,
}

end find_k_l487_487009


namespace sum_of_reciprocals_of_factors_of_12_l487_487806

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487806


namespace parallel_lines_condition_l487_487548

theorem parallel_lines_condition (a : ℝ) : 
  (a = -2) ↔ (∀ x y : ℝ, ax + 2 * y = 0 → y = 1 + x) := 
sorry

end parallel_lines_condition_l487_487548


namespace sum_of_reciprocals_of_factors_of_12_l487_487907

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487907


namespace sum_reciprocals_factors_12_l487_487948

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487948


namespace sum_reciprocals_factors_of_12_l487_487684

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487684


namespace sum_of_reciprocals_of_factors_of_12_l487_487754

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487754


namespace sum_of_reciprocals_factors_12_l487_487728

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487728


namespace sum_reciprocals_factors_12_l487_487699

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487699


namespace geometric_sequence_sum_less_than_2_l487_487429

-- Define the sequence a_n based on the conditions given.
noncomputable def a_seq : ℕ → ℝ
| 0       := 1  -- Note: we use a_0 because Lean lists are 0-indexed by default
| (n + 1) := 
    let prev := a_seq n in
    (-prev + (prev ^ 2 - 4 * prev) ^ 0.5) / (-2)

-- Define S_n as the sum of the first n terms of the sequence.
noncomputable def S (n : ℕ) : ℝ :=
  ∑ k in range (n + 1), a_seq k

-- Problem (1)
theorem geometric_sequence :
  ∃ r : ℝ, ∀ n > 0, (1 / a_seq n + 1) = r * (1 / a_seq (n-1) + 1) :=
sorry

-- Problem (2)
theorem sum_less_than_2 (n : ℕ) :
  S n < 2 :=
sorry

end geometric_sequence_sum_less_than_2_l487_487429


namespace find_radii_correct_l487_487428

noncomputable def find_radii (chord_length : ℝ) (ring_width : ℝ) : ℝ × ℝ :=
  let r := (sqrt (256 - 64)) / 2
  let R := r + ring_width
  (r, R)

theorem find_radii_correct 
  (O : ℝ) 
  (r : ℝ) 
  (chord_length : ℝ := 32)
  (ring_width : ℝ := 8) :
  let (r_found, R_found) := find_radii chord_length ring_width
  r_found = 12 ∧ R_found = 20 := 
by
  sorry

end find_radii_correct_l487_487428


namespace expected_value_of_unfair_die_l487_487306

noncomputable def probability_of_1_or_2_or_3_or_4_or_5 : ℚ := 1/10
noncomputable def probability_of_6 : ℚ := 1/2

theorem expected_value_of_unfair_die : 
  let EV : ℚ := 
    probability_of_1_or_2_or_3_or_4_or_5 * 1 + 
    probability_of_1_or_2_or_3_or_4_or_5 * 2 + 
    probability_of_1_or_2_or_3_or_4_or_5 * 3 + 
    probability_of_1_or_2_or_3_or_4_or_5 * 4 + 
    probability_of_1_or_2_or_3_or_4_or_5 * 5 + 
    probability_of_6 * 6
  in EV = 4.5 :=
by
  let EV : ℚ := 
    probability_of_1_or_2_or_3_or_4_or_5 * 1 + 
    probability_of_1_or_2_or_3_or_4_or_5 * 2 + 
    probability_of_1_or_2_or_3_or_4_or_5 * 3 + 
    probability_of_1_or_2_or_3_or_4_or_5 * 4 + 
    probability_of_1_or_2_or_3_or_4_or_5 * 5 + 
    probability_of_6 * 6
  have : EV = 4.5,
    sorry
  exact this

end expected_value_of_unfair_die_l487_487306


namespace sum_of_reciprocals_factors_12_l487_487725

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487725


namespace least_positive_multiple_of_35_gt_450_l487_487173

theorem least_positive_multiple_of_35_gt_450 : ∃ k : ℕ, k > 450 ∧ k % 35 = 0 ∧ k = 455 := 
by
  use 455
  split
  . exact Nat.lt_trans (450 : ℕ) (455 : ℕ) (Nat.lt_succ_self 450)
  split
  . exact Nat.mod_eq_zero_of_dvd (dvd.intro 13 rfl)
  exact rfl

end least_positive_multiple_of_35_gt_450_l487_487173


namespace problem_solution_l487_487299

noncomputable def count_valid_perimeters (p_bound : ℕ) : ℕ :=
  let valid_ys := { y | ∃ n : ℕ, y = n^2 + 1 ∧ 2 + 2*y + 2*Nat.sqrt(4*(y - 1)) < p_bound }
  Set.card valid_ys

theorem problem_solution : count_valid_perimeters 2015 = 31 :=
by
  sorry

end problem_solution_l487_487299


namespace number_of_elements_l487_487208

theorem number_of_elements (S n x : ℝ) (h_avg : n = 4 * (S / x))
  (h_sum : n = (1 / 6) * (S + n)) : x + 1 = 21 := by
  have h_eq : 4 * (S / x) = (1 / 6) * (S + 4 * (S / x)) := by
    rw [h_avg, h_sum]
  have h_clear : 24 * S = x * (S + 4 * (S / x)) := by
    linarith
  have h_simp : 24 * S = x * S + 4 * S := by
    rw [mul_add, mul_div_cancel' _ (ne_of_gt (by linarith))] at h_clear
    assumption
  have h_simp2 : 24 * S - 4 * S = x * S := by
    linarith
  have h_div : 20 * S = x * S := by
    linarith
  have h_x : x = 20 := by
    linarith
  show x + 1 = 21 from
    linarith

end number_of_elements_l487_487208


namespace smaller_number_l487_487143

theorem smaller_number (a b : ℕ) (h1 : 10 ≤ a ∧ a < 100) (h2 : 10 ≤ b ∧ b < 100) (h3 : a * b = 4851) : min a b = 53 :=
sorry

end smaller_number_l487_487143


namespace sum_reciprocals_of_factors_12_l487_487847

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487847


namespace sum_reciprocal_factors_of_12_l487_487599

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487599


namespace sum_reciprocals_of_factors_of_12_l487_487962

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487962


namespace length_XZ_l487_487424

-- Define the problem conditions
def WZ := 4
def XY := 8
def folded_rectangle (W X Y Z C : Point) : Prop :=
  is_rectangle W X Y Z ∧ W = Y ∧ C = W ∧ distance W Z = 4 ∧ distance X Y = 8

-- Define the theorem to prove
theorem length_XZ {W X Y Z C : Point} (h : folded_rectangle W X Y Z C) :
  distance X Z = 2 * sqrt (24 + 2 * sqrt 7) :=
sorry

end length_XZ_l487_487424


namespace sum_of_reciprocals_factors_12_l487_487826

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487826


namespace race_meeting_time_l487_487479

noncomputable def track_length : ℕ := 500
noncomputable def first_meeting_from_marie_start : ℕ := 100
noncomputable def time_until_first_meeting : ℕ := 2
noncomputable def second_meeting_time : ℕ := 12

theorem race_meeting_time
  (h1 : track_length = 500)
  (h2 : first_meeting_from_marie_start = 100)
  (h3 : time_until_first_meeting = 2)
  (h4 : ∀ t v1 v2 : ℕ, t * (v1 + v2) = track_length)
  (h5 : 12 = second_meeting_time) :
  second_meeting_time = 12 := by
  sorry

end race_meeting_time_l487_487479


namespace polynomial_abs_sum_eq_4_pow_9_l487_487307

theorem polynomial_abs_sum_eq_4_pow_9 :
  let p := (1 - 3 * x)^9 in
  (|p.coeff 0| + |p.coeff 1| + |p.coeff 2| + |p.coeff 3| + 
   |p.coeff 4| + |p.coeff 5| + |p.coeff 6| + 
   |p.coeff 7| + |p.coeff 8| + |p.coeff 9|) = 4^9 := 
by
  sorry

end polynomial_abs_sum_eq_4_pow_9_l487_487307


namespace sum_of_reciprocals_factors_12_l487_487715

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487715


namespace roots_of_polynomial_l487_487285

def p (x : ℝ) : ℝ := x^3 + x^2 - 4*x - 4

theorem roots_of_polynomial :
  (p (-1) = 0) ∧ (p 2 = 0) ∧ (p (-2) = 0) ∧ 
  ∀ x, p x = 0 → (x = -1 ∨ x = 2 ∨ x = -2) :=
by
  sorry

end roots_of_polynomial_l487_487285


namespace range_of_t_l487_487356

theorem range_of_t (A : Set ℝ) (t : ℝ) (h : ∃ x ∈ Iic t, x^2 - 4 * x + t ≤ 0) : t ∈ Iio 0 ∨ t ∈ Ioi 4 :=
by
  sorry

end range_of_t_l487_487356


namespace sum_of_reciprocals_of_factors_of_12_l487_487894

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487894


namespace biased_coin_problem_l487_487981

theorem biased_coin_problem 
  (h : ℝ)
  (H : 7.choose 2 * h^2 * (1-h)^5 = 7.choose 3 * h^3 * (1-h)^4)
  (h_ne_zero : h ≠ 0)
  (h_ne_one : h ≠ 1)
  : ∃ p q : ℕ, nat.gcd p q = 1 ∧ (35 * h^4 * (1-h)^3).num = p ∧ (35 * h^4 * (1-h)^3).denom = q ∧ p + q = 187 := 
by
  sorry

end biased_coin_problem_l487_487981


namespace sum_reciprocals_factors_12_l487_487888

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487888


namespace solution_l487_487534

def imaginary_part (z : ℂ) : ℝ := z.im

def problem_statement : Prop :=
  let z : ℂ := (2 + complex.i) / complex.i
  imaginary_part z = -2

theorem solution : problem_statement :=
  sorry

end solution_l487_487534


namespace sum_of_reciprocals_of_factors_of_12_l487_487802

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487802


namespace part1_price_reduction_5_part2_price_reduction_2100_part3_impossible_2200_l487_487235

variable (original_profit_per_item : ℝ)
variable (original_daily_sales_volume : ℝ)
variable (price_reduction_per_unit : ℝ)
variable (additional_items_per_dollar : ℝ)

/-- Part (1): Proving new sales volume and profit after a $5 reduction -/
theorem part1_price_reduction_5 (original_profit_per_item : ℝ) (original_daily_sales_volume : ℝ)
(additional_items_per_dollar : ℝ) :
  let reduced_price := 5 in
  let new_sales_volume := original_daily_sales_volume + additional_items_per_dollar * reduced_price in
  let total_daily_profit := (original_profit_per_item - reduced_price) * new_sales_volume in
  new_sales_volume = 40 ∧ total_daily_profit = 1800 :=
sorry

/-- Part (2): Proving price reduction for $2100 daily profit -/
theorem part2_price_reduction_2100 (original_profit_per_item : ℝ) (original_daily_sales_volume : ℝ)
(additional_items_per_dollar : ℝ) :
  ∃ (reduced_price : ℝ), 
    let new_sales_volume := original_daily_sales_volume + additional_items_per_dollar * reduced_price in
    let total_daily_profit := (original_profit_per_item - reduced_price) * new_sales_volume in
    total_daily_profit = 2100 :=
sorry

/-- Part (3): Proving impossibility of achieving $2200 daily profit -/
theorem part3_impossible_2200 (original_profit_per_item : ℝ) (original_daily_sales_volume : ℝ)
(additional_items_per_dollar : ℝ) :
  ¬ ∃ (reduced_price : ℝ), 
    let new_sales_volume := original_daily_sales_volume + additional_items_per_dollar * reduced_price in
    let total_daily_profit := (original_profit_per_item - reduced_price) * new_sales_volume in
    total_daily_profit = 2200 :=
sorry

end part1_price_reduction_5_part2_price_reduction_2100_part3_impossible_2200_l487_487235


namespace packages_to_buy_l487_487986

noncomputable def total_tshirts : ℝ := 71.0
noncomputable def tshirts_per_package : ℝ := 6.0
noncomputable def number_of_packages : ℝ := 12.0

theorem packages_to_buy :
  number_of_packages = Real.ceil (total_tshirts / tshirts_per_package) :=
by
  sorry

end packages_to_buy_l487_487986


namespace sum_of_reciprocals_of_factors_of_12_l487_487755

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487755


namespace sum_reciprocals_factors_12_l487_487942

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487942


namespace magical_stack_card_number_157_l487_487518

-- Definition of the problem
def is_magical_stack (n : ℕ) (number_of_cards : ℕ) : Prop :=
  ∃ (pos_A : ℕ) (pos_B : ℕ), 
  pos_A = (157 - 1) / 2 ∧ 
  pos_B = 78 ∧
  number_of_cards = 2 * n

-- The original card retains its initial position
def card_retains_position (stack : list ℕ) (pos : ℕ) : Prop :=
  stack.nth pos = some (pos + 1)

-- Proof statement
theorem magical_stack_card_number_157 :
  ∃ n : ℕ, is_magical_stack n 470 :=
by
  exists 235
  use 78, 78
  sorry

end magical_stack_card_number_157_l487_487518


namespace solve_for_t_l487_487396

theorem solve_for_t (s t : ℤ) (h1 : 11 * s + 7 * t = 160) (h2 : s = 2 * t + 4) : t = 4 :=
by
  sorry

end solve_for_t_l487_487396


namespace perpendicular_lines_k_values_l487_487537

theorem perpendicular_lines_k_values (k : ℝ) 
  (l1 : ∀ x y : ℝ, k * x + (1 - k) * y - 3 = 0) 
  (l2 : ∀ x y : ℝ, (k - 1) * x + (2 * k + 3) * y - 2 = 0) 
  (perp : ∀ x1 y1 x2 y2 : ℝ, l1 x1 y1 = 0 ∧ l2 x2 y2 = 0 → (k * (k - 1) + (1 - k) * (2 * k + 3) = 0)) :
  k = -3 ∨ k = 1 :=
sorry

end perpendicular_lines_k_values_l487_487537


namespace projection_of_b_onto_a_l487_487383

noncomputable def vector_projection (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let magnitude_squared := a.1^2 + a.2^2 + a.3^2
  let scalar := dot_product / magnitude_squared
  (scalar * a.1, scalar * a.2, scalar * a.3)

theorem projection_of_b_onto_a :
  vector_projection (2, -1, 2) (1, 2, 3) = (4 / 3, -2 / 3, 4 / 3) :=
by sorry

end projection_of_b_onto_a_l487_487383


namespace carly_butterfly_hours_l487_487257

-- Definitions for the problem
def butterfly_days_per_week : ℕ := 4
def weeks_in_a_month : ℕ := 4
def backstroke_hours_per_day : ℕ := 2
def backstroke_days_per_week : ℕ := 6
def total_practice_hours_per_month : ℕ := 96

noncomputable def butterfly_hours_per_day (B : ℕ) : Prop :=
  let butterfly_days_per_month := butterfly_days_per_week * weeks_in_a_month in
  let backstroke_days_per_month := backstroke_days_per_week * weeks_in_a_month in
  let backstroke_hours_per_month := backstroke_hours_per_day * backstroke_days_per_month in
  let butterfly_hours_per_month := total_practice_hours_per_month - backstroke_hours_per_month in
  butterfly_hours_per_month = B * butterfly_days_per_month

-- Statement to prove
theorem carly_butterfly_hours :
  butterfly_hours_per_day 3 :=
by
  -- Proof is omitted as instructed
  sorry

end carly_butterfly_hours_l487_487257


namespace sum_of_reciprocals_of_factors_of_12_l487_487904

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487904


namespace sum_reciprocals_12_l487_487917

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487917


namespace proof_problem_l487_487070

-- Definitions and conditions
variables {α β γ : Type} [plane α] [plane β] [plane γ]
variables {m n : Type} [line m] [line n]

-- Condition of distinct planes
axiom distinct_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ
-- Condition of non-intersecting lines
axiom non_intersecting_lines : ¬(m ∩ n).nonempty

-- Statement A
def statementA := (α ⊥ β ∧ β ⊥ γ) → α ⊥ γ

-- Statement B
def statementB := (α ∥ β ∧ ¬m ⊆ β ∧ m ∥ α) → m ∥ β

-- Statement C
def statementC := (α ⊥ β ∧ m ⊥ α) → m ∥ β

-- Statement D
def statementD := (m ∥ α ∧ n ∥ β ∧ α ⊥ β) → m ⊥ n

-- Proof problem statement (final formulation)
theorem proof_problem :
  (¬statementA) ∧ statementB ∧ statementC ∧ statementD :=
by sorry

end proof_problem_l487_487070


namespace triangle_side_c_value_l487_487432

theorem triangle_side_c_value (A B C a b c : ℝ) 
  (hA : A = Real.pi / 3)
  (ha : a = Real.sqrt 3)
  (hb : b = 1)
  (hC : C = Real.pi / 2)
  (h_triangle : A + B + C = Real.pi) :
  c = 2 :=
by
  sorry

end triangle_side_c_value_l487_487432


namespace initial_balance_before_check_deposit_l487_487196

theorem initial_balance_before_check_deposit (new_balance : ℝ) (initial_balance : ℝ) : 
  (50 = 1 / 4 * new_balance) → (initial_balance = new_balance - 50) → initial_balance = 150 :=
by
  sorry

end initial_balance_before_check_deposit_l487_487196


namespace problem1_problem2_l487_487414

noncomputable def triangle_boscos_condition (a b c A B : ℝ) : Prop :=
  b * Real.cos A = (2 * c + a) * Real.cos (Real.pi - B)

noncomputable def triangle_area (a b c : ℝ) (S : ℝ) : Prop :=
  S = (1 / 2) * a * c * Real.sin (2 * Real.pi / 3)

noncomputable def triangle_perimeter (a b c : ℝ) (P : ℝ) : Prop :=
  P = b + a + c

theorem problem1 (a b c A : ℝ) (h : triangle_boscos_condition a b c A (2 * Real.pi / 3)) : 
  ∃ B : ℝ, B = 2 * Real.pi / 3 :=
by
  sorry

theorem problem2 (a c : ℝ) (b : ℝ := 4) (area : ℝ := Real.sqrt 3) (P : ℝ) (h : triangle_area a b c area) (h_perim : triangle_perimeter a b c P) :
  ∃ x : ℝ, x = 4 + 2 * Real.sqrt 5 :=
by
  sorry

end problem1_problem2_l487_487414


namespace sum_of_reciprocals_factors_12_l487_487723

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487723


namespace sum_reciprocals_factors_12_l487_487618

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487618


namespace percentage_difference_l487_487519

variable (G : ℝ)
def P := 0.90 * G
def R := 1.5000000000000002 * G

theorem percentage_difference :
  ((R - P) / R) * 100 = 40.00000000000001 :=
by
  sorry

end percentage_difference_l487_487519


namespace sum_of_reciprocals_factors_12_l487_487790

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487790


namespace sum_of_reciprocals_of_factors_of_12_l487_487656

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487656


namespace sum_reciprocals_of_factors_12_l487_487842

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487842


namespace sum_reciprocals_factors_12_l487_487702

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487702


namespace symmetric_point_origin_l487_487036

-- Define the original point P with given coordinates
def P : ℝ × ℝ := (-2, 3)

-- Define the symmetric point P' with respect to the origin
def P'_symmetric (P : ℝ × ℝ) : ℝ × ℝ := (-P.1, -P.2)

-- The theorem states that the symmetric point of P is (2, -3)
theorem symmetric_point_origin : P'_symmetric P = (2, -3) := 
by
  sorry

end symmetric_point_origin_l487_487036


namespace max_intersections_two_circles_three_lines_l487_487571

theorem max_intersections_two_circles_three_lines :
  ∀ (C1 C2 : ℝ × ℝ × ℝ) (L1 L2 L3 : ℝ × ℝ × ℝ), 
  C1 ≠ C2 → L1 ≠ L2 → L2 ≠ L3 → L1 ≠ L3 →
  ∃ (P : ℕ), P = 17 :=
by 
  sorry

end max_intersections_two_circles_three_lines_l487_487571


namespace area_of_squares_and_difference_l487_487566

theorem area_of_squares_and_difference {r : ℝ} (h : r = 6) :
  let side1 := 2 * r,
      area1 := side1^2,
      side2 := side1 - 2,
      area2 := side2^2
  in area1 = 144 ∧ (area1 - area2) = 44 :=
by
  sorry

end area_of_squares_and_difference_l487_487566


namespace triangle_indeterminate_bc_l487_487018

open Set

theorem triangle_indeterminate_bc (A B C : Point) (AB AC : ℝ) (hAB : AB = 3) (hAC : AC = 2) :
  ∃ BC, BC = 3 ∨ BC = 4 ∨ BC = 5 ∨ … := sorry

end triangle_indeterminate_bc_l487_487018


namespace rowing_distance_correct_l487_487991

variable (D : ℝ) -- distance to the place
variable (speed_in_still_water : ℝ := 10) -- rowing speed in still water
variable (current_speed : ℝ := 2) -- speed of the current
variable (total_time : ℝ := 30) -- total time for round trip
variable (effective_speed_with_current : ℝ := speed_in_still_water + current_speed) -- effective speed with current
variable (effective_speed_against_current : ℝ := speed_in_still_water - current_speed) -- effective speed against current

theorem rowing_distance_correct : 
  D / effective_speed_with_current + D / effective_speed_against_current = total_time → 
  D = 144 := 
by
  intros h
  sorry

end rowing_distance_correct_l487_487991


namespace sum_of_reciprocals_factors_12_l487_487793

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487793


namespace arrangement_problem_l487_487494

def num_boxes : ℕ := 5
def num_balls : ℕ := 5
def num_fixed_points : ℕ := 2

noncomputable def arrangement_count (n m k : ℕ) : ℕ :=
  ∑ i in (finset.range n).filter (λ i, (i.choose k) = i), ∑ j in (finset.univ.filter (λ j : finset _ (finset.range n), j.card = m)), finset.card (j.derangements)

theorem arrangement_problem :
  arrangement_count num_boxes num_balls num_fixed_points = 20 :=
sorry

end arrangement_problem_l487_487494


namespace range_of_x_sqrt_4_2x_l487_487544

theorem range_of_x_sqrt_4_2x (x : ℝ) : (4 - 2 * x ≥ 0) ↔ (x ≤ 2) :=
by
  sorry

end range_of_x_sqrt_4_2x_l487_487544


namespace area_between_lines_one_third_l487_487563

theorem area_between_lines_one_third
  (A B C D M N K L : Point)
  (convex_ABCD : ConvexQuadrilateral A B C D)
  (divides_AB : AM = MN ∧ MN = NB)
  (divides_DC : DL = LK ∧ LK = KC) :
  area_between_lines (MN, KL) = (1 / 3) * area_quadrilateral (A B C D) :=
sorry

end area_between_lines_one_third_l487_487563


namespace base_eight_to_base_ten_l487_487567

theorem base_eight_to_base_ten : (4 * 8^1 + 5 * 8^0 = 37) := by
  sorry

end base_eight_to_base_ten_l487_487567


namespace partial_sum_difference_l487_487080

open_locale big_operators

-- Definitions from conditions
def sequence_condition (x : ℕ → ℕ) : Prop :=
  ∀ t : ℕ, t > 0 → ∑ i in finset.range (7 * t + 1) \ finset.range (7 * (t - 1)), x i ≤ 12

def partial_sum (x : ℕ → ℕ) (i : ℕ) : ℕ :=
  ∑ j in finset.range (i + 1), x j

def exists_indices (x : ℕ → ℕ) (n : ℕ) : Prop := 
  ∃ j k : ℕ, j < k ∧ partial_sum x k - partial_sum x j = n

-- Theorem statement
theorem partial_sum_difference (x : ℕ → ℕ) (h : sequence_condition x) (n : ℕ) (hn : n > 0) : 
  exists_indices x n :=
sorry

end partial_sum_difference_l487_487080


namespace yonder_license_plates_l487_487426

theorem yonder_license_plates : 
  let L := 26 in
  let D := 10 in
  (L * L * D * D * D * L = 17576000) := 
by
  let L := 26 in
  let D := 10 in
  show L * L * D * D * D * L = 17576000, from sorry

end yonder_license_plates_l487_487426


namespace number_of_possible_M_l487_487476

theorem number_of_possible_M :
  {M : Set ℕ // {1, 2} ⊆ M ∧ M ⊆ {1, 2, 3, 4} ∧ M ≠ {1, 2, 3, 4}}.to_finset.card = 3 :=
by
  sorry

end number_of_possible_M_l487_487476


namespace polar_coordinates_of_P_l487_487373

variable (x y : ℝ)
def P : ℝ × ℝ := (1, - (Real.sqrt 3))

def toPolar (P : ℝ × ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (P.1^2 + P.2^2)
  let θ := Real.arctan2 P.2 P.1
  (r, θ)

theorem polar_coordinates_of_P :
  toPolar (1, - (Real.sqrt 3)) = (2, - (π / 3)) :=
by
  sorry

end polar_coordinates_of_P_l487_487373


namespace sum_reciprocals_factors_12_l487_487946

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487946


namespace sum_reciprocals_factors_of_12_l487_487689

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487689


namespace darry_small_ladder_climbs_l487_487268

-- Define the constants based on the conditions
def full_ladder_steps := 11
def full_ladder_climbs := 10
def small_ladder_steps := 6
def total_steps := 152

-- Darry's total steps climbed via full ladder
def full_ladder_total_steps := full_ladder_steps * full_ladder_climbs

-- Define x as the number of times Darry climbed the smaller ladder
variable (x : ℕ)

-- Prove that x = 7 given the conditions
theorem darry_small_ladder_climbs (h : full_ladder_total_steps + small_ladder_steps * x = total_steps) : x = 7 :=
by 
  sorry

end darry_small_ladder_climbs_l487_487268


namespace trigonometric_expression_in_third_quadrant_l487_487203

theorem trigonometric_expression_in_third_quadrant (α : ℝ) 
  (h1 : Real.sin α < 0) 
  (h2 : Real.cos α < 0) 
  (h3 : Real.tan α > 0) : 
  ¬ (Real.tan α - Real.sin α < 0) :=
sorry

end trigonometric_expression_in_third_quadrant_l487_487203


namespace count_even_decreasing_digits_correct_l487_487389

def digits_in_decreasing_order (a b c : ℕ) : Prop :=
a > b ∧ b > c ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b < 10 ∧ 0 ≤ c ∧ c < 10

def is_even_digit (c : ℕ) : Prop :=
c ∈ {2, 4, 6, 8}

def count_even_decreasing_digits : ℕ :=
34

theorem count_even_decreasing_digits_correct :
  ∃ n : ℕ, (∀ a b c : ℕ, digits_in_decreasing_order a b c → is_even_digit c → 100 ≤ a * 100 + b * 10 + c ∧ a * 100 + b * 10 + c ≤ 999) ∧ n = count_even_decreasing_digits  :=
sorry

end count_even_decreasing_digits_correct_l487_487389


namespace absences_mean_median_difference_l487_487301

/-- Verify that the difference between the mean and median number of days 
    absent is \(1/10\), given specific frequency counts of absences. -/
theorem absences_mean_median_difference :
  let h_0 := 4
  let h_1 := 2
  let h_2 := 5
  let h_3 := 6
  let h_4 := 3
  let total_students := h_0 + h_1 + h_2 + h_3 + h_4
  let median := 2          -- Computed as the 10th and 11th values in the list of ordered absences
  let mean := (0 * h_0 + 1 * h_1 + 2 * h_2 + 3 * h_3 + 4 * h_4) / total_students
  in mean - median = 1 / 10 := 
by
  sorry

end absences_mean_median_difference_l487_487301


namespace sum_reciprocals_factors_12_l487_487643

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487643


namespace sum_reciprocals_factors_12_l487_487629

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487629


namespace sum_reciprocal_factors_of_12_l487_487605

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487605


namespace jake_correct_speed_l487_487097

noncomputable def distance (d t : ℝ) : Prop :=
  d = 50 * (t + 4/60) ∧ d = 70 * (t - 4/60)

noncomputable def correct_speed (d t : ℝ) : ℝ :=
  d / t

theorem jake_correct_speed (d t : ℝ) (h1 : distance d t) : correct_speed d t = 58 :=
by
  sorry

end jake_correct_speed_l487_487097


namespace simplify_and_evaluate_expression_l487_487117

def a : ℚ := 1 / 3
def b : ℚ := -1
def expr : ℚ := 4 * (3 * a^2 * b - a * b^2) - (2 * a * b^2 + 3 * a^2 * b)

theorem simplify_and_evaluate_expression : expr = -3 := 
by
  sorry

end simplify_and_evaluate_expression_l487_487117


namespace tip_percentage_is_10_l487_487151

-- Define the conditions
def total_bill : ℝ := 139.00
def number_of_people : ℕ := 8
def amount_paid_per_person : ℝ := 19.1125

-- Define the total paid by all people
def total_paid : ℝ := number_of_people * amount_paid_per_person

-- Define the total tip
def total_tip : ℝ := total_paid - total_bill

-- Define the tip percentage
def tip_percentage : ℝ := (total_tip / total_bill) * 100

-- Prove that the tip percentage is 10%
theorem tip_percentage_is_10 : tip_percentage = 10 := by
  sorry

end tip_percentage_is_10_l487_487151


namespace min_distance_from_origin_l487_487382

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (x - 1, 2)
noncomputable def vector_b (y : ℝ) : ℝ × ℝ := (4, y)

lemma perpendicular_vectors (x y : ℝ) (h : (x - 1, 2).1 * 4 + (x - 1, 2).2 * y = 0) :
  2 * x + y - 2 = 0 :=
begin
  rw [mul_comm] at h,
  simp only [mul_add, add_assoc, add_zero, add_right_eq_self, mul_comm] at h,
  exact h,
end

theorem min_distance_from_origin (x y : ℝ) (h : (x - 1, 2).1 * 4 + (x - 1, 2).2 * y = 0) :
  ∃ d : ℝ, d = 2 * real.sqrt 5 / 5 ∧ ∀ p : ℝ × ℝ, (abs (-2) / real.sqrt (2^2 + 1) = d) :=
begin
  use [(2 * real.sqrt 5) / 5],
  -- Add the geometric reasoning and steps here
  sorry, -- proof to be completed
end

end min_distance_from_origin_l487_487382


namespace simplify_expression_l487_487503

theorem simplify_expression : (2^8 + 4^5) * ((1^3 - (-1)^3)^8) = 327680 := by
  sorry

end simplify_expression_l487_487503


namespace sum_reciprocals_factors_12_l487_487875

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487875


namespace find_2u_plus_3v_l487_487362

theorem find_2u_plus_3v (u v : ℚ) (h1 : 5 * u - 6 * v = 28) (h2 : 3 * u + 5 * v = -13) :
  2 * u + 3 * v = -7767 / 645 := 
sorry

end find_2u_plus_3v_l487_487362


namespace solve_triangle_bisector_problem_l487_487126

noncomputable def triangle_bisector_proof : Prop :=
  ∀ (A B C D E F : Type)
    (h_triangle : Triangle A B C)
    (h_bisector : IsAngleBisector A B C D)
    (h_E_on_AC : PointOnLine E A C)
    (h_AD_BE_intersect : LinesIntersect_at AD BE F)
    (h_AF_FD_ratio : Ratio AF FD 3)
    (h_BF_FE_ratio : Ratio BF FE (5/3)),
    |AB| = |AC|

theorem solve_triangle_bisector_problem : triangle_bisector_proof :=
sorry

end solve_triangle_bisector_problem_l487_487126


namespace tv_cost_l487_487477

-- Define the total savings
def total_savings : ℕ := 600

-- Define the fraction spent on furniture
def fraction_spent_on_furniture : ℝ := 3/4

-- Define the amount spent on furniture
def amount_spent_on_furniture : ℝ := fraction_spent_on_furniture * total_savings

-- Define the cost of the TV
def cost_of_tv := total_savings - amount_spent_on_furniture

-- Prove that the cost of the TV is $150
theorem tv_cost : cost_of_tv = 150 := by
  sorry

end tv_cost_l487_487477


namespace cyclic_quadrilateral_relationship_l487_487360

-- Definitions and conditions:
-- Let A, B, C, D be the vertices of the cyclic quadrilateral.
-- E = AC ∩ BD, F = AD ∩ BC
variables {A B C D E F U X Y : Type} [cyclic_quadrilateral A B C D]

-- Define that U is the intersection point of lines AB and CD.
noncomputable def is_intersection_point (P Q R S : Type) : Prop := sorry

axiom intersection_U : is_intersection_point A B C D → U

-- Proving the necessary relationship UX * UY = UC * UD = UA * UB
theorem cyclic_quadrilateral_relationship (h₁ : is_intersection_point A B C D)
  (h₂ : E = intersection_point AC BD)
  (h₃ : F = intersection_point AD BC)
  (h₄ : similar_triangles EAD ECD)
  (h₅ : similar_triangles FAD FBC)
  (h₆ : proportional_segments (DY / YC) (DE / EC) (AD / BC) (UD / UB)) :
  (UX * UY = UC * UD = UA * UB) :=
sorry

end cyclic_quadrilateral_relationship_l487_487360


namespace ad_dot_bc_l487_487030

variables {V : Type*} [inner_product_space ℝ V]
variables (A B C D : V)
variables (AB BC AD : V)

-- Assume the properties of the equilateral triangle
def equilateral_triangle (A B C : V) (side : ℝ) : Prop :=
  ∥B - A∥ = side ∧ ∥C - B∥ = side ∧ ∥A - C∥ = side

-- We are given these conditions
def given_conditions (A B C : V) (side : ℝ) (D : V) : Prop :=
  equilateral_triangle A B C side ∧
  D - B = (1 / 3) • (C - B)

-- The dot product calculation we need to prove
theorem ad_dot_bc {A B C D : V} (side : ℝ) (h : given_conditions A B C side D) :
  let AD := (A - B) + (1 / 3) • (C - B) in
  AD ⬝ (C - B) = -2 / 3 :=
begin
  sorry
end

end ad_dot_bc_l487_487030


namespace sum_reciprocal_factors_of_12_l487_487606

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487606


namespace expression_value_l487_487280

def evaluate_expression : ℤ :=
  let base := -3 in
  let exponent := 3 - (-3) in
  let power := base ^ exponent in
  2 - power

theorem expression_value : evaluate_expression = -727 := by
  sorry

end expression_value_l487_487280


namespace distances_circumcenter_sides_l487_487082

variables {α β γ r : ℝ}
variables {a b c p_a p_b p_c : ℝ}

-- Conditions
def triangle_sides := a = 2 * r * sin α ∧ b = 2 * r * sin β ∧ c = 2 * r * sin γ

-- Distances from the circumcenter to the sides are p_a, p_b, p_c

theorem distances_circumcenter_sides :
  a = 2 * r * sin α ∧ b = 2 * r * sin β ∧ c = 2 * r * sin γ → 
  p_a * sin α + p_b * sin β + p_c * sin γ = 2 * r * sin α * sin β * sin γ :=
begin
  sorry
end

end distances_circumcenter_sides_l487_487082


namespace range_of_f_l487_487146

def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem range_of_f : set.range (λ x : ℝ, f x) ∩ set.Icc 0 2 = set.Icc (-3) 5 :=
sorry

end range_of_f_l487_487146


namespace polygon_sides_l487_487410

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1440) : n = 10 :=
by sorry

end polygon_sides_l487_487410


namespace sum_of_reciprocals_of_factors_of_12_l487_487669

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487669


namespace max_points_of_intersection_l487_487577

theorem max_points_of_intersection (circles : Fin 2 → Circle) (lines : Fin 3 → Line) :
  number_of_intersections circles lines = 17 :=
sorry

end max_points_of_intersection_l487_487577


namespace product_of_binomials_l487_487293

theorem product_of_binomials (x : ℝ) : 
  (4 * x - 3) * (2 * x + 7) = 8 * x^2 + 22 * x - 21 := by
  sorry

end product_of_binomials_l487_487293


namespace right_triangle_no_k_values_l487_487545

theorem right_triangle_no_k_values (k : ℕ) (h : k > 0) : 
  ¬ (∃ k, k > 0 ∧ ((17 > k ∧ 17^2 = 13^2 + k^2) ∨ (k > 17 ∧ k < 30 ∧ k^2 = 13^2 + 17^2))) :=
sorry

end right_triangle_no_k_values_l487_487545


namespace sum_of_reciprocals_factors_12_l487_487733

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487733


namespace petyas_number_l487_487492

theorem petyas_number {N : ℕ}
  (h1 : Nat.digits 10 N > 2) -- given that N has more than 2 digits.
  (h2 : (N % 10) % 2 = 1) -- given that N is odd.
  (h3 : ∀ m, (m * 10^(Nat.digits 10 N - 2) + N % 10^(Nat.digits 10 N - 2) = N) ∧ m * 10^(Nat.digits 10 N - 2) = 148 * (N % 10^(Nat.digits 10 N - 2))) -- given when removing the first two digits, the resulting number divides perfectly into 149 equal parts.
  : N = 745 ∨ N = 3725 := 
sorry

end petyas_number_l487_487492


namespace cone_base_radius_to_sphere_radius_ratio_l487_487214

theorem cone_base_radius_to_sphere_radius_ratio (R r : ℝ)
  (h1: ∃ (d l: ℝ), d = 2*R ∧ l = d)
  (h2: ∃ (O1 O2 O3: ℝ×ℝ×ℝ), (dist O1 O2) = 2*r ∧ (dist O2 O3) = 2*r ∧
       (dist O3 O1) = 2*r ∧ O1.2 = 0 ∧ O2.2 = 0 ∧ O3.2 = r)
  (h3: ∃ (O4: ℝ×ℝ×ℝ), (dist O4 O1) = r ∧ (dist O4 O2) = r ∧
       (plane_of_centers O1 O2 O3).symm = (tangent_point O4 lateral_surface)):
  R / r = 5/4 + real.sqrt 3 := sorry

end cone_base_radius_to_sphere_radius_ratio_l487_487214


namespace right_triangle_hypotenuse_l487_487114

theorem right_triangle_hypotenuse
  (AD AB EF AF : ℝ)
  (h_AD : AD = 8)
  (h_AB : AB = 7)
  (h_area_eq : AB * AD = 1 / 2 * AD * EF)
  (h_ef_perpendicular_ad : EF ≠ 0) :
  AF = 2 * Real.sqrt 65 :=
by
  have area_rect : 56 = AB * AD,
  { rw [h_AB, h_AD], norm_num, },
  have area_tri : 56 = 1 / 2 * AD * EF,
  { rw [h_area_eq, area_rect], },
  have eqn_for_x : 1 / 2 * AD * EF = 56,
  { rw [area_tri], },
  have h_x : EF = 14,
  { linarith [h_AD], },
  have af_sq : AF^2 = AD^2 + EF^2,
  { ring_nf, },
  have h_af2 : AF^2 = 260,
  { rw [h_AD, ← h_x], norm_num, },
  have h_af : AF = Real.sqrt 260,
  { exact Real.sqrt_eq_iff_sq_eq.mpr ⟨Real.sq_nonneg 260, h_af2.symm⟩, },
  rw ← Real.mul_sqrt,
  rw Real.sqrt_mul,
  norm_num,
  exact h_af.symm,
  any_goals { exact Real.sq_nonneg _ },
  sorry

end right_triangle_hypotenuse_l487_487114


namespace find_angle_A_l487_487418

-- Define the given conditions
variables (B C A : ℝ)
variable (angle_in_small_triangle : ℝ)
variable (linear_pair_property : ∀ x y : ℝ, x + y = 180)

-- Specify the conditions
axiom B_is_120_degrees : B = 120
axiom C_adjacent_to_B : C + B = 180
axiom angle_in_triangle_is_50 : angle_in_small_triangle = 50

-- Define the theorem to prove
theorem find_angle_A (B C A : ℝ) (angle_in_small_triangle : ℝ) 
  [B_is_120_degrees : B = 120] 
  [C_adjacent_to_B : C + B = 180]
  [angle_in_triangle_is_50 : angle_in_small_triangle = 50]
  [linear_pair_property : ∀ x y : ℝ, x + y = 180] :
  A = 60 :=
  sorry

end find_angle_A_l487_487418


namespace equilateral_triangle_ABC_l487_487054

open Classical

-- Let D, E, and F be points on sides BC, CA, and AB respectively in triangle ABC
variables (A B C D E F : Type)
-- Assume there is a triangle ABC with points D, E, F such that BD = CE = AF and angles are equal
variables [BG : Geometry ABC] [DG : Distance]

noncomputable def triangle_condition (A B C D E F : Type) :=
  let x : ℝ := sorry in
  BD = x ∧ CE = x ∧ AF = x ∧
  ∠ BDF = ∠ CED ∧ ∠ CED = ∠ AFE

-- Define the main proof problem
theorem equilateral_triangle_ABC (A B C D E F : Type)
  [h : triangle_condition A B C D E F] : is_equilateral A B C :=
sorry

end equilateral_triangle_ABC_l487_487054


namespace sum_of_reciprocals_of_factors_of_12_l487_487661

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487661


namespace sum_reciprocal_factors_12_l487_487862

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487862


namespace sum_reciprocals_of_factors_12_l487_487836

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487836


namespace sum_reciprocals_factors_of_12_l487_487676

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487676


namespace digit_9_appears_140_times_l487_487393

theorem digit_9_appears_140_times :
  ∑ n in Finset.range 700, (n.digits 10).count 9 = 140 :=
by sorry

end digit_9_appears_140_times_l487_487393


namespace maria_total_money_l487_487092

theorem maria_total_money (Rene Florence Isha : ℕ) (hRene : Rene = 300)
  (hFlorence : Florence = 3 * Rene) (hIsha : Isha = Florence / 2) :
  Isha + Florence + Rene = 1650 := by
  sorry

end maria_total_money_l487_487092


namespace smallest_number_l487_487487

theorem smallest_number (x y z : ℕ) (h1 : y = 4 * x) (h2 : z = 2 * y) 
(h3 : (x + y + z) / 3 = 78) : x = 18 := 
by 
    sorry

end smallest_number_l487_487487


namespace count_complex_numbers_l487_487461

-- Define the complex function f and the main problem
def f (z : ℂ) : ℂ := z^2 + 2 * complex.I * z + 2

theorem count_complex_numbers :
  let valid_z_count : ℕ := (∑ a in (-5 : ℤ)..5, 
                          ∑ b in (-5 : ℤ)..5, 
                            if a^2 - 3 * a + b^2 > 0 then 1 else 0)
  in valid_z_count = 86 := by
  let valid_z := (∑ a in (-5 : ℤ)..5, 
                  ∑ b in (-5 : ℤ)..5, 
                    if a^2 - 3 * a + b^2 > 0 then 1 else 0)
  have : valid_z = 86 := sorry
  exact this

end count_complex_numbers_l487_487461


namespace angle_A_eq_pi_div_3_find_a_l487_487019

-- Define the first problem statement.
theorem angle_A_eq_pi_div_3 (a b c : ℝ) (B : ℝ) (h : 2 * c - 2 * a * Real.cos B = b) :
  let A := (Real.arccos (1 / 2))
  in A = Real.pi / 3 := sorry

-- Define the second problem statement.
theorem find_a (a b c A : ℝ) (area_eq : (Real.sqrt 3) / 4 = (b * c * Real.sin A) / 2) 
  (hA: A = Real.pi / 3) (h: c^2 + a * b * Real.cos (Real.complex_cos (A / 2)) + a^2 = 4) :
  a = (Real.sqrt 7) / 2 := sorry

end angle_A_eq_pi_div_3_find_a_l487_487019


namespace sum_octal_eq_1021_l487_487296

def octal_to_decimal (n : ℕ) : ℕ :=
  let d0 := n % 10
  let r1 := n / 10
  let d1 := r1 % 10
  let r2 := r1 / 10
  let d2 := r2 % 10
  (d2 * 64) + (d1 * 8) + d0

def decimal_to_octal (n : ℕ) : ℕ :=
  let d0 := n % 8
  let r1 := n / 8
  let d1 := r1 % 8
  let r2 := r1 / 8
  let d2 := r2 % 8
  d2 * 100 + d1 * 10 + d0

theorem sum_octal_eq_1021 :
  decimal_to_octal (octal_to_decimal 642 + octal_to_decimal 157) = 1021 := by
  sorry

end sum_octal_eq_1021_l487_487296


namespace tom_walked_kilometers_l487_487159

theorem tom_walked_kilometers :
  let base7_number := 4536
  in let converted_number := 4 * 7^3 + 5 * 7^2 + 3 * 7^1 + 6 * 7^0
  in converted_number = 1644 :=
by
  let base7_number := 4536
  let converted_number := 4 * 7^3 + 5 * 7^2 + 3 * 7^1 + 6 * 7^0
  show converted_number = 1644 from
    sorry

end tom_walked_kilometers_l487_487159


namespace sqrt_nested_l487_487400

theorem sqrt_nested (x : ℝ) (hx : 0 ≤ x) : Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (15 / 16) := by
  sorry

end sqrt_nested_l487_487400


namespace product_not_50_l487_487193

theorem product_not_50 :
  (1 / 2 * 100 = 50) ∧
  (-5 * -10 = 50) ∧
  ¬(5 * 11 = 50) ∧
  (2 * 25 = 50) ∧
  (5 / 2 * 20 = 50) :=
by
  sorry

end product_not_50_l487_487193


namespace sum_reciprocals_factors_12_l487_487710

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487710


namespace opal_total_savings_l487_487101

def initial_winnings : ℝ := 100.00
def savings_first : ℝ := initial_winnings / 2
def betting_first : ℝ := initial_winnings / 2
def profit_rate : ℝ := 0.60
def profit_second : ℝ := profit_rate * betting_first
def earnings_second : ℝ := betting_first + profit_second
def savings_second : ℝ := earnings_second / 2
def total_savings : ℝ := savings_first + savings_second

theorem opal_total_savings : total_savings = 90.00 :=
by
  -- proof will go here
  sorry

end opal_total_savings_l487_487101


namespace sum_reciprocals_factors_12_l487_487631

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487631


namespace option_B_is_correct_l487_487399

-- Definitions and Conditions
variable {Line : Type} {Plane : Type}
variable (m n : Line) (α β γ : Plane)
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Conditions
axiom m_perp_β : perpendicular m β
axiom m_parallel_α : parallel m α

-- Statement to prove
theorem option_B_is_correct : perpendicular_planes α β :=
by
  sorry

end option_B_is_correct_l487_487399


namespace pH_pure_water_l487_487513

theorem pH_pure_water (H_pos : ℝ)
  (log10 : ℝ → ℝ)
  (log10_pos : ∀ x, x > 0 → log10 x > 0)
  (log10_inv : ∀ x : ℝ, x > 0 → log10 (1 / x) = - log10 x)
  (H_pure : H_pos = 10 ^ (-7)) :
  - log10 H_pos = 7 :=
by
  -- include all premises
  have H1 : H_pos = 10 ^ (-7) := H_pure
  -- transform the equation using log properties
  rw H1
  apply log10_inv
  apply pow_pos
  norm_num
  sorry

end pH_pure_water_l487_487513


namespace sum_of_digits_of_d_l487_487115

-- Given definitions based on the conditions
def dollars_to_canadian (d : ℕ) : ℕ := (8 * d) / 5
def spent (d : ℕ) : ℕ := dollars_to_canadian d - d

-- Main statement to prove the sum of the digits of d is 7
theorem sum_of_digits_of_d (d : ℕ) (h1 : dollars_to_canadian d - 80 = d) : digitSum d = 7 :=
by 
  sorry 

end sum_of_digits_of_d_l487_487115


namespace dons_profit_l487_487441

-- Definitions from the conditions
def bundles_jamie_bought := 20
def bundles_jamie_sold := 15
def profit_jamie := 60

def bundles_linda_bought := 34
def bundles_linda_sold := 24
def profit_linda := 69

def bundles_don_bought := 40
def bundles_don_sold := 36

-- Variables representing the unknown prices
variables (b s : ℝ)

-- Conditions written as equalities
axiom eq_jamie : bundles_jamie_sold * s - bundles_jamie_bought * b = profit_jamie
axiom eq_linda : bundles_linda_sold * s - bundles_linda_bought * b = profit_linda

-- Statement to prove Don's profit
theorem dons_profit : bundles_don_sold * s - bundles_don_bought * b = 252 :=
by {
  sorry -- proof goes here
}

end dons_profit_l487_487441


namespace sum_reciprocals_factors_12_l487_487628

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487628


namespace sum_reciprocals_factors_12_l487_487737

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487737


namespace sum_reciprocal_factors_of_12_l487_487613

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487613


namespace complete_the_square_l487_487185

theorem complete_the_square (x : ℝ) : 
    (x^2 - 2 * x - 5 = 0) -> (x - 1)^2 = 6 :=
by sorry

end complete_the_square_l487_487185


namespace sum_of_reciprocals_factors_12_l487_487816

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487816


namespace time_bob_cleans_room_l487_487243

variable (timeAlice : ℕ) (fractionBob : ℚ)

-- Definitions based on conditions from the problem
def timeAliceCleaningRoom : ℕ := 40
def fractionOfTimeBob : ℚ := 3 / 8

-- Prove the time it takes Bob to clean his room
theorem time_bob_cleans_room : (timeAliceCleaningRoom * fractionOfTimeBob : ℚ) = 15 := 
by
  sorry

end time_bob_cleans_room_l487_487243


namespace projection_onto_plane_l487_487061

open Matrix

noncomputable def normal_vector : Vector ℝ 3 := ![2, -1, 2]

noncomputable def projection_matrix : Matrix (Fin 3) (Fin 3) ℝ := 
![![5 / 9, 2 / 9, -4 / 9], 
  ![2 / 9, 8 / 9, 2 / 9], 
  ![-4 / 9, 2 / 9, 5 / 9]]

theorem projection_onto_plane (v : Vector ℝ 3) :
  let Q := proj ℝ (span ℝ (Set.range ![normal_vector])) in 
  projection_matrix.mul_vec v = Q v :=
sorry

end projection_onto_plane_l487_487061


namespace sum_reciprocals_factors_12_l487_487876

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487876


namespace sum_of_reciprocals_factors_12_l487_487722

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487722


namespace sum_reciprocals_factors_12_l487_487705

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487705


namespace part1_reduced_sales_volume_part1_reduced_profit_part2_price_reduction_for_2100_part3_cannot_achieve_2200_l487_487233

variables (initial_profit_per_item : ℕ) (initial_sales_volume : ℕ) (price_reduction_per_item : ℕ) (additional_sales_per_reduction : ℕ)

-- Conditions given in the problem
def conditions : Prop := 
  initial_profit_per_item = 50 ∧ initial_sales_volume = 30 ∧ price_reduction_per_item = 1 ∧ additional_sales_per_reduction = 2

-- Part 1: Sales volume and profit after $5 reduction
def reduced_price_5_sales_volume : Prop := 
  (initial_sales_volume + additional_sales_per_reduction * 5 = 40)

def reduced_price_5_profit : Prop := 
  (initial_profit_per_item - 5) * (initial_sales_volume + additional_sales_per_reduction * 5) = 1800

-- Part 2: Price reduction needed for $2100 profit
def price_reduction_for_2100_profit : Prop := 
  ∃ x : ℕ, (initial_profit_per_item - x) * (initial_sales_volume + additional_sales_per_reduction * x) = 2100

-- Part 3: Possibility of achieving $2200 profit
def can_achieve_2200_profit : Prop := 
  ¬(∃ x : ℕ, (initial_profit_per_item - x) * (initial_sales_volume + additional_sales_per_reduction * x) = 2200)

-- Main theorem statements
theorem part1_reduced_sales_volume (h : conditions) : reduced_price_5_sales_volume := sorry

theorem part1_reduced_profit (h : conditions) : reduced_price_5_profit := sorry

theorem part2_price_reduction_for_2100 (h : conditions) : price_reduction_for_2100_profit := sorry

theorem part3_cannot_achieve_2200 (h : conditions) : can_achieve_2200_profit := sorry

end part1_reduced_sales_volume_part1_reduced_profit_part2_price_reduction_for_2100_part3_cannot_achieve_2200_l487_487233


namespace max_intersections_two_circles_three_lines_l487_487568

theorem max_intersections_two_circles_three_lines :
  ∀ (C1 C2 : ℝ × ℝ × ℝ) (L1 L2 L3 : ℝ × ℝ × ℝ), 
  C1 ≠ C2 → L1 ≠ L2 → L2 ≠ L3 → L1 ≠ L3 →
  ∃ (P : ℕ), P = 17 :=
by 
  sorry

end max_intersections_two_circles_three_lines_l487_487568


namespace sum_reciprocals_factors_12_l487_487703

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487703


namespace no_solution_in_natural_numbers_l487_487497

theorem no_solution_in_natural_numbers (x y z : ℕ) : ¬((2 * x) ^ (2 * x) - 1 = y ^ (z + 1)) := 
  sorry

end no_solution_in_natural_numbers_l487_487497


namespace sum_of_reciprocals_factors_12_l487_487726

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487726


namespace shelter_total_cats_l487_487053

theorem shelter_total_cats (total_adult_cats num_female_cats num_litters avg_kittens_per_litter : ℕ) 
  (h1 : total_adult_cats = 150) 
  (h2 : num_female_cats = 2 * total_adult_cats / 3)
  (h3 : num_litters = 2 * num_female_cats / 3)
  (h4 : avg_kittens_per_litter = 5):
  total_adult_cats + num_litters * avg_kittens_per_litter = 480 :=
by
  sorry

end shelter_total_cats_l487_487053


namespace sum_of_reciprocals_of_factors_of_12_l487_487659

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487659


namespace yolanda_walking_rate_correct_l487_487198

-- Definitions and conditions
def distance_XY : ℕ := 65
def bobs_walking_rate : ℕ := 7
def bobs_distance_when_met : ℕ := 35
def yolanda_start_time (t: ℕ) : ℕ := t + 1 -- Yolanda starts walking 1 hour earlier

-- Yolanda's walking rate calculation
def yolandas_walking_rate : ℕ := 5

theorem yolanda_walking_rate_correct { time_bob_walked : ℕ } 
  (h1 : distance_XY = 65)
  (h2 : bobs_walking_rate = 7)
  (h3 : bobs_distance_when_met = 35) 
  (h4 : time_bob_walked = bobs_distance_when_met / bobs_walking_rate)
  (h5 : yolanda_start_time time_bob_walked = 6) -- since bob walked 5 hours, yolanda walked 6 hours
  (h6 : distance_XY - bobs_distance_when_met = 30) :
  yolandas_walking_rate = ((distance_XY - bobs_distance_when_met) / yolanda_start_time time_bob_walked) := 
sorry

end yolanda_walking_rate_correct_l487_487198


namespace sum_of_reciprocals_of_factors_of_12_l487_487772

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487772


namespace part1_part2_l487_487367

/-- Part (1) -/
theorem part1 (a : ℝ) (p : ∀ x : ℝ, x^2 - a*x + 4 > 0) (q : ∀ x y : ℝ, (0 < x ∧ x < y) → x^a < y^a) : 
  0 < a ∧ a < 4 :=
sorry

/-- Part (2) -/
theorem part2 (a : ℝ) (p_iff: ∀ x : ℝ, x^2 - a*x + 4 > 0 ↔ -4 < a ∧ a < 4)
  (q_iff: ∀ x y : ℝ, (0 < x ∧ x < y) ↔ x^a < y^a ∧ a > 0) (hp : ∃ x : ℝ, ¬(x^2 - a*x + 4 > 0))
  (hq : ∀ x y : ℝ, (x^a < y^a) → (0 < x ∧ x < y)) : 
  (a >= 4) ∨ (-4 < a ∧ a <= 0) :=
sorry

end part1_part2_l487_487367


namespace projection_matrix_correct_l487_487067

noncomputable def Q_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  !![
    [7/9, 1/9, -2/9],
    [1/9, 11/18, 1/9],
    [-2/9, 1/9, 7/9]
  ]

def normal_vector : Vector (Fin 3) ℝ :=
  !![
    [2],
    [-1],
    [2]
  ]

def projection_on_plane (v : Vector (Fin 3) ℝ) : Vector (Fin 3) ℝ :=
  Q_matrix ⬝ v

theorem projection_matrix_correct (v : Vector (Fin 3) ℝ) :
  projection_on_plane v = Q_matrix ⬝ v :=
sorry

end projection_matrix_correct_l487_487067


namespace sum_of_reciprocals_of_factors_of_12_l487_487903

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487903


namespace sum_reciprocals_factors_12_l487_487878

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487878


namespace general_term_correct_inequality_l487_487474

-- Define the sequence {a_n} and its recurrence relationship

def sequence_a : ℕ → ℝ 
| 0 := 1 
| (n+1) := let a_n := sequence_a n in 
            let x := 1 / real.sqrt(n + 1) in 
            real.sqrt(x * (x + a_n)/a_n)  

-- Define the recurrence relation as a property
def recurrence_relation (a_n a_n1: ℝ) (n: ℕ) :=
  a_n1 * (a_n1 - a_n) = (1 / real.sqrt (n)) * ((1 / real.sqrt (n)) + a_n)

-- Question 1: Prove the general term for the sequence
def general_term (n : ℕ) : ℝ :=
  (finset.range n).sum (λ k, real.sqrt (k + 1) / (k + 1))

theorem general_term_correct : ∀ n : ℕ, sequence_a n = general_term n := 
sorry 

-- Question 2: Prove the inequality for distinct positive integers
theorem inequality (n : ℕ) (b : fin n → ℕ) :
  (∀ i j : fin n, i ≠ j → b i ≠ b j) →
  sequence_a n ≤ (finset.range n).sum (λ k, real.sqrt (b ⟨k, sorry⟩) / (k + 1))  := 
sorry

end general_term_correct_inequality_l487_487474


namespace sum_of_x_coords_l487_487128

def g : ℝ → ℝ
| x if x ≤ -4 := -5 + 2 * (x + 4)
| x if -4 < x ∧ x ≤ -2 := -5 + 2 * (x + 4)
| x if -2 < x ∧ x ≤ -1 := -1
| x if -1 < x ∧ x ≤ 1 := -2 + 2 * (x + 1)
| x if 1 < x ∧ x ≤ 2 := 2
| x if 2 < x ∧ x ≤ 4 := 1 + 2 * (x - 2)

theorem sum_of_x_coords :
  ∑ x in {-4, 1, 4}.to_finset, x = 1 := by
  sorry

end sum_of_x_coords_l487_487128


namespace sum_of_reciprocals_of_factors_of_12_l487_487905

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487905


namespace part1_part2_part3_l487_487300

-- Define the function f(x)
def f (x : ℝ) : ℝ := ∫ t in 0..x, 1 / (1 + t^2)

-- Prove f(√3) = π / 3
theorem part1 : f (Real.sqrt 3) = Real.pi / 3 :=
by
  sorry

-- Prove ∫ x in 0..√3, x * f(x) dx = (ln 4) / 2
theorem part2 : (∫ x in 0..Real.sqrt 3, x * f x) = Real.log 4 / 2 :=
by
  sorry

-- Prove f(x) + f(1/x) = π / 2 for x > 0
theorem part3 (x : ℝ) (hx : 0 < x) : f(x) + f(1/x) = Real.pi / 2 :=
by
  sorry

end part1_part2_part3_l487_487300


namespace repeating_decimal_to_fraction_denominator_l487_487129

theorem repeating_decimal_to_fraction_denominator :
  ∀ (S : ℚ), (S = 0.27) → (∃ a b : ℤ, b ≠ 0 ∧ S = a / b ∧ Int.gcd a b = 1 ∧ b = 3) :=
by
  sorry

end repeating_decimal_to_fraction_denominator_l487_487129


namespace find_k_l487_487381

variables (k : ℝ)

def a := (k, 3)
def b := (1, 4)
def c := (2, 1)
def lhs := (2 * (a.1) - 3 * (b.1), 2 * (a.2) - 3 * (b.2))

theorem find_k (h : lhs.1 * c.1 + lhs.2 * c.2 = 0) : k = 3 := by
  sorry

end find_k_l487_487381


namespace sum_reciprocals_factors_12_l487_487949

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487949


namespace sum_reciprocals_of_factors_of_12_l487_487960

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487960


namespace distance_a_beats_b_l487_487207

noncomputable def time_a : ℕ := 90 -- A's time in seconds 
noncomputable def time_b : ℕ := 180 -- B's time in seconds 
noncomputable def distance : ℝ := 4.5 -- distance in km

theorem distance_a_beats_b : distance = (distance / time_a) * (time_b - time_a) :=
by
  -- sorry placeholder for proof
  sorry

end distance_a_beats_b_l487_487207


namespace sum_reciprocals_factors_of_12_l487_487688

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487688


namespace least_positive_integer_l487_487589

theorem least_positive_integer :
  ∃ (a : ℕ), (a ≡ 1 [MOD 3]) ∧ (a ≡ 2 [MOD 4]) ∧ (∀ b, (b ≡ 1 [MOD 3]) → (b ≡ 2 [MOD 4]) → b ≥ a → b = a) :=
sorry

end least_positive_integer_l487_487589


namespace right_triangle_area_l487_487358

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := 1 / 2 * a * b

theorem right_triangle_area {a b : ℝ} 
  (h1 : a + b = 4) 
  (h2 : a^2 + b^2 = 14) : 
  area_of_right_triangle a b = 1 / 2 :=
by 
  sorry

end right_triangle_area_l487_487358


namespace sum_of_reciprocals_of_factors_of_12_l487_487911

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487911


namespace sum_reciprocals_factors_12_l487_487893

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487893


namespace digit_9_appears_140_times_l487_487392

theorem digit_9_appears_140_times :
  ∑ n in Finset.range 700, (n.digits 10).count 9 = 140 :=
by sorry

end digit_9_appears_140_times_l487_487392


namespace number_of_triangles_l487_487098

theorem number_of_triangles (n : ℕ) (h : n = 9) : (Nat.choose n 3) = 84 :=
by
  rw [h]
  dsimp [Nat.choose]
  exact calc
    Nat.choose 9 3 = 9 * 8 * 7 / (3 * 2 * 1) : by sorry
              ... = 504 / 6                 : by sorry
              ... = 84                      : by sorry

end number_of_triangles_l487_487098


namespace sum_of_reciprocals_of_factors_of_12_l487_487897

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487897


namespace sum_of_reciprocals_of_factors_of_12_l487_487758

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487758


namespace chess_grandmaster_time_l487_487440

theorem chess_grandmaster_time :
  let time_to_learn_rules : ℕ := 2
  let factor_to_get_proficient : ℕ := 49
  let factor_to_become_master : ℕ := 100
  let time_to_get_proficient := factor_to_get_proficient * time_to_learn_rules
  let combined_time := time_to_learn_rules + time_to_get_proficient
  let time_to_become_master := factor_to_become_master * combined_time
  let total_time := time_to_learn_rules + time_to_get_proficient + time_to_become_master
  total_time = 10100 :=
by
  sorry

end chess_grandmaster_time_l487_487440


namespace algebraic_expression_is_integer_l487_487466

noncomputable def roots_of_cubic : List ℂ := sorry -- assume these are the complex roots a, b, c

lemma problem_conditions :
  ∃ (a b c : ℂ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    (∀ x, x ∈ roots_of_cubic → x^3 - x^2 - x - 1 = 0) :=
sorry

theorem algebraic_expression_is_integer :
  ∀ (a b c : ℂ), 
    a ≠ b → b ≠ c → c ≠ a → 
    (∀ x, x ∈ roots_of_cubic → x^3 - x^2 - x - 1 = 0) →
    (∃ (n : ℕ), 
      (n = 1993) →
      let expr := (a ^ n - b ^ n) / (a - b) + 
                  (b ^ n - c ^ n) / (b - c) + 
                  (c ^ n - a ^ n) / (c - a)
      in expr ∈ ℤ) :=
begin
  intros a b c h1 h2 h3 h4 n hn,
  use n,
  split,
  { exact hn, },
  { sorry, }
end

end algebraic_expression_is_integer_l487_487466


namespace sum_of_reciprocals_of_factors_of_12_l487_487756

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487756


namespace correct_integer_count_l487_487554

-- Definitions
def within_range (n : ℤ) : Prop :=
  (-2 : ℝ) < (n : ℝ) ∧ (n : ℝ) < (3 : ℝ)

def integer_count_within_range : ℕ :=
  ([-2, -1, 0, 1, 2, 3].filter within_range).length

-- The statement we aim to prove
theorem correct_integer_count : integer_count_within_range = 6 :=
by sorry

end correct_integer_count_l487_487554


namespace sum_of_roots_eq_neg_two_l487_487531

theorem sum_of_roots_eq_neg_two (α : Type) [LinearOrderedField α] :
  let f := λ x : α, 4*x^2 + 6*x + 3 in
  (∀ x : α, f x = 0 → x = -3/2 ∨ x = -1/2) →
  ((-3/2) + (-1/2) = -2) :=
by
  sorry

end sum_of_roots_eq_neg_two_l487_487531


namespace sum_of_reciprocals_of_factors_of_12_l487_487660

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487660


namespace selecting_televisions_l487_487152

theorem selecting_televisions :
  (∃ (tvs : Finset ℕ) (A B C : Finset ℕ), 
     tvs.card = 10 ∧
     A.card = 3 ∧
     B.card = 3 ∧
     C.card = 4 ∧
     A ∪ B ∪ C = tvs ∧ 
     (∀ x ∈ tvs, x ∈ A ∨ x ∈ B ∨ x ∈ C)) →
  (Finset.choose (Finset.range 10) 3).card -
  ((Finset.choose (Finset.range 3) 3).card + 
   (Finset.choose (Finset.Ico 3 6) 3).card + 
   (Finset.choose (Finset.Ico 6 10) 3).card) = 114 :=
begin
  -- Proof goes here
  sorry
end

end selecting_televisions_l487_487152


namespace sum_reciprocals_factors_12_l487_487935

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487935


namespace percentage_of_crows_among_non_pigeons_l487_487025

theorem percentage_of_crows_among_non_pigeons:
  ∀ (total_birds sparrows pigeons crows parrots : ℕ),
    sparrows = 40 * total_birds / 100 →
    pigeons = 20 * total_birds / 100 →
    crows = 15 * total_birds / 100 →
    parrots = total_birds - sparrows - pigeons - crows →
    (100 * crows / (total_birds - pigeons)) = 18.75 :=
by
  intros total_birds sparrows pigeons crows parrots hsparrows hpigeons hcrows hparrots
  sorry

end percentage_of_crows_among_non_pigeons_l487_487025


namespace digits_difference_l487_487132

theorem digits_difference (X Y : ℕ) (h : 10 * X + Y - (10 * Y + X) = 90) : X - Y = 10 :=
by
  sorry

end digits_difference_l487_487132


namespace raine_steps_l487_487112

theorem raine_steps (steps_per_trip : ℕ) (num_days : ℕ) (total_steps : ℕ) : 
  steps_per_trip = 150 → 
  num_days = 5 → 
  total_steps = steps_per_trip * 2 * num_days → 
  total_steps = 1500 := 
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end raine_steps_l487_487112


namespace area_parallelogram_proof_l487_487559

/-- We are given a rectangle with a length of 10 cm and a width of 8 cm.
    We transform it into a parallelogram with a height of 9 cm.
    We need to prove that the area of the parallelogram is 72 square centimeters. -/
def area_of_parallelogram_from_rectangle (length width height : ℝ) : ℝ :=
  width * height

theorem area_parallelogram_proof
  (length width height : ℝ)
  (h_length : length = 10)
  (h_width : width = 8)
  (h_height : height = 9) :
  area_of_parallelogram_from_rectangle length width height = 72 :=
by
  sorry

end area_parallelogram_proof_l487_487559


namespace sum_reciprocals_factors_12_l487_487751

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487751


namespace cistern_filling_time_l487_487213

theorem cistern_filling_time (F_time E_time : ℝ) (hF : F_time = 2) (hE : E_time = 4) : 
  let net_rate := (1 / F_time) - (1 / E_time) in 
  let fill_time := 1 / net_rate in
  fill_time = 4 :=
by
  sorry

end cistern_filling_time_l487_487213


namespace sum_reciprocal_factors_12_l487_487867

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487867


namespace smaller_factor_of_4851_l487_487144

-- Define the condition
def product_lim (m n : ℕ) : Prop := m * n = 4851 ∧ 10 ≤ m ∧ m < 100 ∧ 10 ≤ n ∧ n < 100

-- The lean theorem statement
theorem smaller_factor_of_4851 : ∃ m n : ℕ, product_lim m n ∧ m = 49 := 
by {
    sorry
}

end smaller_factor_of_4851_l487_487144


namespace sum_reciprocal_factors_of_12_l487_487601

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487601


namespace pedal_triangle_similarity_l487_487538

variables {A B C P A1 B1 C1 A2 B2 C2 : Type*} [fintype P]

-- Definitions of a pedal triangle and circumcircle intersections
def is_pedal_triangle (P: Type*) (ABC: set P) (A1 B1 C1 : P) : Prop :=
  ∀ A B C : P, P ∈ ABC → (point.projection_perpendicular P A ∈ A1) ∧
                                (point.projection_perpendicular P B ∈ B1) ∧
                                (point.projection_perpendicular P C ∈ C1)

def intersects_circumcircle (triangle: set P) (lines_intersect: list P) : Prop :=
  ∀ A B C : P, (A ∈ triangle) ∧ (B ∈ triangle) ∧ (C ∈ triangle) →
    (∃ A2 B2 C2 : P, (line.through P A ∩ circumcircle(triangle) = A2) ∧
                          (line.through P B ∩ circumcircle(triangle) = B2) ∧
                          (line.through P C ∩ circumcircle(triangle) = C2))

-- Problem statement: proving the similarity of two triangles under given conditions
theorem pedal_triangle_similarity 
  (ABC : set P) (P : Type*) [triangle_ABC : triangle ABC] 
  (pedal_triangle : is_pedal_triangle P ABC A1 B1 C1)
  (circumcircle_intersections : intersects_circumcircle ABC [A2, B2, C2]) :
  similar_triangles A1 B1 C1 A2 B2 C2 :=
sorry

end pedal_triangle_similarity_l487_487538


namespace sum_of_reciprocals_factors_12_l487_487818

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487818


namespace part1_reduced_sales_volume_part1_reduced_profit_part2_price_reduction_for_2100_part3_cannot_achieve_2200_l487_487232

variables (initial_profit_per_item : ℕ) (initial_sales_volume : ℕ) (price_reduction_per_item : ℕ) (additional_sales_per_reduction : ℕ)

-- Conditions given in the problem
def conditions : Prop := 
  initial_profit_per_item = 50 ∧ initial_sales_volume = 30 ∧ price_reduction_per_item = 1 ∧ additional_sales_per_reduction = 2

-- Part 1: Sales volume and profit after $5 reduction
def reduced_price_5_sales_volume : Prop := 
  (initial_sales_volume + additional_sales_per_reduction * 5 = 40)

def reduced_price_5_profit : Prop := 
  (initial_profit_per_item - 5) * (initial_sales_volume + additional_sales_per_reduction * 5) = 1800

-- Part 2: Price reduction needed for $2100 profit
def price_reduction_for_2100_profit : Prop := 
  ∃ x : ℕ, (initial_profit_per_item - x) * (initial_sales_volume + additional_sales_per_reduction * x) = 2100

-- Part 3: Possibility of achieving $2200 profit
def can_achieve_2200_profit : Prop := 
  ¬(∃ x : ℕ, (initial_profit_per_item - x) * (initial_sales_volume + additional_sales_per_reduction * x) = 2200)

-- Main theorem statements
theorem part1_reduced_sales_volume (h : conditions) : reduced_price_5_sales_volume := sorry

theorem part1_reduced_profit (h : conditions) : reduced_price_5_profit := sorry

theorem part2_price_reduction_for_2100 (h : conditions) : price_reduction_for_2100_profit := sorry

theorem part3_cannot_achieve_2200 (h : conditions) : can_achieve_2200_profit := sorry

end part1_reduced_sales_volume_part1_reduced_profit_part2_price_reduction_for_2100_part3_cannot_achieve_2200_l487_487232


namespace angle_in_third_quadrant_half_l487_487397

theorem angle_in_third_quadrant_half {
  k : ℤ 
} (h1: (k * 360 + 180) < α) (h2 : α < k * 360 + 270) :
  (k * 180 + 90) < (α / 2) ∧ (α / 2) < (k * 180 + 135) :=
sorry

end angle_in_third_quadrant_half_l487_487397


namespace product_of_fractions_l487_487174

theorem product_of_fractions :
  (2 / 3) * (5 / 8) * (1 / 4) = 5 / 48 := by
  sorry

end product_of_fractions_l487_487174


namespace sum_reciprocals_of_factors_of_12_l487_487961

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487961


namespace probability_reaching_five_without_returning_to_zero_l487_487046

def reach_position_without_return_condition (tosses : ℕ) (target : ℤ) (return_limit : ℤ) : ℕ :=
  -- Ideally we should implement the logic to find the number of valid paths here (as per problem constraints)
  sorry

theorem probability_reaching_five_without_returning_to_zero {a b : ℕ} (h_rel_prime : Nat.gcd a b = 1)
    (h_paths_valid : reach_position_without_return_condition 10 5 3 = 15) :
    a = 15 ∧ b = 256 ∧ a + b = 271 :=
by
  sorry

end probability_reaching_five_without_returning_to_zero_l487_487046


namespace sum_reciprocals_factors_12_l487_487707

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487707


namespace roots_cubic_eq_l487_487001

theorem roots_cubic_eq (r s p q : ℝ) (h1 : r + s = p) (h2 : r * s = q) :
    r^3 + s^3 = p^3 - 3 * q * p :=
by
    -- Placeholder for proof
    sorry

end roots_cubic_eq_l487_487001


namespace largest_prime_factor_3750_l487_487170

theorem largest_prime_factor_3750 : 
  ∃ p, nat.prime p ∧ ∀ q, nat.prime q ∧ q ∣ 3750 → q ≤ 5  :=
sorry

end largest_prime_factor_3750_l487_487170


namespace sugar_bought_l487_487543

noncomputable def P : ℝ := 0.50
noncomputable def S : ℝ := 2.0

theorem sugar_bought : 
  (1.50 * S + 5 * P = 5.50) ∧ 
  (3 * 1.50 + P = 5) ∧
  ((1.50 : ℝ) ≠ 0) → (S = 2) :=
by
  sorry

end sugar_bought_l487_487543


namespace probability_correct_l487_487988

def set_D : set ℕ := {1, 2, 4, 5, 10, 20, 25, 50, 100}
def set_Z : set ℤ := {i | 1 ≤ i ∧ i ≤ 100}

noncomputable def probability_d_divides_z : ℚ :=
  (1/9) * (1 + 0.5 + 0.25 + 0.2 + 0.1 + 0.05 + 0.04 + 0.02 + 0.01)

theorem probability_correct : 
  probability_d_divides_z = 217 / 900 :=
by
  sorry

end probability_correct_l487_487988


namespace Sue_made_22_buttons_l487_487448

-- Definitions of the conditions and the goal
def Mari_buttons : ℕ := 8
def Kendra_buttons := 4 + 5 * Mari_buttons
def Sue_buttons := Kendra_buttons / 2

-- Theorem statement
theorem Sue_made_22_buttons : Sue_buttons = 22 :=
by
  -- Definitions used here
  unfold Mari_buttons Kendra_buttons Sue_buttons
  -- Calculation
  rw [show 4 + 5 * 8 = 44, by norm_num, show 44 / 2 = 22, by norm_num]
  rfl

end Sue_made_22_buttons_l487_487448


namespace sum_reciprocals_12_l487_487918

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487918


namespace sum_of_reciprocals_of_factors_of_12_l487_487799

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487799


namespace decimal_to_binary_93_l487_487266

theorem decimal_to_binary_93 : nat.binary_repr 93 = "1011101" :=
by
  -- Ensure that the standard library for nat (natural numbers) with the binary representation function is available
  sorry -- Skipping the proof part as requested.

end decimal_to_binary_93_l487_487266


namespace sum_reciprocals_12_l487_487933

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487933


namespace sum_reciprocals_factors_12_l487_487640

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487640


namespace sum_reciprocals_factors_12_l487_487619

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487619


namespace sum_reciprocals_factors_12_l487_487627

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487627


namespace quadrilateral_is_square_l487_487516

-- Definitions for conditions
def is_cyclic_quadrilateral (A B C D O : Type _) [MetricSpace Type _] : Prop :=
  ∃ (AB : ℝ) (BC : ℝ) (CD : ℝ) (DA : ℝ) (AC : ℝ) (BD : ℝ),
  Area 
    = 8 ∧
  OA + OB + OC + OD 
    = 8

-- Statement of the theorem
theorem quadrilateral_is_square 
  (A B C D O : Type _) 
  [MetricSpace Type _] 
  (h_cyclic : is_cyclic_quadrilateral A B C D O) 
  : is_square A B C D :=
sorry

end quadrilateral_is_square_l487_487516


namespace Yoque_monthly_payment_l487_487987

theorem Yoque_monthly_payment :
  ∃ m : ℝ, m = 15 ∧ ∀ a t : ℝ, a = 150 ∧ t = 11 ∧ (a + 0.10 * a) / t = m :=
by
  sorry

end Yoque_monthly_payment_l487_487987


namespace parabola_points_l487_487522

noncomputable def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_points :
  ∃ (a c m n : ℝ),
  a = 2 ∧ c = -2 ∧
  parabola a 1 c 2 = m ∧
  parabola a 1 c n = -2 ∧
  m = 8 ∧
  n = -1 / 2 :=
by
  use 2, -2, 8, -1/2
  simp [parabola]
  sorry

end parabola_points_l487_487522


namespace direct_proportion_solution_l487_487318

theorem direct_proportion_solution (m : ℝ) (h1 : m + 3 ≠ 0) (h2 : m^2 - 8 = 1) : m = 3 :=
sorry

end direct_proportion_solution_l487_487318


namespace sum_reciprocals_factors_12_l487_487951

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487951


namespace jill_sod_area_l487_487050

noncomputable def area_of_sod (yard_width yard_length sidewalk_width sidewalk_length flower_bed1_depth flower_bed1_length flower_bed2_depth flower_bed2_length flower_bed3_width flower_bed3_length flower_bed4_width flower_bed4_length : ℝ) : ℝ :=
  let yard_area := yard_width * yard_length
  let sidewalk_area := sidewalk_width * sidewalk_length
  let flower_bed1_area := flower_bed1_depth * flower_bed1_length
  let flower_bed2_area := flower_bed2_depth * flower_bed2_length
  let flower_bed3_area := flower_bed3_width * flower_bed3_length
  let flower_bed4_area := flower_bed4_width * flower_bed4_length
  let total_non_sod_area := sidewalk_area + 2 * flower_bed1_area + flower_bed2_area + flower_bed3_area + flower_bed4_area
  yard_area - total_non_sod_area

theorem jill_sod_area : 
  area_of_sod 200 50 3 50 4 25 4 25 10 12 7 8 = 9474 := by sorry

end jill_sod_area_l487_487050


namespace james_milk_left_l487_487045

@[simp] def ounces_in_gallon : ℕ := 128
@[simp] def gallons_james_has : ℕ := 3
@[simp] def ounces_drank : ℕ := 13

theorem james_milk_left :
  (gallons_james_has * ounces_in_gallon - ounces_drank) = 371 :=
by
  sorry

end james_milk_left_l487_487045


namespace sum_reciprocals_12_l487_487921

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487921


namespace sum_reciprocals_factors_of_12_l487_487686

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487686


namespace cooking_people_count_l487_487023

variables (P Y W : ℕ)

def people_practicing_yoga := 25
def people_studying_weaving := 8
def people_studying_only_cooking := 2
def people_studying_cooking_and_yoga := 7
def people_studying_cooking_and_weaving := 3
def people_studying_all_curriculums := 3

theorem cooking_people_count :
  P = people_studying_only_cooking + (people_studying_cooking_and_yoga - people_studying_all_curriculums)
    + (people_studying_cooking_and_weaving - people_studying_all_curriculums) + people_studying_all_curriculums →
  P = 9 :=
by
  intro h
  unfold people_studying_only_cooking people_studying_cooking_and_yoga people_studying_cooking_and_weaving people_studying_all_curriculums at h
  sorry

end cooking_people_count_l487_487023


namespace determine_a_for_line_l487_487272

theorem determine_a_for_line (a : ℝ) (h : a ≠ 0)
  (intercept_condition : ∃ (k : ℝ), 
    ∀ x y : ℝ, (a * x - 6 * y - 12 * a = 0) → (x = 12) ∧ (y = 2 * a * x / 6) ∧ (12 = 3 * (-2 * a))) : 
  a = -2 :=
by
  sorry

end determine_a_for_line_l487_487272


namespace minimize_tetrahedron_volume_l487_487239

/-- Let A be a vertex of a parallelepiped, with adjacent vertices B, C, and D, and the vertex opposite to A be E. Let a plane through E intersect the half-lines AB, AC, and AD at points P, Q, and R respectively. We want to show that the volume of the tetrahedron APQR is minimized if and only if AP = 3 * AB, AQ = 3 * AC, and AR = 3 * AD. --/
theorem minimize_tetrahedron_volume (A B C D E P Q R : Point)
  (AB : Segment A B) (AC : Segment A C) (AD : Segment A D)
  (intersection_P : Intersect (Line A E) (Plane E P Q R) P)
  (intersection_Q : Intersect (Line A E) (Plane E P Q R) Q)
  (intersection_R : Intersect (Line A E) (Plane E P Q R) R) :
  (∃ k : ℝ, k > 0 ∧ P = k • B ∧ Q = k • C ∧ R = k • D) ↔ 
  AP = 3 * AB ∧ AQ = 3 * AC ∧ AR = 3 * AD :=
sorry

end minimize_tetrahedron_volume_l487_487239


namespace total_money_l487_487402

namespace MoneyProof

variables (B J T : ℕ)

-- Given conditions
def condition_beth : Prop := B + 35 = 105
def condition_jan : Prop := J - 10 = B
def condition_tom : Prop := T = 3 * (J - 10)

-- Proof that the total money is $360
theorem total_money (h1 : condition_beth B) (h2 : condition_jan B J) (h3 : condition_tom J T) :
  B + J + T = 360 :=
by
  sorry

end MoneyProof

end total_money_l487_487402


namespace mrs_hilt_money_left_l487_487484

theorem mrs_hilt_money_left (initial_money : ℕ) (cost_of_pencil : ℕ) (money_left : ℕ) (h1 : initial_money = 15) (h2 : cost_of_pencil = 11) : money_left = 4 :=
by
  sorry

end mrs_hilt_money_left_l487_487484


namespace sum_reciprocals_of_factors_12_l487_487846

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487846


namespace range_of_a_l487_487012

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a-1)*x + 1 ≤ 0) → (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l487_487012


namespace repair_time_calculation_l487_487401

-- Assume amount of work is represented as units
def work_10_people_45_minutes := 10 * 45
def work_20_people_20_minutes := 20 * 20

-- Assuming the flood destroys 2 units per minute as calculated in the solution
def flood_rate := 2

-- Calculate total initial units of the dike
def dike_initial_units :=
  work_10_people_45_minutes - flood_rate * 45

-- Given 14 people are repairing the dam
def repair_rate_14_people := 14 - flood_rate

-- Statement to prove that 14 people need 30 minutes to repair the dam
theorem repair_time_calculation :
  dike_initial_units / repair_rate_14_people = 30 :=
by
  sorry

end repair_time_calculation_l487_487401


namespace distinct_pawns_placement_l487_487321

/-- The number of ways to place three distinct pawns on a 3x3 chess board 
     such that no row or column contains more than one pawn is 36. -/
theorem distinct_pawns_placement : 
  let rows := 3 in
  let cols := 3 in
  ∃ ways : ℕ, ways = (rows * (rows - 1) * (rows - 2)) * nat.factorial rows ∧ ways = 36 :=
sorry

end distinct_pawns_placement_l487_487321


namespace sum_reciprocals_factors_12_l487_487653

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487653


namespace sum_reciprocal_factors_of_12_l487_487597

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487597


namespace laptop_selection_l487_487305

open Nat

theorem laptop_selection :
  ∃ (n : ℕ), n = (choose 4 2) * (choose 5 1) + (choose 4 1) * (choose 5 2) := 
sorry

end laptop_selection_l487_487305


namespace rhombus_area_correct_l487_487995

noncomputable def rhombus_area (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / 2

theorem rhombus_area_correct :
  rhombus_area 30 12 = 180 :=
by
  sorry

end rhombus_area_correct_l487_487995


namespace sum_reciprocals_of_factors_12_l487_487837

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487837


namespace max_value_norm_sub_vec_l487_487309

open Real

theorem max_value_norm_sub_vec
  (x : ℝ)
  (m : ℝ × ℝ := (cos (x / 2), sin (x / 2)))
  (n : ℝ × ℝ := (- sqrt 3, 1)) :
  ∃ k : ℤ, (x / 2 + π / 6 = 2 * π * k) → (sqrt ((m.1 + sqrt 3) ^ 2 + (m.2 - 1)^2) = 3) :=
begin
  sorry
end

end max_value_norm_sub_vec_l487_487309


namespace steps_in_five_days_l487_487110

def steps_to_school : ℕ := 150
def daily_steps : ℕ := steps_to_school * 2
def days : ℕ := 5

theorem steps_in_five_days : daily_steps * days = 1500 := by
  sorry

end steps_in_five_days_l487_487110


namespace N_plus_15n_l487_487498

variables (x y z : ℝ)

def A := x + y + z
def B := x^2 + y^2 + z^2
def C := x * y + x * z + y * z

theorem N_plus_15n : (3 * A = B) → 
  (let N := max C N, n := min C n in N + 15 * n = 18) :=
by {
  sorry
}

end N_plus_15n_l487_487498


namespace sum_of_reciprocals_of_factors_of_12_l487_487760

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487760


namespace sum_reciprocals_of_factors_of_12_l487_487956

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487956


namespace centers_form_ellipse_or_hyperbola_l487_487108

/-- 
Given a point \( A \) and a circle \( S \) with center \( O \) and radius \( R \), 
prove that the set of all centers of circles passing through \( A \) and tangent to \( S \) 
(not containing \( A \)) forms an ellipse or a hyperbola.
-/
theorem centers_form_ellipse_or_hyperbola
  (A O C : ℝ × ℝ) (R : ℝ)
  (hR : 0 < R) (hAC : (A ≠ O) ∧ (A ≠ C) ∧ (O ≠ C)) :
  ∃ F₁ F₂ : ℝ × ℝ, 
    let d1 := sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) in
    let d2 := |sqrt ((C.1 - O.1)^2 + (C.2 - O.2)^2) - R| in
    (d1 = d2 + R) ∨ (d1 = d2 - R) := sorry

end centers_form_ellipse_or_hyperbola_l487_487108


namespace sum_reciprocals_of_factors_of_12_l487_487957

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487957


namespace smaller_number_l487_487142

theorem smaller_number (a b : ℕ) (h1 : 10 ≤ a ∧ a < 100) (h2 : 10 ≤ b ∧ b < 100) (h3 : a * b = 4851) : min a b = 53 :=
sorry

end smaller_number_l487_487142


namespace max_points_of_intersection_l487_487574

theorem max_points_of_intersection (circles : Fin 2 → Circle) (lines : Fin 3 → Line) :
  number_of_intersections circles lines = 17 :=
sorry

end max_points_of_intersection_l487_487574


namespace ratio_x_y_l487_487591

theorem ratio_x_y (x y : ℤ) (h : (8 * x - 5 * y) * 3 = (11 * x - 3 * y) * 2) :
  x / y = 9 / 2 := by
  sorry

end ratio_x_y_l487_487591


namespace sum_reciprocals_factors_of_12_l487_487675

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487675


namespace sum_reciprocals_factors_of_12_l487_487693

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487693


namespace divisor_of_1053_added_with_5_is_2_l487_487172

theorem divisor_of_1053_added_with_5_is_2 :
  ∃ d : ℕ, d > 1 ∧ ∀ (x : ℝ), x = 5.000000000000043 → (1053 + x) % d = 0 → d = 2 :=
by
  sorry

end divisor_of_1053_added_with_5_is_2_l487_487172


namespace percent_calculation_l487_487979

theorem percent_calculation (x : ℝ) : 
  (∃ y : ℝ, y / 100 * x = 0.3 * 0.7 * x) → ∃ y : ℝ, y = 21 :=
by
  sorry

end percent_calculation_l487_487979


namespace sum_reciprocals_factors_12_l487_487646

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487646


namespace simplify_and_evaluate_l487_487505

def my_expression (x : ℝ) := (x + 2) * (x - 2) + 3 * (1 - x)

theorem simplify_and_evaluate : 
  my_expression (Real.sqrt 2) = 1 - 3 * Real.sqrt 2 := by
    sorry

end simplify_and_evaluate_l487_487505


namespace max_points_of_intersection_l487_487590

theorem max_points_of_intersection (L1 L2 L3 : set ℝ × ℝ) (C : set ℝ × ℝ) :
  (∀ P ∈ C, (P ∉ L1 ∧ P ∉ L2 ∧ P ∉ L3) ∨ (P ∈ L1 ∧ P ∈ L2 ∧ P ∈ L3) ∨
   ((!P ∈ L1) ∧ (P ∉ L2 ∧ P ∉ L3) ∨ (P ∉ L1 ∧ P ∉ L3) ∨ (P ∉ L1 ∧ P ∉ L2))) ∧
  (set.card (L1 ∩ C) ≤ 2) ∧ (set.card (L2 ∩ C) ≤ 2) ∧ (set.card (L3 ∩ C) ≤ 2) ∧
  (set.card (L1 ∩ L2 ∩ L3) = 1) ∧
  (∀ i j (h : i ≠ j), (set.card (L1 ∩ L2) ≤ 1 ∧ set.card (L1 ∩ L3) ≤ 1 ∧
                          set.card (L2 ∩ L3) ≤ 1)) →
  ∃ n, n = 9 :=
begin
  sorry
end

end max_points_of_intersection_l487_487590


namespace train_speed_l487_487990

-- Lean statement
theorem train_speed (distance time : ℝ) (h1 : distance = 160) (h2 : time = 9) : (distance / time).round(2) = 17.78 :=
by
  -- Proof is to be filled in
  sorry

end train_speed_l487_487990


namespace linear_equations_not_always_solvable_l487_487249

theorem linear_equations_not_always_solvable 
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : 
  ¬(∀ x y : ℝ, (a₁ * x + b₁ * y = c₁ ∧ a₂ * x + b₂ * y = c₂) ↔ 
                   a₁ * b₂ - a₂ * b₁ ≠ 0) :=
sorry

end linear_equations_not_always_solvable_l487_487249


namespace perimeter_of_square_land_is_36_diagonal_of_square_land_is_27_33_l487_487514

def square_land (A P D : ℝ) :=
  (5 * A = 10 * P + 45) ∧
  (3 * D = 2 * P + 10)

theorem perimeter_of_square_land_is_36 (A P D : ℝ) (h1 : 5 * A = 10 * P + 45) (h2 : 3 * D = 2 * P + 10) :
  P = 36 :=
sorry

theorem diagonal_of_square_land_is_27_33 (A P D : ℝ) (h1 : P = 36) (h2 : 3 * D = 2 * P + 10) :
  D = 82 / 3 :=
sorry

end perimeter_of_square_land_is_36_diagonal_of_square_land_is_27_33_l487_487514


namespace find_S5_l487_487458

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {q : ℝ}

-- conditions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def positive_terms (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (finset.range n).sum a

axiom a2a4_eq_1 : a 1 * a 3 = 1
axiom S3_eq_7 : S 3 = 7

-- proof goal
theorem find_S5 (hq_pos : q > 0) (geo_seq : geometric_sequence a) (pos_terms : positive_terms a)
  (sum_terms : sum_of_first_n_terms a S) : S 5 = 31 / 4 :=
begin
  -- Proof steps would go here
  sorry
end

end find_S5_l487_487458


namespace sum_reciprocals_factors_12_l487_487642

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487642


namespace sum_reciprocals_factors_of_12_l487_487682

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487682


namespace find_number_l487_487180

theorem find_number (x : ℝ) : (x / 2 = x - 5) → x = 10 :=
by
  intro h
  sorry

end find_number_l487_487180


namespace sum_reciprocals_factors_12_l487_487735

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487735


namespace ratio_of_areas_l487_487562

theorem ratio_of_areas (OR : ℝ) (h : OR > 0) :
  let OY := (1 / 3) * OR
  let area_OY := π * OY^2
  let area_OR := π * OR^2
  (area_OY / area_OR) = (1 / 9) :=
by
  -- Definitions
  let OY := (1 / 3) * OR
  let area_OY := π * OY^2
  let area_OR := π * OR^2
  sorry

end ratio_of_areas_l487_487562


namespace triangle_area_given_conditions_l487_487089

-- Definitions of the given conditions
structure Point (α : Type) :=
  (x : α) (y : α)

def Triangle (α : Type) := {a b c : Point α // (c.y = a.y) ∧ (c.x = b.x)}

def isRightTriangle {α : Type} [Field α] (T : Triangle α) : Prop :=
  T.val.c.y = 0 ∧ T.val.c.x = 0

def hypotenuse_length {α : Type} [Field α] (T : Triangle α) : α :=
  let (P, Q, R) := (T.val.a, T.val.b, T.val.c)
  real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

def median_line_eq (P : Point ℝ) (slope offset : ℝ) : Prop :=
  ∀ m : ℝ, m = P.y / P.x → m = slope * P.x + offset

noncomputable def triangle_area {α : Type} [Field α] (T : Triangle α) : α :=
  let (P, Q, R) := (T.val.a, T.val.b, T.val.c)
  real.abs((P.x * (Q.y - R.y) + Q.x * (R.y - P.y) + R.x * (P.y - Q.y)) / 2)

-- Statement of the proof problem in Lean 4
theorem triangle_area_given_conditions :
  ∀ (P Q : Point ℝ) (R : Point ℝ),
    isRightTriangle ⟨P, Q, R, ⟨rfl, rfl⟩⟩
    → hypotenuse_length ⟨P, Q, R, ⟨rfl, rfl⟩⟩ = 50
    → median_line_eq P 1 (-2)
    → median_line_eq Q 3 (-5)
    → triangle_area ⟨P, Q, R, ⟨rfl, rfl⟩⟩ = 250 :=
by
  intro P Q R h_right h_length h_med_p h_med_q
  -- Proof goes here
  sorry

end triangle_area_given_conditions_l487_487089


namespace sum_reciprocals_factors_12_l487_487937

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487937


namespace sum_reciprocal_factors_of_12_l487_487596

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487596


namespace geom_progression_no_equal_sum_l487_487323

variable {a q : ℤ} (n m : ℕ) (k : fin (m + 1) → ℕ)

theorem geom_progression_no_equal_sum 
  (hq : q ≠ 0) (hq1 : q ≠ 1) (hq_neg1 : q ≠ -1) 
  (ha : a ≠ 0) 
  (distinct_k : ∀ i j, i ≠ j → k i ≠ k j) :
  (a * q ^ k 0 + a * q ^ k 1 + ... + a * q ^ k m ≠ a * q ^ k (m + 1)) := 
  sorry

end geom_progression_no_equal_sum_l487_487323


namespace math_problem_l487_487361

noncomputable def given_eq (a b : ℝ) : ℝ :=
    (ℚ.mk 2334 1000) * A - 
    sqrt (a^3 - b^3 + sqrt (a)) * 
    (sqrt (a^(3/2) + sqrt (b^3 + sqrt (a))) * sqrt (a^(3/2) - sqrt (b^3 + sqrt (a)))) /
    sqrt ((a^3 + b^3)^2 - a * (4 * a^2 * b^3 + 1))

theorem math_problem (a b : ℝ) (h₀ : a > 0) 
    (h₁ : -a^(1/6) <= b) 
    (h₂ : b < (a^3 - sqrt a)^(1/3)) 
    (h₃ : b^3 + sqrt a >= 0) 
    (h₄ : a^3 - b^3 > sqrt a) :
    given_eq a b = 0 :=
sorry

end math_problem_l487_487361


namespace range_of_g_l487_487462

def f (x : ℝ) := 4 * x - 5

def g (x : ℝ) := f (f (f x))

theorem range_of_g :
  ∃ y_min y_max, (∀ x, 1 ≤ x ∧ x ≤ 3 → y_min ≤ g x ∧ g x ≤ y_max) ∧ y_min = -41 ∧ y_max = 87 :=
by
  have h1 : g 1 = -41 := by sorry
  have h2 : g 3 = 87 := by sorry
  use [-41, 87]
  constructor
  · intros x hx
    cases hx
    sorry
  constructor
  · exact h1
  · exact h2

end range_of_g_l487_487462


namespace sum_reciprocals_factors_12_l487_487934

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487934


namespace quadrilateral_angles_combinations_pentagon_angles_combination_l487_487565

-- Define angle types
inductive AngleType
| acute
| right
| obtuse

open AngleType

-- Define predicates for sum of angles in a quadrilateral and pentagon
def quadrilateral_sum (angles : List AngleType) : Bool :=
  match angles with
  | [right, right, right, right] => true
  | [right, right, acute, obtuse] => true
  | [right, acute, obtuse, obtuse] => true
  | [right, acute, acute, obtuse] => true
  | [acute, obtuse, obtuse, obtuse] => true
  | [acute, acute, obtuse, obtuse] => true
  | [acute, acute, acute, obtuse] => true
  | _ => false

def pentagon_sum (angles : List AngleType) : Prop :=
  -- Broad statement, more complex combinations possible
  ∃ a b c d e : ℕ, (a + b + c + d + e = 540) ∧
    (a < 90 ∨ a = 90 ∨ a > 90) ∧
    (b < 90 ∨ b = 90 ∨ b > 90) ∧
    (c < 90 ∨ c = 90 ∨ c > 90) ∧
    (d < 90 ∨ d = 90 ∨ d > 90) ∧
    (e < 90 ∨ e = 90 ∨ e > 90)

-- Prove the possible combinations for a quadrilateral and a pentagon
theorem quadrilateral_angles_combinations {angles : List AngleType} :
  quadrilateral_sum angles = true :=
sorry

theorem pentagon_angles_combination :
  ∃ angles : List AngleType, pentagon_sum angles :=
sorry

end quadrilateral_angles_combinations_pentagon_angles_combination_l487_487565


namespace sum_reciprocals_factors_of_12_l487_487687

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487687


namespace part_one_part_two_l487_487088

theorem part_one (a : ℕ) (h₁ : 3 / 2 ∈ {x : ℚ | |x + 1| > a}) (h₂ : 1 / 2 ∉ {x : ℚ | |x + 1| > a}) : a = 2 :=
sorry

theorem part_two (m n s : ℝ) (hm : m > 0) (hn : n > 0) (hs : s > 0) (h : m + n + sqrt 2 * s = 2) : 
  m^2 + n^2 + s^2 = 1 :=
sorry

end part_one_part_two_l487_487088


namespace sum_reciprocals_factors_12_l487_487747

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487747


namespace count_even_decreasing_digits_correct_l487_487390

def digits_in_decreasing_order (a b c : ℕ) : Prop :=
a > b ∧ b > c ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b < 10 ∧ 0 ≤ c ∧ c < 10

def is_even_digit (c : ℕ) : Prop :=
c ∈ {2, 4, 6, 8}

def count_even_decreasing_digits : ℕ :=
34

theorem count_even_decreasing_digits_correct :
  ∃ n : ℕ, (∀ a b c : ℕ, digits_in_decreasing_order a b c → is_even_digit c → 100 ≤ a * 100 + b * 10 + c ∧ a * 100 + b * 10 + c ≤ 999) ∧ n = count_even_decreasing_digits  :=
sorry

end count_even_decreasing_digits_correct_l487_487390


namespace sum_reciprocals_factors_12_l487_487711

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487711


namespace sum_is_zero_l487_487081

noncomputable def z : ℂ := Complex.cos (3 * Real.pi / 8) + Complex.sin (3 * Real.pi / 8) * Complex.I

theorem sum_is_zero (hz : z^8 = 1) (hz1 : z ≠ 1) :
  (z / (1 + z^3)) + (z^2 / (1 + z^6)) + (z^4 / (1 + z^12)) = 0 :=
by
  sorry

end sum_is_zero_l487_487081


namespace sum_of_reciprocals_factors_12_l487_487791

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487791


namespace sum_of_reciprocals_of_factors_of_12_l487_487800

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487800


namespace distributeStops_l487_487416

-- Define a predicate for the bus routes network
structure BusNetwork where
  stops : Type
  routes : Set (Set stops)
  condition_1 : ∀ r1 r2 ∈ routes, r1 ≠ r2 → ∃! s, s ∈ r1 ∧ s ∈ r2
  condition_2 : ∀ r ∈ routes, (Set.card r) ≥ 4

-- Main theorem to be proven
theorem distributeStops (network : BusNetwork) : ∃ company : network.stops → bool, ∀ r ∈ network.routes, ∃ s1 s2 ∈ r, company s1 ≠ company s2 := 
by 
  sorry

end distributeStops_l487_487416


namespace sum_reciprocal_factors_of_12_l487_487612

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487612


namespace sum_reciprocals_factors_12_l487_487746

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487746


namespace subset_size_condition_l487_487200

open Set

-- Define the nature of expressions as tuples within a specified range
def Expression (n : ℕ) (k : ℕ) := Fin n → Fin (k + 1)

-- Main theorem statement
theorem subset_size_condition (n k : ℕ) (h_k : 1 < k)
  (P Q : Finset (Expression n k))
  (h : ∀ p ∈ P, ∀ q ∈ Q, ∃ m, p m = q m) :
  P.card ≤ k^(n - 1) ∨ Q.card ≤ k^(n - 1) :=
  by sorry

end subset_size_condition_l487_487200


namespace find_height_l487_487515

-- Define variables for the problem
variables (base height : ℝ) (area : ℝ)

-- Given conditions
def area_triangle (base height : ℝ) : ℝ := (base * height) / 2
axiom given_base : base = 4.5
axiom given_area : area = 13.5

-- Goal
theorem find_height : height = 6 :=
by 
  have h1 : 13.5 = (4.5 * height) / 2 := by rw [given_base, given_area, area_triangle]
  have h2 : 27 = 4.5 * height := by linarith
  sorry

end find_height_l487_487515


namespace sum_reciprocals_factors_12_l487_487944

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487944


namespace ratio_sqrt_2_l487_487435

theorem ratio_sqrt_2 {a b : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : a^2 + b^2 = 6 * a * b) :
  (a + b) / (a - b) = Real.sqrt 2 :=
by
  sorry

end ratio_sqrt_2_l487_487435


namespace num_points_P_on_ellipse_l487_487536

noncomputable def ellipse : Set (ℝ × ℝ) := {p | (p.1)^2 / 16 + (p.2)^2 / 9 = 1}
noncomputable def line : Set (ℝ × ℝ) := {p | p.1 / 4 + p.2 / 3 = 1}
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := 
  (1 / 2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

theorem num_points_P_on_ellipse (A B : ℝ × ℝ) 
  (hA_on_line : A ∈ line) (hA_on_ellipse : A ∈ ellipse) 
  (hB_on_line : B ∈ line) (hB_on_ellipse : B ∈ ellipse)
  : ∃ P1 P2 : ℝ × ℝ, P1 ∈ ellipse ∧ P2 ∈ ellipse ∧ 
    area_triangle A B P1 = 3 ∧ area_triangle A B P2 = 3 ∧ 
    P1 ≠ P2 ∧ 
    (∀ P : ℝ × ℝ, P ∈ ellipse ∧ area_triangle A B P = 3 → P = P1 ∨ P = P2) := 
sorry

end num_points_P_on_ellipse_l487_487536


namespace exists_a_satisfying_inequality_l487_487289

theorem exists_a_satisfying_inequality (x : ℝ) : 
  x < -2 ∨ (0 < x ∧ x < 1) ∨ 1 < x → 
  ∃ a ∈ Set.Icc (-1 : ℝ) 2, (2 - a) * x^3 + (1 - 2 * a) * x^2 - 6 * x + 5 + 4 * a - a^2 < 0 := 
by 
  intros h
  sorry

end exists_a_satisfying_inequality_l487_487289


namespace sum_reciprocals_factors_12_l487_487712

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487712


namespace smallest_diff_PR_PQ_l487_487560

theorem smallest_diff_PR_PQ (PQ PR QR : ℤ) (h1 : PQ < PR) (h2 : PR ≤ QR) (h3 : PQ + PR + QR = 2021) : 
  ∃ PQ PR QR : ℤ, PQ < PR ∧ PR ≤ QR ∧ PQ + PR + QR = 2021 ∧ PR - PQ = 1 :=
by
  sorry

end smallest_diff_PR_PQ_l487_487560


namespace sum_reciprocals_of_factors_12_l487_487844

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487844


namespace Mary_ends_with_31_eggs_l487_487482

theorem Mary_ends_with_31_eggs (a b : ℕ) (h1 : a = 27) (h2 : b = 4) : a + b = 31 := by
  sorry

end Mary_ends_with_31_eggs_l487_487482


namespace rope_length_fourth_cut_rope_length_nth_cut_l487_487229

theorem rope_length_fourth_cut :
  let initial_length := 1
  let cut_factor := 3
  let after_fourth_cut := (initial_length : ℚ) * (1 / cut_factor) ^ 4
  in after_fourth_cut = 1 / (cut_factor ^ 4) :=
by
  let initial_length := 1
  let cut_factor := 3
  let after_fourth_cut := (initial_length : ℚ) * (1 / cut_factor) ^ 4
  have h : after_fourth_cut = 1 / (cut_factor ^ 4) := by
    sorry
  exact h

theorem rope_length_nth_cut (n : ℕ) :
  let initial_length := 1
  let cut_factor := 3
  let after_nth_cut := (initial_length : ℚ) * (1 / cut_factor) ^ n
  in after_nth_cut = 1 / (cut_factor ^ n) :=
by
  let initial_length := 1
  let cut_factor := 3
  let after_nth_cut := (initial_length : ℚ) * (1 / cut_factor) ^ n
  have h : after_nth_cut = 1 / (cut_factor ^ n) := by
    sorry
  exact h

end rope_length_fourth_cut_rope_length_nth_cut_l487_487229


namespace sum_of_reciprocals_factors_12_l487_487823

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487823


namespace sum_reciprocals_factors_12_l487_487641

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487641


namespace problem_c_l487_487984

theorem problem_c (x y : ℝ) (h : x - 3 = y - 3): x - y = 0 :=
by
  sorry

end problem_c_l487_487984


namespace sum_reciprocal_factors_12_l487_487865

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487865


namespace sum_of_reciprocals_of_factors_of_12_l487_487902

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487902


namespace sum_of_reciprocals_of_factors_of_12_l487_487654

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487654


namespace sum_reciprocals_factors_12_l487_487750

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487750


namespace find_parabola_equation_find_line_CD_l487_487372

-- Definitions for the conditions
def parabola (p : ℝ) := ∀ x y, y^2 = 2 * p * x

def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

def on_parabola (p m : ℝ) (hx : m > 1) : Prop :=
  ∃ y, y = 2 ∧ y^2 = 2 * p * m

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- Proof statement 1: Proving the equation of the parabola
theorem find_parabola_equation (p : ℝ) (m : ℝ) 
  (hp : p > 0) (hm : m > 1) (A_on_parabola : on_parabola p m hm)
  (AF_eq : distance m 2 (focus p).fst (focus p).snd = 5 / 2) :
  ∃ x y, y^2 = 2 * x := sorry

-- Proof statement 2: Proving the equation of line CD
theorem find_line_CD (m : ℝ) (x1 y1 y2 : ℝ)
  (hx1 : x1 = -2) (hy1 : y1 = 0) (hx2 : x2 = 2) (hy2 : y2 = 0)
  (S_MCD_eq : triangle_area (-2) 0 x1 y1 x2 y2 = 16) :
  ∃ x y, x = ±2*√3*y + 2 := sorry

end find_parabola_equation_find_line_CD_l487_487372


namespace sum_reciprocals_of_factors_12_l487_487851

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487851


namespace sum_reciprocals_of_factors_12_l487_487853

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487853


namespace f_unique_and_n_eq_fp_l487_487077

noncomputable def f (n : ℕ) : ℕ := sorry

axiom f_prime (p : ℕ) (hp : Nat.Prime p) : f p = 1
axiom f_mul (a b : ℕ) : f (a * b) = a * f b + f a * b

open Nat

theorem f_unique_and_n_eq_fp (n : ℕ) (hp : Prime n) : (∃ p, n = p ^ p ∧ Prime p) := sorry

end f_unique_and_n_eq_fp_l487_487077


namespace rotation_270_deg_result_l487_487248

def rotate_complex_by_270_deg (z : Complex) : Complex :=
  -Complex.i * z

theorem rotation_270_deg_result :
    rotate_complex_by_270_deg (-7 - 4 * Complex.i) = -4 + 7 * Complex.i := 
by
  sorry

end rotation_270_deg_result_l487_487248


namespace upper_bound_t_l487_487005

variable (X : Type) (n r t : ℕ)
variables (Hk : finset (finset X))
variables (A : finset (finset X))
variables [fintype X]

/- Conditions -/
def satisfies_conditions (k r : ℕ) (A : finset (finset X)) : Prop :=
  (∃ t, A.card = t ∧ t ≤ nat.choose (n - 1) (r - 1)) ∧
  ∀ a ∈ A, a.card = r

theorem upper_bound_t (k r : ℕ) (A : finset (finset X))
  (h1 : k < r)
  (h2 : satisfies_conditions k r A) :
  A.card ≤ nat.choose (n - 1) (r - 1) :=
sorry

end upper_bound_t_l487_487005


namespace sum_of_reciprocals_of_factors_of_12_l487_487909

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487909


namespace sum_of_reciprocals_of_factors_of_12_l487_487658

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487658


namespace sum_reciprocals_factors_12_l487_487877

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487877


namespace sum_reciprocals_factors_12_l487_487882

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487882


namespace sum_of_reciprocals_factors_12_l487_487719

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487719


namespace sum_reciprocals_factors_12_l487_487709

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487709


namespace number_of_elements_in_intersection_l487_487341

open Set

-- Define set A and B
def SetA : Set ℤ := {x | x^2 + 2 * x - 3 ≤ 0}
def SetB : Set ℤ := {x | x ≥ -1}

-- Define the intersection between A and B
def SetIntersect : Set ℤ := SetA ∩ SetB

-- State the theorem about the number of elements in intersection
theorem number_of_elements_in_intersection : Set.card SetIntersect = 3 := 
    sorry      -- We are not providing the proof here

end number_of_elements_in_intersection_l487_487341


namespace sum_reciprocal_factors_12_l487_487855

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487855


namespace volume_displacement_of_cube_l487_487218

theorem volume_displacement_of_cube (radius_height_cube : ℝ× ℝ× ℝ× ℝ)
 (radius := radius_height_cube.1) 
 (height := radius_height_cube.2.1)
 (cube_side := radius_height_cube.2.2.1)
 (cube_orientation := radius_height_cube.2.2.2) 
 (h_radius : radius = 5) 
 (h_height : height = 15)
 (h_cube_side: cube_side = 10)
 (h_cube_orientation: cube_orientation = 1) : 
let v := 1000 in
v^2 = 1000000 :=
by {
  rw v,
  exact pow_two 1000
}

end volume_displacement_of_cube_l487_487218


namespace sum_of_reciprocals_factors_12_l487_487815

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487815


namespace sum_of_reciprocals_factors_12_l487_487717

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487717


namespace angela_hours_per_night_in_january_l487_487246

noncomputable def hours_dec : ℝ := 6.5
noncomputable def days_dec : ℕ := 31
noncomputable def additional_hours_jan : ℝ := 62

theorem angela_hours_per_night_in_january :
  let total_hours_dec := hours_dec * days_dec
      total_hours_jan := total_hours_dec + additional_hours_jan
      days_jan := days_dec
      hours_per_night_jan := total_hours_jan / days_jan
  in hours_per_night_jan = 8.5 :=
by
  have total_hours_dec := hours_dec * ↑days_dec
  have total_hours_jan := total_hours_dec + additional_hours_jan
  have hours_per_night_jan := total_hours_jan / ↑days_dec
  exact sorry

end angela_hours_per_night_in_january_l487_487246


namespace digit_9_appearances_l487_487395

theorem digit_9_appearances (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 700) : 
  let digit_count := (List.range' 1 700).map (λ x, x.toString.count '9') in
  digit_count.sum = 140 :=
by
  sorry

end digit_9_appearances_l487_487395


namespace sum_of_reciprocals_of_factors_of_12_l487_487762

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487762


namespace geometric_sequence_common_ratio_l487_487354

variable {α : Type*} [Field α] 
variable {a : ℕ → α} {S : ℕ → α} {n : ℕ}
variable {q : α}

-- The sequence {a_n} is a geometric sequence
def is_geometric_sequence (a : ℕ → α) (q : α) : Prop :=
  ∀ n, a (n + 1) = q * a n

-- The sum of the first n terms of the geometric sequence {a_n}
def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop :=
  ∀ n, S n = if n = 0 then a 0 else a 0 * (1 - q ^ (n + 1)) / (1 - q)

-- Given conditions
def condition1 (a : ℕ → α) (S : ℕ → α) : Prop :=
  a 5 = 2 * S 4 + 3

def condition2 (a : ℕ → α) (S : ℕ → α) : Prop :=
  a 6 = 2 * S 5 + 3

-- Prove that the common ratio q = 3
theorem geometric_sequence_common_ratio 
  (h1 : is_geometric_sequence a q)
  (h2 : sum_of_first_n_terms a S)
  (h3 : condition1 a S)
  (h4 : condition2 a S) :
  q = 3 :=
sorry

end geometric_sequence_common_ratio_l487_487354


namespace sum_of_reciprocals_factors_12_l487_487727

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487727


namespace find_m_range_l487_487315

-- Define the mathematical objects and conditions
def condition_p (m : ℝ) : Prop :=
  (|1 - m| / Real.sqrt 2) > 1

def condition_q (m : ℝ) : Prop :=
  m < 4

-- Define the proof problem
theorem find_m_range (p q : Prop) (m : ℝ) 
  (hp : ¬ p) (hq : q) (hpq : p ∨ q)
  (hP_imp : p → condition_p m)
  (hQ_imp : q → condition_q m) : 
  1 - Real.sqrt 2 ≤ m ∧ m ≤ 1 + Real.sqrt 2 := 
sorry

end find_m_range_l487_487315


namespace sum_of_coordinates_of_D_l487_487105

theorem sum_of_coordinates_of_D (x y : ℝ) (h1 : (x + 6) / 2 = 2) (h2 : (y + 2) / 2 = 6) :
  x + y = 8 := 
by
  sorry

end sum_of_coordinates_of_D_l487_487105


namespace sum_reciprocals_factors_12_l487_487943

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487943


namespace cost_per_pound_beef_is_correct_l487_487091

variable (budget initial_chicken_cost pounds_beef remaining_budget_after_purchase : ℝ)
variable (spending_on_beef cost_per_pound_beef : ℝ)

axiom h1 : budget = 80
axiom h2 : initial_chicken_cost = 12
axiom h3 : pounds_beef = 5
axiom h4 : remaining_budget_after_purchase = 53
axiom h5 : spending_on_beef = budget - initial_chicken_cost - remaining_budget_after_purchase
axiom h6 : cost_per_pound_beef = spending_on_beef / pounds_beef

theorem cost_per_pound_beef_is_correct : cost_per_pound_beef = 3 :=
by
  sorry

end cost_per_pound_beef_is_correct_l487_487091


namespace unique_f_l487_487069

-- Definitions and conditions for the function f
def S := { x : ℝ // x ≠ 0 }

def f (x : S) : ℝ 

-- Given conditions
axiom f_2 : f ⟨2, by norm_num⟩ = 2

axiom f_prop2 (x y : S) (h : x.val + y.val ≠ 0) : 
  f ⟨1 / (x.val + y.val), by field_simp [h]; assumption⟩ = 
  f ⟨1 / x, by field_simp⟩ + f ⟨1 / y, by field_simp⟩

axiom f_prop3 (x y : S) (h : x.val + y.val ≠ 0) : 
  2 * (x.val + y.val) * f ⟨x.val + y.val, h⟩ = (x.val * y.val) * (f x) * (f y)

-- Theorem statement to be proved
theorem unique_f : ∃! (f : S → ℝ), 
  (f ⟨2, by norm_num⟩ = 2) ∧ 
  (∀ x y : S, x.val + y.val ≠ 0 → 
    f ⟨1 / (x.val + y.val), by field_simp [x.prop, y.prop]; assumption⟩ = 
    f ⟨1 / x, by field_simp [x.prop]⟩ + f ⟨1 / y, by field_simp [y.prop]⟩) ∧ 
  (∀ x y : S, x.val + y.val ≠ 0 → 
    2 * (x.val + y.val) * f ⟨x.val + y.val, x.prop⟩ = (x.val * y.val) * (f x) * (f y)) :=
sorry

end unique_f_l487_487069


namespace min_PQ_length_l487_487021

open Real

variables {a b c : ℝ} (c_lt_b : c < b) (b_lt_a : b < a) (a_lt_2c : a < 2 * c) 
variables (α β γ : ℝ) (triangle_ABC_area : ℝ) (half_area : triangle_ABC_area / 2)

def min_length_PQ : ℝ := sqrt (2 * a * b) * sin (γ / 2)

theorem min_PQ_length (c_lt_b : c < b) (b_lt_a : b < a) (a_lt_2c : a < 2 * c) :
  ∃ PQ, PQ = min_length_PQ a b γ := 
by
  sorry

end min_PQ_length_l487_487021


namespace min_diff_l487_487368

noncomputable def f (x : ℝ) : ℝ := Real.log (x / 2) + 1 / 2
noncomputable def g (x : ℝ) : ℝ := Real.exp (x - 2)

theorem min_diff (a : ℝ) : ∃ b > 0, g(a) = f(b) ∧ ∀ b', g(a) = f(b') → b' - a ≥ Real.log 2 := 
by 
  sorry

end min_diff_l487_487368


namespace robotics_club_problem_l487_487488

theorem robotics_club_problem 
    (total_students cs_students eng_students both_students : ℕ)
    (h1 : total_students = 120)
    (h2 : cs_students = 75)
    (h3 : eng_students = 50)
    (h4 : both_students = 10) :
    total_students - (cs_students - both_students + eng_students - both_students + both_students) = 5 := by
  sorry

end robotics_club_problem_l487_487488


namespace sum_of_reciprocals_of_factors_of_12_l487_487795

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487795


namespace sum_reciprocals_factors_of_12_l487_487685

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487685


namespace sum_reciprocals_factors_12_l487_487884

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487884


namespace angle_C_90_degrees_l487_487310

theorem angle_C_90_degrees (A B C : ℝ) (T : Triangle) 
  (h1 : |sin A - (1/2)| > 0)
  (h2 : (tan B - real.sqrt 3)^2 = 0):
  ∠C = 90 :=
  begin
    sorry
  end

end angle_C_90_degrees_l487_487310


namespace incenters_equidistant_l487_487450

theorem incenters_equidistant {A B C K L : Point}
  (hK : is_on_arc A B K)
  (hL : is_on_arc B C L)
  (h_parallel : parallel K L A C)
  (circum_circle : Circle)
  (hA : on_circle A circum_circle)
  (hB : on_circle B circum_circle)
  (hC : on_circle C circum_circle) :
  let D := incenter (triangle A B K),
      E := incenter (triangle C B L),
      N := arc_midpoint A B C circum_circle in
  distance D N = distance E N :=
sorry

end incenters_equidistant_l487_487450


namespace find_p_value_l487_487264

theorem find_p_value
  (p q : ℝ) 
  (quadratic_eqn : ∀ x, x^2 + p * x + q = 0) 
  (H_pos_p : p > 0) 
  (H_pos_q : q > 0) 
  (H_complex_roots : p^2 - 4 * q < 0) 
  (H_real_part : ∀ (r : ℂ), (r = (-(p : ℂ) + complex.sqrt ((p : ℂ)^2 - 4 * (q : ℂ))) / 2 → r.re = 1/2) ∧ 
                             (r = (-(p : ℂ) - complex.sqrt ((p : ℂ)^2 - 4 * (q : ℂ))) / 2 → r.re = 1/2)) :
  p = 1 :=
by
  sorry

end find_p_value_l487_487264


namespace max_points_of_intersection_l487_487576

theorem max_points_of_intersection (circles : Fin 2 → Circle) (lines : Fin 3 → Line) :
  number_of_intersections circles lines = 17 :=
sorry

end max_points_of_intersection_l487_487576


namespace find_xy_l487_487992

noncomputable def xy_value (x y : ℝ) := x * y

theorem find_xy :
  ∃ x y : ℝ, (x + y = 2) ∧ (x^2 * y^3 + y^2 * x^3 = 32) ∧ xy_value x y = -8 :=
by
  sorry

end find_xy_l487_487992


namespace jill_sod_area_l487_487049

noncomputable def area_of_sod (yard_width yard_length sidewalk_width sidewalk_length flower_bed1_depth flower_bed1_length flower_bed2_depth flower_bed2_length flower_bed3_width flower_bed3_length flower_bed4_width flower_bed4_length : ℝ) : ℝ :=
  let yard_area := yard_width * yard_length
  let sidewalk_area := sidewalk_width * sidewalk_length
  let flower_bed1_area := flower_bed1_depth * flower_bed1_length
  let flower_bed2_area := flower_bed2_depth * flower_bed2_length
  let flower_bed3_area := flower_bed3_width * flower_bed3_length
  let flower_bed4_area := flower_bed4_width * flower_bed4_length
  let total_non_sod_area := sidewalk_area + 2 * flower_bed1_area + flower_bed2_area + flower_bed3_area + flower_bed4_area
  yard_area - total_non_sod_area

theorem jill_sod_area : 
  area_of_sod 200 50 3 50 4 25 4 25 10 12 7 8 = 9474 := by sorry

end jill_sod_area_l487_487049


namespace TrainTravelDays_l487_487238

-- Definition of the problem conditions
def train_start (days: ℕ) : ℕ := 
  if days = 0 then 0 -- no trains to meet on the first day
  else days -- otherwise, meet 'days' number of trains

/-- 
  Prove that if a train comes across 4 trains on its way from Amritsar to Bombay and starts at 9 am, 
  then it takes 5 days for the train to reach its destination.
-/
theorem TrainTravelDays (meet_train_count : ℕ) : meet_train_count = 4 → train_start (meet_train_count) + 1 = 5 :=
by
  intro h
  rw [h]
  sorry

end TrainTravelDays_l487_487238


namespace identify_counterfeits_n2_identify_counterfeits_n3_l487_487438

-- Definitions based on problem conditions
variables {α : Type*} [linear_ordered_field α]

-- Condition: There are distinct natural denominations
def distinct_denominations (denoms : list α) : Prop :=
  denoms.nodup

-- Condition: There are exactly N counterfeit banknotes
def has_n_counterfeits (denoms : list α) (N : ℕ) (counterfeits : list α) : Prop :=
  N = counterfeits.length ∧ ∀ c ∈ counterfeits, c ∈ denoms

-- Condition: Detector checks sum of denominations of genuine notes in a selected set
def valid_check (denoms : list α) (selected : list α) : Prop :=
  ∀ x ∈ selected, x ∈ denoms

-- Question: Identify all counterfeit banknotes in N checks

-- For N = 2
theorem identify_counterfeits_n2 (denoms : list α) (counterfeits : list α) (detector : list α → α) :
  distinct_denominations denoms →
  has_n_counterfeits denoms 2 counterfeits →
  (∀ denotes_subset, valid_check denoms denotes_subset → (detector denotes_subset = detector denotes_subset)) →
  ∃ (counterfeit1 counterfeit2 : α), 
    counterfeit1 ∈ counterfeits ∧
    counterfeit2 ∈ counterfeits ∧
    counterfeit1 ≠ counterfeit2 :=
sorry

-- For N = 3
theorem identify_counterfeits_n3 (denoms : list α) (counterfeits : list α) (detector : list α → α) :
  distinct_denominations denoms →
  has_n_counterfeits denoms 3 counterfeits →
  (∀ denotes_subset, valid_check denoms denotes_subset → (detector denotes_subset = detector denotes_subset)) →
  ∃ (counterfeit1 counterfeit2 counterfeit3 : α), 
    counterfeit1 ∈ counterfeits ∧
    counterfeit2 ∈ counterfeits ∧
    counterfeit3 ∈ counterfeits ∧
    counterfeit1 ≠ counterfeit2 ∧
    counterfeit1 ≠ counterfeit3 ∧
    counterfeit2 ≠ counterfeit3 :=
sorry

end identify_counterfeits_n2_identify_counterfeits_n3_l487_487438


namespace sum_of_reciprocals_of_factors_of_12_l487_487801

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487801


namespace star_shell_arrangements_l487_487051

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Conditions
def outward_points : ℕ := 6
def inward_points : ℕ := 6
def total_points : ℕ := outward_points + inward_points
def unique_shells : ℕ := 12

-- The problem statement translated into Lean 4:
theorem star_shell_arrangements : (factorial unique_shells / 12 = 39916800) :=
by
  sorry

end star_shell_arrangements_l487_487051


namespace total_savings_is_correct_l487_487304

theorem total_savings_is_correct :
  let fox_price := 15
  let pony_price := 18
  let num_fox_pairs := 3
  let num_pony_pairs := 2
  let total_discount_rate := 22
  let pony_discount_rate := 13.999999999999993
  let fox_discount_rate := total_discount_rate - pony_discount_rate
  let total_regular_price := (num_fox_pairs * fox_price) + (num_pony_pairs * pony_price)
  let fox_savings := (fox_discount_rate / 100) * (num_fox_pairs * fox_price)
  let pony_savings := (pony_discount_rate / 100) * (num_pony_pairs * pony_price)
  let total_savings := fox_savings + pony_savings
  total_savings = 8.64 :=
by
  -- Definitions as per problem conditions
  let fox_price := 15 
  let pony_price := 18
  let num_fox_pairs := 3
  let num_pony_pairs := 2
  let total_discount_rate := 22
  let pony_discount_rate := 13.999999999999993
  let fox_discount_rate := total_discount_rate - pony_discount_rate
  let total_regular_price := (num_fox_pairs * fox_price) + (num_pony_pairs * pony_price)
  let fox_savings := (fox_discount_rate / 100) * (num_fox_pairs * fox_price)
  let pony_savings := (pony_discount_rate / 100) * (num_pony_pairs * pony_price)
  let total_savings := fox_savings + pony_savings
  have correct_total_savings : total_savings = 8.64 := sorry
  exact correct_total_savings

end total_savings_is_correct_l487_487304


namespace sum_of_reciprocals_of_factors_of_12_l487_487908

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487908


namespace sum_of_reciprocals_of_factors_of_12_l487_487803

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487803


namespace ratio_equivalence_to_minutes_l487_487204

-- Define conditions and equivalence
theorem ratio_equivalence_to_minutes :
  ∀ (x : ℝ), (8 / 4 = 8 / x) → x = 4 / 60 :=
by
  intro x
  sorry

end ratio_equivalence_to_minutes_l487_487204


namespace trapezoid_area_identity_l487_487131

variable {ℝ : Type*}

structure Trapezoid (A B C D O : Point) : Prop :=
  (parallel : is_parallel A B C D)
  (intersection : intersects A C B D O)

def area (P Q R : Point) : ℝ := sorry -- Dummy definition for the sake of example

theorem trapezoid_area_identity
  (A B C D O : Point)
  (h : Trapezoid A B C D O) :
  Real.sqrt (area A B C D) = Real.sqrt (area A B O) + Real.sqrt (area C D O) :=
  sorry

end trapezoid_area_identity_l487_487131


namespace determinant_problem_l487_487343

theorem determinant_problem 
  (x y z w : ℝ) 
  (h : x * w - y * z = 7) : 
  ((x * (8 * z + 4 * w)) - (z * (8 * x + 4 * y))) = 28 :=
by 
  sorry

end determinant_problem_l487_487343


namespace count_valid_pairs_l487_487261

-- Define the problem constraints and functions
def valid_pair (a b : ℤ) : Prop :=
  2 ≤ a ∧ a ≤ 2021 ∧ 2 ≤ b ∧ b ≤ 2021 ∧
  a^log b (a^(-4)) = b^log a (b * a^(-3))

-- Define the main theorem
theorem count_valid_pairs : 
  (∃ n : ℤ, n = 43 ∧ (∀ a b : ℤ, valid_pair a b → (a = 2 ∧ b = a ^ 2))) :=
sorry

end count_valid_pairs_l487_487261


namespace sum_of_reciprocals_factors_12_l487_487825

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487825


namespace sum_reciprocals_factors_12_l487_487620

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487620


namespace least_number_subtracted_l487_487975

theorem least_number_subtracted (a b c : ℕ) (h1 : a = 7) (h2 : b = 9) (h3 : c = 11) :
  ∃ x, 0 ≤ x ∧ x < 1398 ∧ (1398 - x) % a = 5 ∧ (1398 - x) % b = 5 ∧ (1398 - x) % c = 5 ∧ x = 22 :=
by {
  sorry
}

end least_number_subtracted_l487_487975


namespace other_acute_angle_measure_l487_487029

-- Definitions based on the conditions
def right_triangle_sum (a b : ℝ) : Prop := a + b = 90
def is_right_triangle (a b : ℝ) : Prop := right_triangle_sum a b ∧ a = 20

-- The statement to prove
theorem other_acute_angle_measure {a b : ℝ} (h : is_right_triangle a b) : b = 70 :=
sorry

end other_acute_angle_measure_l487_487029


namespace trapezoid_AP_length_l487_487470

noncomputable def proof_AP (A B C D P : Type) (distance : A → A → ℝ)
  (angle : A → A → A → ℝ) : Prop :=
  let AB := distance A B
  let AD := distance A D
  let CD := distance C D
  ∧ (distance A C = √(21^2 + 28^2))
  ∧ (distance P A = distance A C - (8 * 28) / 35)
  ∧ (distance P A = 143 / 5)
  
theorem trapezoid_AP_length (A B C D P : Type) (distance : A → A → ℝ)
  (angle : A → A → A → ℝ)
  (h1 : convex_trapezoid ABCD)
  (h2 : angle B A D = 90 ∧ angle A D C = 90)
  (h3 : distance A B = 20)
  (h4 : distance A D = 21)
  (h5 : distance C D = 28)
  (h6 : ∃ P ≠ A, point_on_segment P A C ∧ angle B P D = 90) :
  distance A P = 143 / 5 := 
sorry

end trapezoid_AP_length_l487_487470


namespace truck_toll_l487_487549

-- Define the required values and conditions
def R (weight : ℝ) : ℝ :=
  if weight ≤ 5 then 1.50
  else if weight ≤ 10 then 2.50
  else 4.00

def W (axles_wheels : List ℕ) : ℝ :=
  if any (λ w => w > 2) axles_wheels then 0.80 else 0.60

def S (y : ℕ) : ℝ :=
  if y % 2 = 0 then 0.50 * (y - 4) else 0.50 * (y - 1)

def T (R : ℝ) (W : ℝ) (S : ℝ) (x : ℕ) : ℝ :=
  R + W * (x - 2) + S

theorem truck_toll :
  let weight := 9
  let axles_wheels := [2, 4, 3, 3, 3, 3] -- Flats represent wheels on each axle
  let y := 18
  let x := axles_wheels.length
  T (R weight) (W axles_wheels) (S y) x = 12.70 :=
by {
  let weight := 9
  let axles_wheels := [2, 4, 3, 3, 3, 3]
  let y := 18
  let x := axles_wheels.length
  have R_val : R weight = 2.50 := by simp [R, weight]
  have W_val : W axles_wheels = 0.80 := by simp [W, axles_wheels]
  have S_val : S y = 7.00 := by simp [S, y]
  have T_val : T R_val W_val S_val x = 2.50 + 0.80 * (x - 2) + 7.00 := rfl
  have x_val : x = 6 := by simp [axles_wheels]

  simp [T_val, x_val, T],
  ring,
}

end truck_toll_l487_487549


namespace caiden_cost_l487_487483

def cost_per_foot : ℕ := 8
def discount_rate (feet : ℕ) : ℕ → ℕ := 
  if feet >= 300 then 15 else if feet >= 200 then 10 else 0
def shipping_fee (feet : ℕ) : ℕ :=
  if feet > 200 then ((feet - 200 + 99) / 100) * 150 else 0 
def sales_tax (cost : ℕ) : ℕ := 
  if cost > 2000 then 7 else 5
def total_feet : ℕ := 300
def free_feet : ℕ := 250
def paid_feet := total_feet - free_feet

def calc_total_cost : ℕ :=
  let initial_cost := paid_feet * cost_per_foot
  let discount := initial_cost * discount_rate(total_feet) / 100
  let discounted_price := initial_cost - discount
  let shipping := shipping_fee(paid_feet)
  let subtotal := discounted_price + shipping
  let tax := subtotal * sales_tax(subtotal) / 100
  subtotal + tax

theorem caiden_cost : calc_total_cost = 357 := by
  sorry

end caiden_cost_l487_487483


namespace sum_reciprocal_factors_12_l487_487870

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487870


namespace greatest_3_digit_base7_divisible_by_7_l487_487169

def base7ToDec (a b c : ℕ) : ℕ := a * 7^2 + b * 7^1 + c * 7^0

theorem greatest_3_digit_base7_divisible_by_7 :
  ∃ (a b c : ℕ), a ≠ 0 ∧ a < 7 ∧ b < 7 ∧ c < 7 ∧
  base7ToDec a b c % 7 = 0 ∧ base7ToDec a b c = 342 :=
begin
  use [6, 6, 6],
  split, { repeat { norm_num } }, -- a ≠ 0
  split, { norm_num }, -- a < 7
  split, { norm_num }, -- b < 7
  split, { norm_num }, -- c < 7
  split,
  { norm_num },
  norm_num,
end

end greatest_3_digit_base7_divisible_by_7_l487_487169


namespace sum_of_reciprocals_of_factors_of_12_l487_487667

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487667


namespace sum_of_reciprocals_of_factors_of_12_l487_487766

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487766


namespace range_a_b_c_l487_487412

-- Definitions of the conditions
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x - c / 2 + b * sin x * cos x + c * cos^2 x

def property_T (g : ℝ → ℝ) : Prop :=
∃ x1 x2, x1 ≠ x2 ∧ g'(x1) * g'(x2) = -1

-- The main theorem statement we want to prove
theorem range_a_b_c (a b c : ℝ) (h1: b^2 + c^2 = 1) (h2: property_T (g a b c)) : 
  a + b + c ∈ set.Icc (-(real.sqrt 2)) (real.sqrt 2) := 
sorry

end range_a_b_c_l487_487412


namespace Sue_made_22_buttons_l487_487445

def Mari_buttons : Nat := 8
def Kendra_buttons : Nat := 5 * Mari_buttons + 4
def Sue_buttons : Nat := Kendra_buttons / 2

theorem Sue_made_22_buttons : Sue_buttons = 22 :=
by
  -- proof to be added
  sorry

end Sue_made_22_buttons_l487_487445


namespace log_sum_geometric_sequence_l487_487039

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) :=
  ∀ n m : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + m) = a n * r ^ m

noncomputable def geom_mean (x y : ℝ) := sqrt (x * y)

theorem log_sum_geometric_sequence :
  (∀ n m : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + m) = a n * r ^ m) →
  (a 2 * a 10 = 1 / 3) →
  (log 3 (a 4) + log 3 (a 8) = -1) :=
by
  intros h₁ h₂
  sorry

end log_sum_geometric_sequence_l487_487039


namespace correct_calculation_l487_487187

theorem correct_calculation :
  (∀ x, (x = (\sqrt 3)^2 → x ≠ 9)) ∧
  (∀ y, (y = \sqrt ((-2)^2) → y ≠ -2)) ∧
  (∀ z, (z = \sqrt 3 * \sqrt 2 → z ≠ 6)) ∧
  (∀ w, (w = \sqrt 8 / \sqrt 2 → w = 2)) :=
by
  sorry

end correct_calculation_l487_487187


namespace chris_money_before_birthday_l487_487259

def money_from_grandmother : ℕ := 25
def money_from_aunt_uncle : ℕ := 20
def money_from_parents : ℕ := 75
def current_total_money : ℕ := 279

def total_birthday_money : ℕ := money_from_grandmother + money_from_aunt_uncle + money_from_parents := by
  exact 120

def money_before_birthday : ℕ := current_total_money - total_birthday_money := by
  exact 159

theorem chris_money_before_birthday :
  (current_total_money - (money_from_grandmother + money_from_aunt_uncle + money_from_parents)) = 159 := by
  sorry

end chris_money_before_birthday_l487_487259


namespace card_probability_ratio_l487_487303

theorem card_probability_ratio :
  let total_cards := 40
  let numbers := 10
  let cards_per_number := 4
  let choose (n k : ℕ) := Nat.choose n k
  let p := 10 / choose total_cards 4
  let q := 1440 / choose total_cards 4
  (q / p) = 144 :=
by
  sorry

end card_probability_ratio_l487_487303


namespace sum_of_reciprocals_factors_12_l487_487814

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487814


namespace sum_reciprocals_of_factors_12_l487_487838

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487838


namespace problem1_l487_487256

variable (x : ℝ)

theorem problem1 : 5 * x^2 * x^4 + x^8 / (-x)^2 = 6 * x^6 :=
  sorry

end problem1_l487_487256


namespace sum_of_reciprocals_factors_12_l487_487716

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487716


namespace sum_of_reciprocals_factors_12_l487_487822

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487822


namespace parabola_focus_directrix_distance_l487_487269

theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), x^2 = y → (∃ p : ℝ, p = 1/2) :=
by
  intros x y h
  use 1/2
  sorry

end parabola_focus_directrix_distance_l487_487269


namespace sum_of_reciprocals_of_factors_of_12_l487_487670

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487670


namespace candy_partition_l487_487244

theorem candy_partition :
  let candies := 10
  let boxes := 3
  ∃ ways : ℕ, ways = Nat.choose (candies + boxes - 1) (boxes - 1) ∧ ways = 66 :=
by
  let candies := 10
  let boxes := 3
  let ways := Nat.choose (candies + boxes - 1) (boxes - 1)
  have h : ways = 66 := sorry
  exact ⟨ways, ⟨rfl, h⟩⟩

end candy_partition_l487_487244


namespace quadratic_general_form_l487_487530

theorem quadratic_general_form :
  ∀ (x : ℝ), 2 * x^2 - x * (x - 4) = 5 → x^2 + 4 * x - 5 = 0 :=
by
  intros x h,
  sorry

end quadratic_general_form_l487_487530


namespace sum_reciprocals_factors_12_l487_487626

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487626


namespace sum_reciprocals_factors_12_l487_487648

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487648


namespace sum_reciprocals_factors_12_l487_487694

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487694


namespace sum_reciprocal_factors_of_12_l487_487608

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487608


namespace total_oranges_after_increase_l487_487481

theorem total_oranges_after_increase :
  let Mary := 122
  let Jason := 105
  let Tom := 85
  let Sarah := 134
  let increase_rate := 0.10
  let new_Mary := Mary + Mary * increase_rate
  let new_Jason := Jason + Jason * increase_rate
  let new_Tom := Tom + Tom * increase_rate
  let new_Sarah := Sarah + Sarah * increase_rate
  let total_new_oranges := new_Mary + new_Jason + new_Tom + new_Sarah
  Float.round total_new_oranges = 491 := 
by
  sorry

end total_oranges_after_increase_l487_487481


namespace sum_of_reciprocals_of_factors_of_12_l487_487665

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487665


namespace minimize_sum_of_distances_l487_487337

variables {α : Type*} [LinearOrder α] (P : α) (P1 P2 P3 P4 P5 P6 P7 P8 P9 : α)

def s (P : α) := abs (P - P1) + abs (P - P2) + abs (P - P3) + abs (P - P4) + abs (P - P5) + abs (P - P6) + abs (P - P7) + abs (P - P8) + abs (P - P9)

theorem minimize_sum_of_distances : (∀ P, s P ≥ s P5) :=
sorry

end minimize_sum_of_distances_l487_487337


namespace find_x4_l487_487227

theorem find_x4 (x : ℝ) (hx_pos : 0 < x) (h_eq : sqrt (1 - x^2) + sqrt (1 + x^2) = 2) : x^4 = 0 :=
by
  sorry

end find_x4_l487_487227


namespace number_of_students_selected_correct_l487_487423

noncomputable def systematic_sampling_campsites : (ℕ × ℕ × ℕ) :=
  let first_campsite : Finset ℕ := {n ∈ Finset.range 156 | (n - 4) % 10 = 0}
  let second_campsite : Finset ℕ := {n ∈ Finset.range' 156 226 | (n - 4) % 10 = 0}
  let third_campsite : Finset ℕ := {n ∈ Finset.range' 256 401 | (n - 4) % 10 = 0}
  (first_campsite.card, second_campsite.card, third_campsite.card)
  
theorem number_of_students_selected_correct :
  systematic_sampling_campsites = (16, 10, 14) := by
  sorry

end number_of_students_selected_correct_l487_487423


namespace problem_statement_l487_487122

noncomputable def inequality_to_prove (x : ℝ) :=
  8^Real.sqrt(log 2 x) - 2^(Real.sqrt(4 * log 2 x) + 3) + 21 * x^Real.sqrt(log x 2) ≤ 18

theorem problem_statement : 
  ∀ x : ℝ, (x ∈ Icc 1 2 ∨ x = 3^Real.log 2 3) → inequality_to_prove x :=
by
  intros x hx
  sorry

end problem_statement_l487_487122


namespace even_three_digit_desc_order_count_l487_487388

/-
Problem: Prove that the number of even three-digit integers with digits in strictly decreasing order is 4.
-/

def is_digit (d : ℕ) : Prop := d ∈ {2, 4, 6, 8}

/-- Define a three-digit even integer with digits in strictly decreasing order. -/
def even_three_digit_desc_order_digits (a b c : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ a > b ∧ b > c

theorem even_three_digit_desc_order_count :
  ∃ n : ℕ, n = 4 ∧ (∀ a b c, even_three_digit_desc_order_digits a b c → true) := sorry

end even_three_digit_desc_order_count_l487_487388


namespace sum_reciprocals_factors_12_l487_487749

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487749


namespace thirty_percent_of_forty_percent_of_x_l487_487004

theorem thirty_percent_of_forty_percent_of_x (x : ℝ) (h : 0.12 * x = 24) : 0.30 * 0.40 * x = 24 :=
sorry

end thirty_percent_of_forty_percent_of_x_l487_487004


namespace range_of_a_l487_487317

variables {a x : ℝ}

def p : Prop := x^2 - 12*x + 20 < 0
def q (a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 > 0

axiom ha_pos : a > 0
axiom h_suff : (¬ q a) → (¬ p)

theorem range_of_a (h : (¬ q a) → (¬ p)) : 0 < a ∧ a < 1 := 
by 
  sorry

end range_of_a_l487_487317


namespace no_x3_term_l487_487353

open Polynomial

noncomputable def poly1 := 5 - 3 * X + (X^2) * C m - 6 * X^3
noncomputable def poly2 := 1 - 2 * X

theorem no_x3_term (m : ℝ) :
  (poly1 * poly2).coeff 3 = 0 → m = -3 :=
by 
  intros h
  sorry

end no_x3_term_l487_487353


namespace find_sin_A_find_b_plus_c_l487_487413

/-- Lean statement corresponding to the math problem --/

-- Define the triangle and its sides opposite to the angles
variables (a b c : ℝ) (A B C : ℝ)

-- Define the conditions in the problem
axiom triangle_sides_and_angles_eq : a * cos B = (3 * c - b) * cos A

-- Define the properties of the triangle
axiom angles_sum_to_pi : A + B + C = π

-- Define the first part of the problem
theorem find_sin_A (h : a * cos B = (3 * c - b) * cos A) (h2 : A + B + C = π) : sin A = 2 * sqrt 2 / 3 := 
sorry

-- Additional conditions for the second part
axiom area_of_triangle_eq : 1 / 2 * b * c * sin A = sqrt 2
axiom a_eq : a = 2 * sqrt 2

-- Define the second part of the problem with additional conditions
theorem find_b_plus_c (h : a * cos B = (3 * c - b) * cos A) 
                     (h2 : A + B + C = π)
                     (h3 : a = 2 * sqrt 2)
                     (h4 : 1 / 2 * b * c * sin A = sqrt 2) : b + c = 4 :=
sorry

end find_sin_A_find_b_plus_c_l487_487413


namespace smallest_n_is_35_l487_487226

def positive_integer (n : ℕ) := n > 0

def not_divisible_by_2_or_3 (n : ℕ) := ¬ (2 ∣ n) ∧ ¬ (3 ∣ n)

def no_a_b_such_that_2_pow_a_minus_3_pow_b_eq_n (n : ℕ) :=
  ∀ (a b : ℕ), |2 ^ a - 3 ^ b| ≠ n

theorem smallest_n_is_35 :
  (positive_integer 35) ∧
  (not_divisible_by_2_or_3 35) ∧
  (no_a_b_such_that_2_pow_a_minus_3_pow_b_eq_n 35) ∧
  (∀ m : ℕ, (positive_integer m) ∧
    (not_divisible_by_2_or_3 m) ∧
    (no_a_b_such_that_2_pow_a_minus_3_pow_b_eq_n m) → m ≥ 35) :=
by
  sorry

end smallest_n_is_35_l487_487226


namespace lulu_pop_tarts_l487_487090

theorem lulu_pop_tarts (lola_mini: ℕ) (lola_pop: ℕ) (lola_blue: ℕ) 
                       (lulu_mini: ℕ) (lulu_blue: ℕ) (total_pastries: ℕ) (lulu_pop: ℕ) :
  lola_mini = 13 ∧ lola_pop = 10 ∧ lola_blue = 8 ∧
  lulu_mini = 16 ∧ lulu_blue = 14 ∧ total_pastries = 73 →
  lulu_pop = 12 :=
by
  intros h
  cases h with h1 h_rest
  cases h_rest with h2 h_rest
  cases h_rest with h3 h_rest
  cases h_rest with h4 h_rest
  cases h_rest with h5 h_total
  sorry

end lulu_pop_tarts_l487_487090


namespace complete_the_square_l487_487186

theorem complete_the_square (x : ℝ) : 
    (x^2 - 2 * x - 5 = 0) -> (x - 1)^2 = 6 :=
by sorry

end complete_the_square_l487_487186


namespace sum_of_reciprocals_of_factors_of_12_l487_487765

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487765


namespace sum_reciprocals_factors_12_l487_487697

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487697


namespace rational_interval_in_S_l487_487057

noncomputable def S : Set ℚ := {x | sorry}

theorem rational_interval_in_S :
  (∀ (x : ℚ), (0 < x ∧ x < 1) → x ∈ S) :=
begin
  have base_case : (1 / 2) ∈ S,
  { sorry },
  have inductive_step : ∀ (x : ℚ), x ∈ S → (1 / (x + 1)) ∈ S ∧ (x / (x + 1)) ∈ S,
  { sorry },
  sorry
end

end rational_interval_in_S_l487_487057


namespace sum_reciprocals_12_l487_487927

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487927


namespace angle_between_a_and_b_l487_487002

variables {ℝ} {n : Type*} [inner_product_space ℝ n]

noncomputable def magnitude (v : n) : ℝ := real.sqrt (inner_product_space.is_R_or_C.re (inner_product_space.inner v v))

noncomputable def angle_between_vectors (a b : n) : ℝ := real.arccos ((inner_product_space.inner a b) / (magnitude a * magnitude b))

theorem angle_between_a_and_b
  (a b : n)
  (ha : magnitude a = 2)
  (hb : magnitude b = 4)
  (ha_perp_ab : inner_product_space.inner (a + b) a = 0) :
  angle_between_vectors a b = 2 * real.pi / 3 :=
by
  sorry

end angle_between_a_and_b_l487_487002


namespace adam_change_l487_487241

def original_price := 4.28
def discount_rate := 0.10
def sales_tax_rate := 0.07
def money_adam_has := 5.00

noncomputable def discount_amount := discount_rate * original_price
noncomputable def new_price := original_price - discount_amount
noncomputable def sales_tax_amount := sales_tax_rate * new_price
noncomputable def total_cost := new_price + sales_tax_amount
noncomputable def total_cost_rounded := Float.round (total_cost * 100.0) / 100.0
def change := money_adam_has - total_cost_rounded

theorem adam_change : change = 0.88 := by
  sorry

end adam_change_l487_487241


namespace part_I_part_II_l487_487425

variables (a : ℝ) (varphi : ℝ) (theta : ℝ)
variables (x y : ℝ) (rho1 rho2 : ℝ)

-- Conditions
-- Curve C_1 with parametric equations and the given point A(2, 0)
def curve_C1 := (x = a * real.cos varphi) ∧ (y = a * real.sin varphi) ∧ (a > 0) ∧ (2 = a * real.cos varphi) ∧ (0 = y)

-- Polar equation of Curve C_2
def curve_C2 := (rho1 = a * real.cos theta)

-- Points M and N on C_1 with specific polar coordinates
def points_M_N_on_C1 := 
  (rho1 * real.cos theta, rho1 * real.sin theta) ∈ (set_of (λ p, (p.1^2 / 4 + p.2^2 = 1))) ∧
  (rho2 * real.cos (theta + real.pi / 2), rho2 * real.sin (theta + real.pi / 2)) ∈ (set_of (λ p, (p.1^2 / 4 + p.2^2 = 1)))

-- The proof goals
theorem part_I : curve_C2 a θ → (x-1)^2 + y^2 = 1 := by
  sorry

theorem part_II : points_M_N_on_C1 a θ rho1 rho2 → 1 / (rho1^2) + 1 / (rho2^2) = 5 / 4 := by
  sorry

end part_I_part_II_l487_487425


namespace min_value_abs_x_minus_2_plus_1_l487_487137

theorem min_value_abs_x_minus_2_plus_1 : ∃ x : ℝ, (∀ y : ℝ, |y - 2| + 1 ≥ |x - 2| + 1) ∧ |x - 2| + 1 = 1 :=
by {
  use 2,
  split,
  {
    intro y,
    exact le_abs_self_sub_add_one y,
  },
  {
    norm_num,
  },
}

noncomputable def le_abs_self_sub_add_one {y : ℝ} : |y - 2| + 1 ≥ |2 - 2| + 1 := sorry

end min_value_abs_x_minus_2_plus_1_l487_487137


namespace sum_reciprocals_factors_12_l487_487635

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487635


namespace minimum_value_l487_487469

noncomputable theory
open Complex

def condition (z : ℂ) : Prop := abs (z - 5 * I) + abs (z - 7) = 10

theorem minimum_value (z : ℂ) (h : condition z) : abs z = 35 / Real.sqrt 74 :=
sorry

end minimum_value_l487_487469


namespace percent_calculation_l487_487978

theorem percent_calculation (x : ℝ) : 
  (∃ y : ℝ, y / 100 * x = 0.3 * 0.7 * x) → ∃ y : ℝ, y = 21 :=
by
  sorry

end percent_calculation_l487_487978


namespace percentage_to_decimal_l487_487385

theorem percentage_to_decimal (p : ℝ) (h : p = 3) : p / 100 = 0.03 := 
begin
  sorry
end

end percentage_to_decimal_l487_487385


namespace sum_reciprocals_factors_12_l487_487634

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487634


namespace sum_of_reciprocals_factors_12_l487_487780

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487780


namespace sum_of_reciprocals_factors_12_l487_487776

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487776


namespace solution_set_of_inequality_l487_487398

-- Define the given conditions
variables {a b m n x : ℝ}
variables (a_ne_b : a ≠ b)
variables (h1 : m > 0)
variables (h2 : n > 0)
variables (zeroes : ∀ x, x^2 - m * x + n = 0 → x = a ∨ x = b)
variables (progression : (a, b, -1).is_arithmetic_or_geometric_seq)

-- State the main theorem
theorem solution_set_of_inequality : {x : ℝ | x < 1 ∨ x ≥ 5 / 2} = {x : ℝ | (x - m) / (x - n) ≥ 0} :=
sorry

end solution_set_of_inequality_l487_487398


namespace sum_reciprocals_factors_12_l487_487880

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487880


namespace complement_of_A_in_U_l487_487375

open Set

def U : Set ℕ := {x | x < 8}
def A : Set ℕ := {x | (x - 1) * (x - 3) * (x - 4) * (x - 7) = 0}

theorem complement_of_A_in_U : (U \ A) = {0, 2, 5, 6} := by
  sorry

end complement_of_A_in_U_l487_487375


namespace integer_root_of_polynomial_l487_487140

theorem integer_root_of_polynomial (p q : ℚ) (h1 : (2 : ℝ) - (√5) ∈ {x : ℝ | x ^ 3 + ↑p * x + ↑q = 0}) :
  ∃ (r : ℤ), r = -4 ∧ (r : ℝ) ∈ {x : ℝ | x ^ 3 + ↑p * x + ↑q = 0} :=
by {
  sorry
}

end integer_root_of_polynomial_l487_487140


namespace sum_reciprocals_factors_12_l487_487953

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487953


namespace sum_odd_integers_9_to_39_l487_487150

theorem sum_odd_integers_9_to_39 : 
  (∀ (n : ℕ), 2*n+1 = (n+1)*(n+1)) → 
  (let N := 15 + 1 in (39 * (39 + 1) - 9 * (9 - 1)) / 2 = 384) := 
by
  intros h
  have step1 : 20^2 = 400 := by norm_num
  have step2 : 4^2 = 16 := by norm_num
  have step3 : 400 - 16 = 384 := by norm_num
  exact step3

end sum_odd_integers_9_to_39_l487_487150


namespace sum_reciprocal_factors_12_l487_487857

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487857


namespace sum_reciprocals_factors_of_12_l487_487690

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487690


namespace minimize_segment_length_l487_487491

theorem minimize_segment_length (O A B M N : Point) (OA OB x : ℝ) (hOA_gt_OB : OA > OB)
  (hM_on_OA : point_on_segment M O A) (hN_on_ext_OB : point_on_extension N O B)
  (hAM_eq_BN : AM = x ∧ BN = x) : x = (OA - OB) / 2 := 
by {
  sorry
}

end minimize_segment_length_l487_487491


namespace sum_reciprocals_factors_of_12_l487_487683

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487683


namespace omega_range_l487_487365

-- Defining the function and its conditions
def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)

-- Main problem statement to prove the range of ω
theorem omega_range :
  ∀ ω φ : ℝ,
    (0 < ω) →
    (0 < φ ∧ φ < Real.pi / 2) →
    ((f 0 ω φ = Real.sqrt 2 / 2) ∧ 
    (∀ x1 x2 : ℝ, (x1 ≠ x2) → (x1 ∈ Set.Ioo  (Real.pi / 2) Real.pi) → 
                             (x2 ∈ Set.Ioo  (Real.pi / 2)  Real.pi) → 
                             (((x1 - x2) / (f x1 ω φ - f x2 ω φ)) < 0)) →
    (1 / 2 ≤ ω ∧ ω ≤ 5 / 4) :=
by
  intros ω φ ω_pos φ_range conditions
  sorry -- Proof is skipped here

end omega_range_l487_487365


namespace sum_reciprocal_factors_12_l487_487860

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487860


namespace correct_statements_l487_487042

open Real

-- Definition and conditions
variables {A B C a b c : ℝ} 
variable (Δ : (cos B / cos C) = (b / (2 * a - c)))
variable hA  : sin A ≠ 0

-- Statements to prove
def statement_A (h : Δ) : Prop := B = π / 3

def statement_B (h₁ : sin C = 2 * sin A) (area : Real := √3) : Prop := 
  let a := 2 in
  let \bigtriangleup := (a^2 * (sqrt 3) / 2) = 2 * sqrt 3 in
  a = 2

def statement_C (h₂ : b = 2 * sqrt 3) : Prop := 
  a > 2 * sqrt 3

def statement_D (h₂ : b = 2 * sqrt 3) : Prop := 
  (a + b + c) > 4 * sqrt 3 ∧ (a + b + c) ≤ 6 * sqrt 3

-- The theorem to prove
theorem correct_statements : 
  statement_A Δ ∧ 
  statement_B (sin C = 2 * sin A) (2 * sqrt 3) ∧ 
  ¬ statement_C (b = 2 * sqrt 3) ∧ 
  statement_D (b = 2 * sqrt 3) :=
  by
    split
    -- proof for each statement would follow, but is replaced by 'sorry' in this context
    {
        sorry
    }
    {
        sorry
    }
    {
        sorry
    }
    {
        sorry
    }

end correct_statements_l487_487042


namespace largest_power_of_two_factor_of_e_q_l487_487980

noncomputable def q : ℝ :=
  ∑ k in Finset.range 8, (k + 1 : ℝ) * Real.log (k + 1 : ℝ)

theorem largest_power_of_two_factor_of_e_q :
  let e_q := Real.exp q in
  ∃ n : ℕ, e_q = 2^40 * n ∧ n % 2 ≠ 0 :=
begin
  sorry
end

end largest_power_of_two_factor_of_e_q_l487_487980


namespace sum_reciprocals_factors_of_12_l487_487679

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487679


namespace sum_of_reciprocals_of_factors_of_12_l487_487804

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487804


namespace diagonal_BD_size_cos_A_value_l487_487032

noncomputable def AB := 250
noncomputable def CD := 250
noncomputable def angle_A := 120
noncomputable def angle_C := 120
noncomputable def AD := 150
noncomputable def BC := 150
noncomputable def perimeter := 800

/-- The size of the diagonal BD in isosceles trapezoid ABCD is 350, given the conditions -/
theorem diagonal_BD_size (AB CD AD BC : ℕ) (angle_A angle_C : ℝ) :
  AB = 250 → CD = 250 → AD = 150 → BC = 150 →
  angle_A = 120 → angle_C = 120 →
  ∃ BD : ℝ, BD = 350 :=
by
  sorry

/-- The cosine of angle A is -0.5, given the angle is 120 degrees -/
theorem cos_A_value (angle_A : ℝ) :
  angle_A = 120 → ∃ cos_A : ℝ, cos_A = -0.5 :=
by
  sorry

end diagonal_BD_size_cos_A_value_l487_487032


namespace hyperbola_eccentricity_l487_487526

theorem hyperbola_eccentricity (m : ℤ) (h : m^2 - 4 < 0) (h0 : m ≠ 0) :
  let a := 1 in let b := √3 in let c := √(a^2 + b^2) in c / a = 2 := sorry

end hyperbola_eccentricity_l487_487526


namespace sum_reciprocals_factors_12_l487_487742

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487742


namespace sum_of_reciprocals_of_factors_of_12_l487_487673

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487673


namespace rest_area_location_l487_487527

theorem rest_area_location :
  ∃ (rest_area : ℝ), rest_area = 35 + (95 - 35) / 2 :=
by
  -- Here we set the variables for the conditions
  let fifth_exit := 35
  let seventh_exit := 95
  let rest_area := 35 + (95 - 35) / 2
  use rest_area
  sorry

end rest_area_location_l487_487527


namespace sum_of_reciprocals_factors_12_l487_487781

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487781


namespace sum_reciprocals_of_factors_12_l487_487834

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487834


namespace solve_equation_l487_487287

noncomputable def fourthRoot (x : ℝ) := Real.sqrt (Real.sqrt x)

theorem solve_equation (x : ℝ) (hx : x ≥ 0) :
  fourthRoot x = 18 / (9 - fourthRoot x) ↔ x = 81 ∨ x = 1296 :=
by
  sorry

end solve_equation_l487_487287


namespace jill_sod_area_needed_l487_487048

def plot_width : ℕ := 200
def plot_length : ℕ := 50
def sidewalk_width : ℕ := 3
def sidewalk_length : ℕ := 50
def flower_bed1_depth : ℕ := 4
def flower_bed1_length : ℕ := 25
def flower_bed1_count : ℕ := 2
def flower_bed2_width : ℕ := 10
def flower_bed2_length : ℕ := 12
def flower_bed3_width : ℕ := 7
def flower_bed3_length : ℕ := 8

theorem jill_sod_area_needed :
  (plot_width * plot_length) - 
  (sidewalk_width * sidewalk_length + 
   flower_bed1_depth * flower_bed1_length * flower_bed1_count + 
   flower_bed2_width * flower_bed2_length + 
   flower_bed3_width * flower_bed3_length) = 9474 :=
by
  sorry

end jill_sod_area_needed_l487_487048


namespace intersection_of_A_and_B_l487_487340

theorem intersection_of_A_and_B :
  let A := {x : ℝ | -2 < x ∧ x < 3}
  let B := {x : ℝ | 0 < x}
  A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by
begin
  sorry
end

end intersection_of_A_and_B_l487_487340


namespace distances_from_A_to_tangents_l487_487106

variables {A B C D E : Type}
variables (p a b c : ℝ)
variables (triangle_ABC : Triangle A B C)
variables (tangent1 : Tangent (Incircle triangle_ABC) A B D)
variables (tangent2 : Tangent (Incircle triangle_ABC) A C E)

noncomputable def semiperimeter (a b c : ℝ) : ℝ :=
  (a + b + c) / 2

theorem distances_from_A_to_tangents 
  (triangle_ABC : Triangle A B C)
  (a_eq_bc : a = |B - C|)
  (p_def : p = semiperimeter a b c)
  (tangent1 : Tangent (Incircle triangle_ABC) A B D)
  (tangent2 : Tangent (Incircle triangle_ABC) A C E) :
  distance A D = p - a ∧ distance A E = p - a :=
sorry

end distances_from_A_to_tangents_l487_487106


namespace sum_reciprocal_factors_12_l487_487859

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487859


namespace solution_set_of_inequality_l487_487363

def f (x : ℝ) : ℝ :=
  if x > 1 then 2^x - x else 1

theorem solution_set_of_inequality :
  { x : ℝ | f x < f (2 / x) } = { x : ℝ | 0 < x ∧ x < Real.sqrt 2 } :=
by
  sorry

end solution_set_of_inequality_l487_487363


namespace total_students_in_circle_l487_487507

theorem total_students_in_circle (N : ℕ) (h1 : ∃ (students : Finset ℕ), students.card = N)
  (h2 : ∃ (a b : ℕ), a = 6 ∧ b = 16 ∧ b - a = N / 2): N = 18 :=
by
  sorry

end total_students_in_circle_l487_487507


namespace sum_reciprocal_factors_of_12_l487_487604

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487604


namespace sum_reciprocals_factors_12_l487_487752

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487752


namespace problem_min_value_l487_487468

noncomputable def min_value (x y : ℝ) : ℝ :=
  x^2 + y^2 + 4 / x^2 + y / x

theorem problem_min_value {x y : ℝ} (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (m : ℝ), m = sqrt 15 ∧ (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → min_value x y ≥ m) :=
sorry

end problem_min_value_l487_487468


namespace sum_reciprocals_factors_12_l487_487741

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487741


namespace sequence_abs_limit_l487_487475

open Complex

theorem sequence_abs_limit {z : ℕ → ℂ} {a : ℂ} (h : tendsto z at_top (𝓝 a)) :
  tendsto (fun n => abs (z n)) at_top (𝓝 (abs a)) :=
sorry

end sequence_abs_limit_l487_487475


namespace find_f8_l487_487336

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom odd_function (h_odd : ∀ x : ℝ, f (-x) = -f x)
axiom periodicity (h_period : ∀ x : ℝ, f (x + 2) = -f x)

-- Goal
theorem find_f8 (h_odd : ∀ x : ℝ, f (-x) = -f x) (h_period : ∀ x : ℝ, f (x + 2) = -f x) :
  f 8 = 0 :=
sorry

end find_f8_l487_487336


namespace no_opposite_identical_numbers_l487_487160

open Finset

theorem no_opposite_identical_numbers : 
  ∀ (f g : Fin 20 → Fin 20), 
  (∀ i : Fin 20, ∃ j : Fin 20, f j = i ∧ g j = (i + j) % 20) → 
  ∃ k : ℤ, ∀ i : Fin 20, f (i + k) % 20 ≠ g i 
  := by
    sorry

end no_opposite_identical_numbers_l487_487160


namespace sum_reciprocals_of_factors_12_l487_487841

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487841


namespace sector_properties_l487_487352

/--
Given a sector with central angle θ = 3/2 radians and radius r = 6 cm,
prove that the arc length is 9 cm and the area of the sector is 27 cm².
-/
theorem sector_properties (θ r : ℝ) (hθ : θ = 3 / 2) (hr : r = 6) :
  let l := θ * r in
  let S := 1 / 2 * l * r in
  l = 9 ∧ S = 27 :=
by
  sorry
  -- Proof is omitted

end sector_properties_l487_487352


namespace fraction_to_decimal_l487_487195

theorem fraction_to_decimal : (5 / 8 : ℝ) = 0.625 := 
  by sorry

end fraction_to_decimal_l487_487195


namespace partition_contains_prod_l487_487075

-- Definition of the set T
def T (n : ℕ) : Set ℕ := { m | 4 ≤ m ∧ m ≤ n }

-- Prove that for n = 1024, every partition of T into two subsets
-- contains integers x, y, z such that xy = z
theorem partition_contains_prod (n : ℕ) (h : n = 1024) :
  ∀ A B : Set ℕ, A ∪ B = T n → A ∩ B = ∅ → 
  ∃ x y z ∈ T(n), (x ∈ A ∨ x ∈ B) ∧ (y ∈ A ∨ y ∈ B) ∧ (z ∈ A ∨ z ∈ B) ∧ x * y = z := 
sorry

end partition_contains_prod_l487_487075


namespace sum_of_reciprocals_factors_12_l487_487819

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487819


namespace max_points_of_intersection_l487_487587

-- Definitions from the conditions
def circles := 2
def lines := 3

-- Define the problem of the greatest intersection number
theorem max_points_of_intersection (c : ℕ) (l : ℕ) (h_c : c = circles) (h_l : l = lines) : 
  (2 + (l * 2 * c) + (l * (l - 1) / 2)) = 17 :=
by
  rw [h_c, h_l]
  -- We have 2 points from circle intersections
  -- 12 points from lines intersections with circles
  -- 3 points from lines intersections with lines
  -- Hence, 2 + 12 + 3 = 17
  exact Eq.refl 17

end max_points_of_intersection_l487_487587


namespace angle_PCQ_45_degrees_l487_487508

theorem angle_PCQ_45_degrees (a b : ℝ) (h1 : 0 ≤ a ∧ a ≤ 1) (h2 : 0 ≤ b ∧ b ≤ 1) (h_perimeter : a + b + Real.sqrt(a^2 + b^2) = 2) :
  ∠((1, 1), (a, 0), (0, b) : ℝ × ℝ × ℝ) = 45 :=
sorry

end angle_PCQ_45_degrees_l487_487508


namespace min_sum_reciprocal_distances_iff_regular_tetrahedron_l487_487449

open Real

noncomputable def is_regular_tetrahedron (P : Fin 4 → ℝ × ℝ × ℝ) : Prop :=
  ∀ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l → dist (P i) (P j) = dist (P k) (P l)

theorem min_sum_reciprocal_distances_iff_regular_tetrahedron
  (P : Fin 4 → ℝ × ℝ × ℝ) (h : ∀ i, ∥P i - (0,0,0)∥ = 1) :
  (∀ Q : Fin 4 → ℝ × ℝ × ℝ, ∀ i, ∥Q i - (0,0,0)∥ = 1 → 
    (∑ i j, if i ≠ j then 1 / dist (Q i) (Q j) else 0) ≥ 
    ∑ i j, if i ≠ j then 1 / dist (P i) (P j) else 0) ↔ 
    is_regular_tetrahedron P :=
sorry

end min_sum_reciprocal_distances_iff_regular_tetrahedron_l487_487449


namespace probability_calculation_l487_487510

noncomputable def probability_same_color (pairs_black pairs_brown pairs_gray : ℕ) : ℚ :=
  let total_shoes := 2 * (pairs_black + pairs_brown + pairs_gray)
  let prob_black := (2 * pairs_black : ℚ) / total_shoes * (pairs_black : ℚ) / (total_shoes - 1)
  let prob_brown := (2 * pairs_brown : ℚ) / total_shoes * (pairs_brown : ℚ) / (total_shoes - 1)
  let prob_gray := (2 * pairs_gray : ℚ) / total_shoes * (pairs_gray : ℚ) / (total_shoes - 1)
  prob_black + prob_brown + prob_gray

theorem probability_calculation :
  probability_same_color 7 4 3 = 37 / 189 :=
by
  sorry

end probability_calculation_l487_487510


namespace sum_of_reciprocals_of_factors_of_12_l487_487913

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487913


namespace distinct_possible_values_l487_487451

noncomputable def distinctComplexNumber (c r s t : ℂ) : Prop :=
∀ z : ℂ, (z - r) * (z - s) * (z - t) = (z - c * r) * (z - c * s) * (z - c * t)

theorem distinct_possible_values (c : ℂ) (r s t : ℂ) 
  (h1 : distinctComplexNumber c r s t) 
  (h2 : r ≠ s) 
  (h3 : s ≠ t) 
  (h4 : t ≠ r) :
  ∃ (vals : Finset ℂ), set.finite { c | ∃ r s t : ℂ, distinctComplexNumber c r s t ∧ r ≠ s ∧ s ≠ t ∧ t ≠ r } ∧
    Finset.card { c | ∃ r s t : ℂ, distinctComplexNumber c r s t ∧ r ≠ s ∧ s ≠ t ∧ t ≠ r } = 4 :=
sorry

end distinct_possible_values_l487_487451


namespace solve_triangle_problem_l487_487043

noncomputable def triangle_problem (a b c S : ℝ) (A B C : ℝ) (h1 : a^2 * sin C = a * c * cos B * sin C + S)
  (h2 : b * sin C + c * sin B = 6 * sin B) 
  (area_eq : S = 1/2 * a * b * sin C) : Prop :=
  (C = π / 3) ∧ (a + b + c ≤ 9)

theorem solve_triangle_problem (a b c S : ℝ) (A B C : ℝ) (h1 : a^2 * sin C = a * c * cos B * sin C + S)
  (h2 : b * sin C + c * sin B = 6 * sin B) 
  (area_eq : S = 1/2 * a * b * sin C) : triangle_problem a b c S A B C h1 h2 area_eq :=
sorry

end solve_triangle_problem_l487_487043


namespace sum_reciprocal_factors_12_l487_487854

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487854


namespace pyramid_cross_section_area_l487_487523

theorem pyramid_cross_section_area 
  (side_base : ℝ) (height_pyramid : ℝ)
  (midline_pass : ∃ (M : ℝ × ℝ), M = (1 / 2 * side_base, 0))
  (perpendicular_base : ∃ (P : ℝ × ℝ), P = (M.1, height_pyramid))
  (side_base_eq : side_base = 6)
  (height_pyramid_eq : height_pyramid = 8) :
  let base_cross_section := 3 in
  let height_cross_section := 6 in
  (1 / 2) * base_cross_section * height_cross_section = 9 :=
by
  sorry

end pyramid_cross_section_area_l487_487523


namespace ratio_of_shaded_to_white_area_l487_487175

theorem ratio_of_shaded_to_white_area :
  (∃ (S : Type) (largest_square : S) (midpoint_subdivision : S -> S) (shaded_area white_area : ℝ),
   ∀ (sq : S), midpoint_subdivision sq = true ∧
               shaded_area = 5 * (area_of_one_small_triangle sq) ∧
               white_area = 3 * (area_of_one_small_triangle sq)) →
   shaded_area / white_area = 5 / 3 :=
sorry

end ratio_of_shaded_to_white_area_l487_487175


namespace odd_functions_identification_l487_487369

theorem odd_functions_identification :
  (∀ x : ℝ, (x^3) = -(-x)^3) ∧ (∀ x : ℝ, (sin x) = -(sin (-x))) :=
by
  sorry

end odd_functions_identification_l487_487369


namespace arithmetic_seq_problem_l487_487333

theorem arithmetic_seq_problem
  (a : ℕ → ℝ)
  (s : ℕ → ℝ)
  (h1 : a 10 = 13)
  (h2 : s 9 = 27)
  (h3 : ∀ n, s n = n * (a 1 + a n) / 2)
  (h4 : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1))
  : (∃ d, d = 2) ∧ (a 100 = 193) :=
begin
  sorry,
end

end arithmetic_seq_problem_l487_487333


namespace sum_reciprocals_factors_12_l487_487706

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487706


namespace negation_of_sin_equals_two_l487_487139

theorem negation_of_sin_equals_two :
  ¬(∃ x : ℝ, sin x = 2) ↔ ∀ x : ℝ, sin x ≠ 2 :=
by
  sorry

end negation_of_sin_equals_two_l487_487139


namespace sum_of_reciprocals_of_factors_of_12_l487_487769

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487769


namespace flight_duration_l487_487443

theorem flight_duration (takeoff landing : Nat)
  (h m : Nat) (h_pos : 0 < m) (m_lt_60 : m < 60)
  (time_takeoff : takeoff = 9 * 60 + 27)
  (time_landing : landing = 11 * 60 + 56)
  (flight_duration : (landing - takeoff) = h * 60 + m) :
  h + m = 31 :=
sorry

end flight_duration_l487_487443


namespace wire_cutting_problem_l487_487557

noncomputable def integer_wire_cuts (wire_length : ℕ) : (ℕ × ℕ) :=
  have h1 : wire_length = 150 := rfl
  have h2 : ∀ l : ℕ, l >= 1 := by sorry
  have h3 : ∀ (a b c : ℕ), ¬(a + b > c ∧ a + c > b ∧ b + c > a) := by sorry
  (10, 7)

theorem wire_cutting_problem : integer_wire_cuts 150 = (10, 7) :=
  by sorry

end wire_cutting_problem_l487_487557


namespace simplify_polynomial_l487_487504

theorem simplify_polynomial (q : ℤ) :
  (4*q^4 - 2*q^3 + 3*q^2 - 7*q + 9) + (5*q^3 - 8*q^2 + 6*q - 1) =
  4*q^4 + 3*q^3 - 5*q^2 - q + 8 :=
sorry

end simplify_polynomial_l487_487504


namespace sum_reciprocals_factors_12_l487_487738

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487738


namespace projection_onto_plane_l487_487060

open Matrix

noncomputable def normal_vector : Vector ℝ 3 := ![2, -1, 2]

noncomputable def projection_matrix : Matrix (Fin 3) (Fin 3) ℝ := 
![![5 / 9, 2 / 9, -4 / 9], 
  ![2 / 9, 8 / 9, 2 / 9], 
  ![-4 / 9, 2 / 9, 5 / 9]]

theorem projection_onto_plane (v : Vector ℝ 3) :
  let Q := proj ℝ (span ℝ (Set.range ![normal_vector])) in 
  projection_matrix.mul_vec v = Q v :=
sorry

end projection_onto_plane_l487_487060


namespace sum_reciprocal_factors_of_12_l487_487610

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487610


namespace sum_reciprocals_of_factors_12_l487_487849

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487849


namespace find_interest_rate_l487_487231

def principal := 428.57
def time := 4
def simple_interest := 60
def rate := 60 / (428.57 * 4)

theorem find_interest_rate : rate = 0.035 := by
  have h : rate = simple_interest / (principal * time) := rfl
  rw [h]
  norm_num
  sorry

end find_interest_rate_l487_487231


namespace sum_of_reciprocals_of_factors_of_12_l487_487796

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487796


namespace sum_reciprocals_factors_12_l487_487744

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487744


namespace volume_of_similar_cube_l487_487215

theorem volume_of_similar_cube (v : ℝ) (hv : v = 27) : 
  let side_length := real.sqrt (real.sqrt (v^3)) in
  let new_side_length := 2 * side_length in
  new_side_length^3 = 216 :=
by sorry

end volume_of_similar_cube_l487_487215


namespace constant_distance_between_circumcenters_l487_487331

variable {α : Type*} [EuclideanGeometry α]

-- Given a trapezoid ABCD with AD parallel to BC
variables {A B C D E : α} 
variables (h1 : ¬ Collinear ({A, B, C, D}))
variables (h2 : AD ⊥ BC)

-- Point E is arbitrary on AB
variables (E : α)
variables (h3 : E ∈ LineSegment A B)

-- Define circumcenters of triangles ADE and BCE
def circumcenter {A B C : α} (h : ¬Collinear ({A, B, C})) : α := sorry

-- circumcenters of ADE and BCE
noncomputable def O1 := circumcenter ({A, D, E}) h1
noncomputable def O2 := circumcenter ({B, C, E}) h1

-- The distance between the circumcenters O1 and O2 is constant
theorem constant_distance_between_circumcenters 
  (hA : is_trapezoid A B C D)
  (hE : E ∈ LineSegment A B) :
  ∃ k : ℝ, ∀ (E' : α), E' ∈ LineSegment A B → dist (O1 A D E' h1) (O2 B C E' h1) = k :=
sorry

end constant_distance_between_circumcenters_l487_487331


namespace total_toothpicks_needed_l487_487221

theorem total_toothpicks_needed (length width : ℕ) (hl : length = 50) (hw : width = 40) : 
  (length + 1) * width + (width + 1) * length = 4090 := 
by
  -- proof omitted, replace this line with actual proof
  sorry

end total_toothpicks_needed_l487_487221


namespace sum_of_coordinates_D_l487_487495

def point := ℝ × ℝ

def midpoint (A B: point) : point :=
((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem sum_of_coordinates_D :
  ∀ (C D M: point), 
  M = (4, 9) → C = (10, 5) → midpoint C D = M → (D.1 + D.2) = 11 := 
by
  intros C D M hM hC hMid
  sorry

end sum_of_coordinates_D_l487_487495


namespace sum_of_reciprocals_factors_12_l487_487778

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487778


namespace find_cone_radius_l487_487359

-- Conditions
def surface_area (r l : ℝ) : ℝ := π * r^2 + π * r * l

def cone_radius_satisfies_condition (r : ℝ) : Prop :=
  ∃ l, l = 2 * r ∧ surface_area r l = 3 * π

-- Problem Statement
theorem find_cone_radius (r : ℝ) :
  cone_radius_satisfies_condition r → r = 1 :=
sorry

end find_cone_radius_l487_487359


namespace completing_square_transformation_l487_487983

theorem completing_square_transformation (x : ℝ) :
  x^2 - 2 * x - 5 = 0 -> (x - 1)^2 = 6 :=
by {
  sorry -- Proof to be completed
}

end completing_square_transformation_l487_487983


namespace product_ab_is_six_l487_487350

theorem product_ab_is_six (a b : ℝ) (A B : Set ℝ)
  (hA : A = { -1, 3 })
  (hB : B = { x | x^2 + a * x + b = 0 })
  (hAB : A = B) :
  a * b = 6 :=
begin
  sorry
end

end product_ab_is_six_l487_487350


namespace sum_reciprocals_factors_12_l487_487632

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487632


namespace sector_area_and_fraction_l487_487291

theorem sector_area_and_fraction 
  (r : ℝ) 
  (θ : ℝ) 
  (h_r : r = 25) 
  (h_θ : θ = 47.3) :
  let area_of_sector := (θ / 360) * π * r^2 in
  let total_circle_area := π * r^2 in
  let area_approx := 809 in
  let fraction := 473 / 3600 in
  (abs (area_of_sector - 809) < 1) ∧ (area_of_sector / total_circle_area = fraction) := 
  by
  sorry

end sector_area_and_fraction_l487_487291


namespace correct_option_C_l487_487541

def number_of_stamps : String := "the number of the stamps"
def number_of_people : String := "a number of people"

def is_singular (subject : String) : Prop := subject = number_of_stamps
def is_plural (subject : String) : Prop := subject = number_of_people

def correct_sentence (verb1 verb2 : String) : Prop :=
  verb1 = "is" ∧ verb2 = "want"

theorem correct_option_C : correct_sentence "is" "want" :=
by
  show correct_sentence "is" "want"
  -- Proof is omitted
  sorry

end correct_option_C_l487_487541


namespace imaginary_part_of_z_l487_487135

open Complex

-- Definition of the complex number as per the problem statement
def z : ℂ := (2 - 3 * Complex.I) * Complex.I

-- The theorem stating that the imaginary part of the given complex number is 2
theorem imaginary_part_of_z : z.im = 2 :=
by
  sorry

end imaginary_part_of_z_l487_487135


namespace proof_of_fifth_and_subsequent_sequences_l487_487086

/-- 
Given the conditions:
1. Sequence defined by \( x_n = n^2 \)
2. First 10 terms: \[1, 4, 9, 16, 25, 36, 49, 64, 81, 100\]
3. First sequence of differences: \[3, 5, 7, 9, 11, 13, 15, 17, 19\]
4. Second sequence of differences: \[2, 2, 2, 2, 2, 2, 2, 2\]
5. Third sequence of differences: \[0, 0, 0, 0, 0, 0, 0\]

We need to prove:
- The fourth, fifth, and subsequent sequences are all zeros.
-/
def fifth_and_subsequent_sequences_are_zero: Prop :=
∀ n : ℕ, n ≥ 4 → nth_diff n (λ k, k^2) = 0

theorem proof_of_fifth_and_subsequent_sequences: fifth_and_subsequent_sequences_are_zero := 
sorry

end proof_of_fifth_and_subsequent_sequences_l487_487086


namespace bobArrivesBefore845Prob_l487_487242

noncomputable def probabilityBobBefore845 (totalTime: ℕ) (cutoffTime: ℕ) : ℚ :=
  let totalArea := (totalTime * totalTime) / 2
  let areaOfInterest := (cutoffTime * cutoffTime) / 2
  (areaOfInterest : ℚ) / totalArea

theorem bobArrivesBefore845Prob (totalTime: ℕ) (cutoffTime: ℕ) (ht: totalTime = 60) (hc: cutoffTime = 45) :
  probabilityBobBefore845 totalTime cutoffTime = 9 / 16 := by
  sorry

end bobArrivesBefore845Prob_l487_487242


namespace sum_reciprocals_factors_12_l487_487708

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487708


namespace sum_reciprocals_12_l487_487915

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487915


namespace sum_of_reciprocals_of_factors_of_12_l487_487770

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487770


namespace line_circle_intersection_probability_l487_487006

noncomputable def probability_intersect_line_circle (m : ℝ) : ℝ :=
  if (-3 ≤ m ∧ m ≤ 3) then if abs (1 + 2 + m) / Real.sqrt 2 ≤ 2 then 1 else 0 else 0

theorem line_circle_intersection_probability :
  (interval_integral (λ m, probability_intersect_line_circle m) (-3) (3)) / 6 = (Real.sqrt 2) / 3 := sorry

end line_circle_intersection_probability_l487_487006


namespace students_not_pictured_after_gym_class_l487_487552

theorem students_not_pictured_after_gym_class 
  (total_students : ℕ) 
  (before_break_fraction : ℕ) 
  (morning_break_fraction : ℕ) 
  (additional_students_after_lunch : ℕ) 
  (before_break : total_students / 4 = before_break_fraction) 
  (remaining_after_first : total_students - before_break_fraction = total_students - 15) 
  (during_break : (total_students - 15) / 3 = morning_break_fraction) 
  (additional_after_lunch : additional_students_after_lunch = 10)
  (total_pictured : before_break_fraction + morning_break_fraction + additional_students_after_lunch = 40) 
  (total_students = 60) 
  : total_students - total_pictured = 20 :=
by
  sorry

end students_not_pictured_after_gym_class_l487_487552


namespace min_value_of_expression_l487_487407

theorem min_value_of_expression (k : ℝ) (h : ∃ x y: ℝ, x ≠ y ∧ x^2 + 2*k*x + k^2 + k + 3 = 0 ∧ y^2 + 2*k*y + k^2 + k + 3 = 0) : 
  ∃ k : ℝ, k ≤ -3 ∧ is_min (λ k, k^2 + k + 3) 9 :=
by sorry


end min_value_of_expression_l487_487407


namespace dot_product_eq_negative_29_l487_487379

def vector := ℝ × ℝ

variables (a b : vector)

theorem dot_product_eq_negative_29 
  (h1 : a + b = (2, -4))
  (h2 : 3 * a - b = (-10, 16)) :
  a.1 * b.1 + a.2 * b.2 = -29 :=
sorry

end dot_product_eq_negative_29_l487_487379


namespace vector_identity_l487_487201

-- Define vectors A, B, and C in a vector space
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C : V)

-- Define the vectors AB, AC, and CB
def AB := B - A
def AC := C - A
def CB := B - C

-- Proof statement
theorem vector_identity : AC - AB + CB = (0 : V) :=
by
  rw [AC, AB, CB],
  -- Simplify the expression using existing identities and given operations
  sorry

end vector_identity_l487_487201


namespace current_speed_l487_487149

-- Define the constants based on conditions
def rowing_speed_kmph : Float := 24
def distance_meters : Float := 40
def time_seconds : Float := 4.499640028797696

-- Intermediate calculation: Convert rowing speed from km/h to m/s
def rowing_speed_mps : Float := rowing_speed_kmph * 1000 / 3600

-- Calculate downstream speed
def downstream_speed_mps : Float := distance_meters / time_seconds

-- Define the expected speed of the current
def expected_current_speed : Float := 2.22311111

-- The theorem to prove
theorem current_speed : 
  (downstream_speed_mps - rowing_speed_mps) = expected_current_speed :=
by 
  -- skipping the proof steps, as instructed
  sorry

end current_speed_l487_487149


namespace sum_reciprocals_factors_12_l487_487639

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487639


namespace sum_reciprocals_12_l487_487914

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487914


namespace sum_reciprocals_factors_12_l487_487633

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487633


namespace positive_integers_with_conditions_l487_487283

theorem positive_integers_with_conditions :
  {n : ℕ | ∃ (d : Fin 16 → ℕ), has_exactly_n_divisors n 16 ∧ d 5 = 18 ∧ (d 8 - d 7 = 17)
    ∧ (∀ i j, i < j → d i < d j) ∧ (∀ i, 1 ≤ d i ∧ d i ≤ n)} = {1998, 3834} :=
  sorry

def has_exactly_n_divisors (n d_count : ℕ) : Prop :=
  ∃ d : Fin d_count → ℕ, (∀ i, n % d i = 0) ∧ (∀ i, (d i)^(d_count / 2) = n) ∧ 
                            (∀ i j, i < j → d i < d j)

end positive_integers_with_conditions_l487_487283


namespace chalkboard_area_l487_487480

theorem chalkboard_area (width : ℝ) (h_w : width = 3) (h_l : 2 * width = length) : width * length = 18 := 
by 
  sorry

end chalkboard_area_l487_487480


namespace sum_of_reciprocals_of_factors_of_12_l487_487901

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487901


namespace sum_reciprocal_factors_12_l487_487861

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487861


namespace large_denominator_of_fractions_l487_487130

theorem large_denominator_of_fractions
  (b d a c : ℕ)
  (h1 : Nat.gcd b a = 1)
  (h2 : Nat.gcd d c = 1)
  (h3 : ∀ (k : ℕ), 1 ≤ k ∧ k ≤ 4999 → (λ differing_positions, 500000 ≤ differing_positions ∧ differing_positions ≤ 1000000))
  : max a c > 10^50 := sorry

end large_denominator_of_fractions_l487_487130


namespace max_intersections_l487_487578

-- Define the number of circles and lines
def num_circles : ℕ := 2
def num_lines : ℕ := 3

-- Define the maximum number of intersection points of circles
def max_circle_intersections : ℕ := 2

-- Define the number of intersection points between each line and each circle
def max_line_circle_intersections : ℕ := 2

-- Define the number of intersection points among lines (using the combination formula)
def num_line_intersections : ℕ := (num_lines.choose 2)

-- Define the greatest number of points of intersection
def total_intersections : ℕ :=
  max_circle_intersections + (num_lines * num_circles * max_line_circle_intersections) + num_line_intersections

-- Prove the greatest number of points of intersection is 17
theorem max_intersections : total_intersections = 17 := by
  -- Calculating individual parts for clarity
  have h1: max_circle_intersections = 2 := rfl
  have h2: num_lines * num_circles * max_line_circle_intersections = 12 := by
    calc
      num_lines * num_circles * max_line_circle_intersections
        = 3 * 2 * 2 := by rw [num_lines, num_circles, max_line_circle_intersections]
        ... = 12 := by norm_num
  have h3: num_line_intersections = 3 := by
    calc
      num_line_intersections = (3.choose 2) := rfl
      ... = 3 := by norm_num

  -- Adding the parts to get the total intersections
  calc
    total_intersections
      = max_circle_intersections + (num_lines * num_circles * max_line_circle_intersections) + num_line_intersections := rfl
      ... = 2 + 12 + 3 := by rw [h1, h2, h3]
      ... = 17 := by norm_num

end max_intersections_l487_487578


namespace sum_of_decimals_is_666_l487_487163

noncomputable def sum_of_decimals (a b c d : ℕ) (h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : ℝ :=
  let digits_sum := a + b + c + d
  6 * digits_sum * (0.1 + 0.01 + 0.001 + 0.0001)

theorem sum_of_decimals_is_666.6 (a b c d : ℕ) (h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  sum_of_decimals a b c d h = 666.6 :=
by
  sorry

end sum_of_decimals_is_666_l487_487163


namespace sum_reciprocals_of_factors_of_12_l487_487965

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487965


namespace max_intersections_two_circles_three_lines_l487_487569

theorem max_intersections_two_circles_three_lines :
  ∀ (C1 C2 : ℝ × ℝ × ℝ) (L1 L2 L3 : ℝ × ℝ × ℝ), 
  C1 ≠ C2 → L1 ≠ L2 → L2 ≠ L3 → L1 ≠ L3 →
  ∃ (P : ℕ), P = 17 :=
by 
  sorry

end max_intersections_two_circles_three_lines_l487_487569


namespace cos_of_sin_in_second_quadrant_l487_487346

theorem cos_of_sin_in_second_quadrant (α : ℝ) (h1 : α > π / 2 ∧ α < π) (h2 : sin α = 5 / 13) :
  cos α = -12 / 13 :=
sorry

end cos_of_sin_in_second_quadrant_l487_487346


namespace sapid_function_min_j17_l487_487245

-- Mathematical definitions
def is_sapid (h : ℕ → ℤ) : Prop :=
  ∀ (x y : ℕ), 0 < x → 0 < y → h(x) + h(y) > 2 * y^2

def minimized_sapid_sum (j : ℕ → ℤ) : Prop :=
  is_sapid j ∧ (∀ k : ℕ → ℤ, is_sapid k → (∑ i in finset.range 30, j(i + 1)) ≤ (∑ i in finset.range 30, k(i + 1)))

-- The theorem we need to prove
theorem sapid_function_min_j17 (j : ℕ → ℤ) (h1 : minimized_sapid_sum j) : 
  j 17 = 256 := 
sorry

end sapid_function_min_j17_l487_487245


namespace find_a_minus_2b_l487_487463

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
if -1 ≤ x ∧ x < 0 then a * x + 1 else
if 0 ≤ x ∧ x ≤ 1 then (b * x + 2) / (x + 1) else
f (x - 2) a b

theorem find_a_minus_2b (a b : ℝ) 
  (h_periodic : ∀ x, f (x) a b = f (x - 2) a b) 
  (h_defined : ∀ x, 
    ((-1 ≤ x ∧ x < 0) → f (x) a b = a * x + 1) ∧ 
    ((0 ≤ x ∧ x ≤ 1) → f (x) a b = (b * x + 2) / (x + 1)))
  (h_value : f (1/2) a b = f (3/2) a b) : 
  a - 2 * b = 42 :=
sorry

end find_a_minus_2b_l487_487463


namespace Parallelepiped_Problem_l487_487471

variables {u v w p : ℝ^3} -- assume all vectors are in ℝ³

noncomputable def AG2 := (u + v + w)•(u + v + w)
noncomputable def BH2 := (u - v + w)•(u - v + w)
noncomputable def CE2 := (-u + v + w)•(-u + v + w)
noncomputable def DF2 := (u + v - w)•(u + v - w)

noncomputable def AP2 := (v + p)•(v + p)
noncomputable def AD2 := w•w
noncomputable def AE2 := u•u

theorem Parallelepiped_Problem :
  (AG2 + BH2 + CE2 + DF2) = 4 * (u•u + v•v + w•w) →
  (AP2 + AD2 + AE2) = v•v + p•p + 2 * (v•p) + w•w + u•u →
  ∀ (AG2 BH2 CE2 DF2 AP2 AD2 AE2 : ℝ),
  (AG2 + BH2 + CE2 + DF2) / (AP2 + AD2 + AE2) = 4
by
  sorry

end Parallelepiped_Problem_l487_487471


namespace bisects_perimeter_lines_intersect_at_one_point_l487_487998

-- Define points and conditions
variables (A B C E C1 F : Type) 
variables (midpoint_arc_E : Prop) -- E is midpoint of arc AB on the circumcircle containing C
variables (midpoint_C1 : Prop) -- C1 is midpoint of side AB
variables (perpendicular_EF : Prop) -- EF ⊥ AC and F is on AC
variables [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq E] [DecidableEq C1] [DecidableEq F]

-- Define part (a)
theorem bisects_perimeter (h₁: midpoint_arc_E) (h₂: midpoint_C1) (h₃: perpendicular_EF) :
  ∃C1 F, bisects_perimeter ABC := sorry

-- Define part (b)
theorem lines_intersect_at_one_point (h₁: midpoint_arc_E) (h₂: midpoint_C1) (h₃: perpendicular_EF) :
  ∃L M N, bisects_each_side ABC L M N ∧ lines_intersect_at_one_point L M N := sorry

end bisects_perimeter_lines_intersect_at_one_point_l487_487998


namespace circumcircle_radius_of_isosceles_triangle_l487_487136

theorem circumcircle_radius_of_isosceles_triangle (a b : ℝ) (h : 4 * a^2 > b^2) :
  ∃ R : ℝ, R = a^2 / (sqrt (4 * a^2 - b^2)) :=
begin
  use a^2 / (sqrt (4 * a^2 - b^2)),
  sorry -- proof to be filled in
end

end circumcircle_radius_of_isosceles_triangle_l487_487136


namespace sum_of_reciprocals_factors_12_l487_487720

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487720


namespace sum_of_reciprocals_of_factors_of_12_l487_487662

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487662


namespace sum_reciprocals_12_l487_487919

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487919


namespace find_g_3_l487_487134

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_condition : ∀ x ≠ 0, 4 * g(x) - 3 * g(1 / x) = x ^ 2

theorem find_g_3 : g 3 = 5.190 := sorry

end find_g_3_l487_487134


namespace polynomial_irreducible_over_rat_l487_487107

theorem polynomial_irreducible_over_rat (n m : ℤ) (h : n > m ∧ m > 0) :
  irreducible (⟨X ^ n + X ^ m - 2, X ^ Int.gcd m (Int.natAbs n) - 1⟩ : Rat) :=
by sorry

end polynomial_irreducible_over_rat_l487_487107


namespace sum_of_reciprocals_factors_12_l487_487828

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487828


namespace conjugate_of_z_l487_487319

open Complex

theorem conjugate_of_z : 
  let z : ℂ := (1 + 2 * I) / I in
  conj z = 2 + I :=
by
  sorry

end conjugate_of_z_l487_487319


namespace identify_counterfeits_n2_identify_counterfeits_n3_l487_487439

-- Definitions based on problem conditions
variables {α : Type*} [linear_ordered_field α]

-- Condition: There are distinct natural denominations
def distinct_denominations (denoms : list α) : Prop :=
  denoms.nodup

-- Condition: There are exactly N counterfeit banknotes
def has_n_counterfeits (denoms : list α) (N : ℕ) (counterfeits : list α) : Prop :=
  N = counterfeits.length ∧ ∀ c ∈ counterfeits, c ∈ denoms

-- Condition: Detector checks sum of denominations of genuine notes in a selected set
def valid_check (denoms : list α) (selected : list α) : Prop :=
  ∀ x ∈ selected, x ∈ denoms

-- Question: Identify all counterfeit banknotes in N checks

-- For N = 2
theorem identify_counterfeits_n2 (denoms : list α) (counterfeits : list α) (detector : list α → α) :
  distinct_denominations denoms →
  has_n_counterfeits denoms 2 counterfeits →
  (∀ denotes_subset, valid_check denoms denotes_subset → (detector denotes_subset = detector denotes_subset)) →
  ∃ (counterfeit1 counterfeit2 : α), 
    counterfeit1 ∈ counterfeits ∧
    counterfeit2 ∈ counterfeits ∧
    counterfeit1 ≠ counterfeit2 :=
sorry

-- For N = 3
theorem identify_counterfeits_n3 (denoms : list α) (counterfeits : list α) (detector : list α → α) :
  distinct_denominations denoms →
  has_n_counterfeits denoms 3 counterfeits →
  (∀ denotes_subset, valid_check denoms denotes_subset → (detector denotes_subset = detector denotes_subset)) →
  ∃ (counterfeit1 counterfeit2 counterfeit3 : α), 
    counterfeit1 ∈ counterfeits ∧
    counterfeit2 ∈ counterfeits ∧
    counterfeit3 ∈ counterfeits ∧
    counterfeit1 ≠ counterfeit2 ∧
    counterfeit1 ≠ counterfeit3 ∧
    counterfeit2 ≠ counterfeit3 :=
sorry

end identify_counterfeits_n2_identify_counterfeits_n3_l487_487439


namespace baking_time_one_batch_l487_487258

theorem baking_time_one_batch (x : ℕ) (time_icing_per_batch : ℕ) (num_batches : ℕ) (total_time : ℕ)
  (h1 : num_batches = 4)
  (h2 : time_icing_per_batch = 30)
  (h3 : total_time = 200)
  (h4 : total_time = num_batches * x + num_batches * time_icing_per_batch) :
  x = 20 :=
by
  rw [h1, h2, h3] at h4
  sorry

end baking_time_one_batch_l487_487258


namespace negation_of_odd_implication_l487_487138

-- Defining what it means for a function to be odd
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Statement of the problem
theorem negation_of_odd_implication (f : ℝ → ℝ) :
  ¬ (is_odd_function f → is_odd_function (λ x, f (-x))) ↔ 
  (¬is_odd_function f → ¬is_odd_function (λ x, f (-x))) :=
by sorry

end negation_of_odd_implication_l487_487138


namespace train_cross_bridge_time_l487_487222

variable (bridge_length train_length train_speed : ℕ)
variable (man_cross_time : ℕ)

-- The lengths and speed
axiom bridge_length_eq : bridge_length = 180
axiom train_length_eq : train_length = 120
axiom train_speed_eq : train_speed = 15
axiom man_cross_time_eq : man_cross_time = 8

-- Prove the time taken for the train to cross the bridge
theorem train_cross_bridge_time :
  let D := train_length + bridge_length,
      S := train_speed,
      T := D / S in
      T = 20 :=
by
  rw [train_length_eq, bridge_length_eq, train_speed_eq]
  let D := 120 + 180
  let S := 15
  let T := D / S
  show T = 20
  sorry

end train_cross_bridge_time_l487_487222


namespace moon_speed_conversion_l487_487199

noncomputable def moon_speed_per_s := 1.02  -- speed in km per second
def seconds_per_hour := 3600  -- number of seconds in an hour
def moon_speed_per_h := moon_speed_per_s * seconds_per_hour  -- speed in km per hour

theorem moon_speed_conversion : moon_speed_per_h = 3672 := by
  -- This is the part where we acknowledge the need for a proof
  sorry

end moon_speed_conversion_l487_487199


namespace sufficient_not_necessary_condition_l487_487273

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Iic (-2) → (x^2 + 2 * a * x - 2) ≤ ((x - 1)^2 + 2 * a * (x - 1) - 2)) ↔ a ≤ 2 := by
  sorry

end sufficient_not_necessary_condition_l487_487273


namespace line_through_intersection_points_of_circles_l487_487376

theorem line_through_intersection_points_of_circles :
  ∀ (x y : ℝ),
    (x^2 + y^2 + 4*x - 4*y - 1 = 0) ∧ (x^2 + y^2 + 2*x - 13 = 0) →
    (x - 2*y + 6 = 0) :=
by
  intro x y h
  -- Condition of circle 1
  have circle1 : x^2 + y^2 + 4*x - 4*y - 1 = 0 := h.left
  -- Condition of circle 2
  have circle2 : x^2 + y^2 + 2*x - 13 = 0 := h.right
  sorry

end line_through_intersection_points_of_circles_l487_487376


namespace knives_more_than_forks_l487_487551

variable (F K S T : ℕ)
variable (x : ℕ)

-- Initial conditions
def initial_conditions : Prop :=
  (F = 6) ∧ 
  (K = F + x) ∧ 
  (S = 2 * K) ∧
  (T = F / 2)

-- Total cutlery added
def total_cutlery_added : Prop :=
  (F + 2) + (K + 2) + (S + 2) + (T + 2) = 62

-- Prove that x = 9
theorem knives_more_than_forks :
  initial_conditions F K S T x →
  total_cutlery_added F K S T →
  x = 9 := 
by
  sorry

end knives_more_than_forks_l487_487551


namespace sum_of_reciprocals_of_factors_of_12_l487_487666

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487666


namespace sum_of_reciprocals_of_factors_of_12_l487_487764

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487764


namespace sum_reciprocals_factors_12_l487_487950

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487950


namespace sum_reciprocals_of_factors_of_12_l487_487954

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487954


namespace sum_reciprocals_factors_12_l487_487644

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487644


namespace sum_reciprocals_of_factors_of_12_l487_487972

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487972


namespace sum_reciprocals_factors_12_l487_487940

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487940


namespace least_n_fac_divides_2200_l487_487588

theorem least_n_fac_divides_2200 (n : ℕ) (h : n ≥ 11) : 2200 ∣ n! :=
by {
  have prime_fact_2200 : 2200 = 2^2 * 5^2 * 11 := by norm_num,
  sorry
}

end least_n_fac_divides_2200_l487_487588


namespace factorization_example_l487_487192

theorem factorization_example : 
  ∀ (a : ℝ), a^2 - 6 * a + 9 = (a - 3)^2 :=
by
  intro a
  sorry

end factorization_example_l487_487192


namespace find_k_value_l487_487357

theorem find_k_value (k : ℝ) :
  (∃ x1 x2 : ℝ, (2 * x1^2 + k * x1 - 2 * k + 1 = 0) ∧ 
                (2 * x2^2 + k * x2 - 2 * k + 1 = 0) ∧ 
                (x1 ≠ x2)) ∧
  ((x1^2 + x2^2 = 29/4)) ↔ (k = 3) := 
sorry

end find_k_value_l487_487357


namespace sum_of_reciprocals_of_factors_of_12_l487_487808

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487808


namespace sum_reciprocals_of_factors_of_12_l487_487971

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487971


namespace sum_reciprocals_of_factors_of_12_l487_487969

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487969


namespace quotient_is_correct_l487_487100

def dividend := 15698
def divisor := 176.22471910112358
def remainder := 14
def quotient := dividend - remainder

theorem quotient_is_correct : quotient / divisor = 89 := by
  sorry

end quotient_is_correct_l487_487100


namespace percentage_excess_calculation_l487_487422

def V : ℕ := 9720
def IV := (20 * V) / 100
def VV := V - IV
def VB : ℕ := 3159
def VA := VV - VB
def percentage_excess := ((VA - VB) * 100) / VB

theorem percentage_excess_calculation
  (h_V : V = 9720)
  (h_IV : IV = (20 * V) / 100)
  (h_VV : VV = V - IV)
  (h_VB : VB = 3159)
  (h_VA : VA = VV - VB):
  percentage_excess ≈ 46.15 :=
by
    sorry

end percentage_excess_calculation_l487_487422


namespace geometric_sequence_first_term_l487_487324

theorem geometric_sequence_first_term (n : ℕ) (a : ℕ → ℝ) (q : ℝ) :
  let m := 2 * n + 1 in
  (∑ i in Finset.range (n + 1), a (2 * i)) = 255 ∧
  (∑ i in Finset.range n, a (2 * i + 1)) = -126 ∧
  a m = 192 →
  a 1 = 3 :=
sorry

end geometric_sequence_first_term_l487_487324


namespace ellipse_equation_correct_exists_fixed_point_l487_487334

-- Variables and initial conditions
def a := 2
def b := sqrt 3
def c := 1
def F1 := (-1, 0)
def F2 := (1, 0)
def ellipse_equation (x y : ℝ) : Prop := 
  (x^2 / (a^2)) + (y^2 / b^2) = 1

theorem ellipse_equation_correct (x y : ℝ) :
  ellipse_equation x y ↔ x^2 / 4 + y^2 / 3 = 1 :=
sorry

theorem exists_fixed_point :
  ∃ M : ℝ × ℝ, M = (-4, 0) ∧ 
  ∀ (A B : ℝ × ℝ) (k : ℝ), 
  (A ≠ B) ∧ 
  (∃ (x1 y1 : ℝ), A = (x1, y1) ∧ ellipse_equation x1 y1) ∧
  (∃ (x2 y2 : ℝ), B = (x2, y2) ∧ ellipse_equation x2 y2) ∧
  let MA := (fst A - -1)^2 + (snd A - 0)^2,
      MB := (fst B - -1)^2 + (snd B - 0)^2 in
  MA + MB = (fst A - k)^2 + (snd A - snd M)^2 :=
sorry

end ellipse_equation_correct_exists_fixed_point_l487_487334


namespace sum_reciprocal_factors_12_l487_487872

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487872


namespace quadratic_eq_with_given_roots_l487_487011

theorem quadratic_eq_with_given_roots (a b : ℝ) (h1 : (a + b) / 2 = 8) (h2 : Real.sqrt (a * b) = 12) :
    (a + b = 16) ∧ (a * b = 144) ∧ (∀ (x : ℝ), x^2 - (a + b) * x + (a * b) = 0 ↔ x^2 - 16 * x + 144 = 0) := by
  sorry

end quadratic_eq_with_given_roots_l487_487011


namespace sequence_value_is_correct_l487_487040

theorem sequence_value_is_correct (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 2) : a 8 = 15 :=
sorry

end sequence_value_is_correct_l487_487040


namespace compare_logs_l487_487460

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.log 5 / Real.log (1 / 2)

theorem compare_logs : c < a ∧ a < b := by
  have h0 : a = Real.log 2 / Real.log 3 := rfl
  have h1 : b = Real.log 3 / Real.log 2 := rfl
  have h2 : c = Real.log 5 / Real.log (1 / 2) := rfl
  sorry

end compare_logs_l487_487460


namespace julia_played_tag_l487_487052

/-
Problem:
Let m be the number of kids Julia played with on Monday.
Let t be the number of kids Julia played with on Tuesday.
m = 24
m = t + 18
Show that t = 6
-/

theorem julia_played_tag (m t : ℕ) (h1 : m = 24) (h2 : m = t + 18) : t = 6 :=
by
  sorry

end julia_played_tag_l487_487052


namespace sum_of_reciprocals_of_factors_of_12_l487_487767

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487767


namespace sum_of_reciprocals_of_factors_of_12_l487_487896

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487896


namespace sum_of_reciprocals_of_factors_of_12_l487_487810

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487810


namespace Sarah_consumed_one_sixth_l487_487116

theorem Sarah_consumed_one_sixth (total_slices : ℕ) (slices_sarah_ate : ℕ) (shared_slices : ℕ) :
  total_slices = 20 → slices_sarah_ate = 3 → shared_slices = 1 → 
  ((slices_sarah_ate + shared_slices / 3) / total_slices : ℚ) = 1 / 6 :=
by
  intros h1 h2 h3
  sorry

end Sarah_consumed_one_sixth_l487_487116


namespace find_a_and_an_l487_487015

-- Given Sequences
def S (n : ℕ) (a : ℝ) : ℝ := 3^n - a

def is_geometric_sequence (a_n : ℕ → ℝ) : Prop := ∃ a1 q, q ≠ 1 ∧ ∀ n, a_n n = a1 * q^n

-- The main statement
theorem find_a_and_an (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (a : ℝ) :
  (∀ n, S_n n = 3^n - a) ∧ is_geometric_sequence a_n →
  ∃ a, a = 1 ∧ ∀ n, a_n n = 2 * 3^(n-1) :=
by
  sorry

end find_a_and_an_l487_487015


namespace sum_reciprocal_factors_of_12_l487_487607

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487607


namespace pencils_given_out_l487_487553
-- Define the problem conditions
def students : ℕ := 96
def dozens_per_student : ℕ := 7
def pencils_per_dozen : ℕ := 12

-- Define the expected total pencils
def expected_pencils : ℕ := 8064

-- Define the statement to be proven
theorem pencils_given_out : (students * (dozens_per_student * pencils_per_dozen)) = expected_pencils := 
  by
  sorry

end pencils_given_out_l487_487553


namespace additional_books_l487_487102

theorem additional_books (initial_books total_books additional_books : ℕ)
  (h_initial : initial_books = 54)
  (h_total : total_books = 77) :
  additional_books = total_books - initial_books :=
by
  sorry

end additional_books_l487_487102


namespace sum_reciprocals_factors_12_l487_487700

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487700


namespace globe_properties_l487_487127

noncomputable def radius_of_globe (circumference_lat_line : ℝ) : ℝ :=
  let radius_lat := circumference_lat_line / (2 * Real.pi)
  in radius_lat / Real.cos (Real.pi / 3)

theorem globe_properties (circumference_lat60N : ℝ) (radius_globe surface_area_globe : ℝ) :
  circumference_lat60N = 4 * Real.pi →
  radius_globe = radius_of_globe circumference_lat60N →
  surface_area_globe = 4 * Real.pi * radius_globe^2 →
  radius_globe = 4 ∧ surface_area_globe = 64 * Real.pi :=
by
  intros h1 h2 h3
  sorry

end globe_properties_l487_487127


namespace smaller_factor_of_4851_l487_487145

-- Define the condition
def product_lim (m n : ℕ) : Prop := m * n = 4851 ∧ 10 ≤ m ∧ m < 100 ∧ 10 ≤ n ∧ n < 100

-- The lean theorem statement
theorem smaller_factor_of_4851 : ∃ m n : ℕ, product_lim m n ∧ m = 49 := 
by {
    sorry
}

end smaller_factor_of_4851_l487_487145


namespace sum_reciprocals_12_l487_487931

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487931


namespace sum_reciprocal_factors_12_l487_487869

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487869


namespace second_divisor_of_1008_l487_487546

theorem second_divisor_of_1008 (n : ℕ) :
  (smallest_num := 1014) →
  (diminished_value == smallest_num - 6 := 1008) →
  (diminished_value ∣ 12) ∧
  (diminished_value ∣ 18) ∧
  (diminished_value ∣ 21) ∧
  (diminished_value ∣ 28) →
  (∀ k : ℕ, 1014 - 6 = 1008 ∧
  k ∣ 1008 ∧
  k ≠ 12 ∧
  k ≠ 18 ∧
  k ≠ 21 ∧
  k ≠ 28 ∧
  k = 14 → True) :=
sorry

end second_divisor_of_1008_l487_487546


namespace isosceles_triangle_exists_l487_487265

noncomputable def triangle_construction (r ρ : ℝ) : Prop :=
  r > 2 * ρ ∧ ∃ (d : ℝ) (K O : ℝ × ℝ), d = Real.sqrt (r * (r - 2 * ρ)) ∧ dist K O = d

theorem isosceles_triangle_exists (r ρ : ℝ) (h : r > 2 * ρ) :
  ∃ (T : Triangle ℝ), T.circumradius = r ∧ T.inradius = ρ ∧ T.is_isosceles := 
  by
    sorry

end isosceles_triangle_exists_l487_487265


namespace sum_of_reciprocals_of_factors_of_12_l487_487898

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487898


namespace sum_reciprocals_12_l487_487922

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487922


namespace sum_of_reciprocals_of_factors_of_12_l487_487899

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487899


namespace total_students_l487_487148

-- Lean statement: Prove the number of students given the conditions.
theorem total_students (num_classrooms : ℕ) (num_buses : ℕ) (seats_per_bus : ℕ) 
  (students : ℕ) (h1 : num_classrooms = 87) (h2 : num_buses = 29) 
  (h3 : seats_per_bus = 2) (h4 : students = num_classrooms * num_buses * seats_per_bus) :
  students = 5046 :=
by
  sorry

end total_students_l487_487148


namespace bridge_length_l487_487535

theorem bridge_length (train_length : ℕ) (train_speed_kmph : ℕ) (crossing_time_sec : ℕ) 
  (h_train_length : train_length = 150) 
  (h_train_speed_kmph : train_speed_kmph = 45) 
  (h_crossing_time_sec : crossing_time_sec = 30) : 
  let speed_mps := (train_speed_kmph * 1000) / 3600 in
  let total_distance := speed_mps * crossing_time_sec in
  let bridge_length := total_distance - train_length in
  bridge_length = 225 := 
by 
  sorry

end bridge_length_l487_487535


namespace part_I_range_part_II_maximum_M_l487_487366

noncomputable theory

-- Definition of function f(x) with constants a and b
def f (a b : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + b

-- Part I: Prove the range of f(x) on the interval [-1, 2] when a = b = 1
theorem part_I_range :
  ∀ x ∈ Icc (-1) 2, 2 ≤ f 1 1 x ∧ f 1 1 x ≤ Real.exp 2 - 1 := 
sorry

-- Part II: Prove the maximum value of M(a, b) = a - b is e, given f(x) ≥ 0 for any x ∈ ℝ
theorem part_II_maximum_M :
  ∀ a b : ℝ, (∀ x : ℝ, 0 ≤ f a b x) → (a - b ≤ Real.exp 1 ∧ (∃ x : ℝ, M a b = Real.exp 1)) :=
sorry

end part_I_range_part_II_maximum_M_l487_487366


namespace find_number_l487_487178

theorem find_number (x : ℝ) (h : x / 2 = x - 5) : x = 10 :=
by
  sorry

end find_number_l487_487178


namespace correct_calculation_l487_487188

theorem correct_calculation :
  (∀ x, (x = (\sqrt 3)^2 → x ≠ 9)) ∧
  (∀ y, (y = \sqrt ((-2)^2) → y ≠ -2)) ∧
  (∀ z, (z = \sqrt 3 * \sqrt 2 → z ≠ 6)) ∧
  (∀ w, (w = \sqrt 8 / \sqrt 2 → w = 2)) :=
by
  sorry

end correct_calculation_l487_487188


namespace sum_reciprocals_factors_12_l487_487739

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487739


namespace sum_reciprocals_factors_12_l487_487701

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487701


namespace length_of_segment_constant_l487_487252

noncomputable def circle_midpoint_arc (O : Type) [metric_space O] {A B : O} (circ : circle O) 
  (C : O) (midpoint : C = midpoint (arc_with AB on circ)) : Prop :=
∃ (D : O), D ∈ (minor_arc AB on circ) ∧ 
∃ (E F G H : O), 
  tangent_at D circ ∧ 
  (tangent intersects tangent_at A circ and tangent_at B circ at E F) ∧
  line CE intersects AB at G ∧ 
  line CF intersects AB at H ∧ 
  segment GH length = (1/2) * segment AB length

theorem length_of_segment_constant (O : Type) [metric_space O] (circ : circle O) 
  (A B C : O) (midpoint : C = midpoint (arc_with AB on circ)) :
  circle_midpoint_arc O A B C midpoint := by
    -- Proof will be here
    sorry

end length_of_segment_constant_l487_487252


namespace sum_reciprocals_of_factors_of_12_l487_487968

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487968


namespace sum_of_reciprocals_of_factors_of_12_l487_487812

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487812


namespace side_length_x_approx_3_58_l487_487542

theorem side_length_x_approx_3_58 :
  let x := (8 * Real.pi / 7) in
  Real.abs (x - 3.58) < 0.01 :=
by
  sorry

end side_length_x_approx_3_58_l487_487542


namespace sum_of_reciprocals_factors_12_l487_487788

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487788


namespace projection_matrix_correct_l487_487063

-- Define the normal vector and the projection matrix.
def normal_vector : ℝ × ℝ × ℝ := (2, -1, 2)
def Q_matrix : ℝ → ℝ → ℝ → ℝ × ℝ × ℝ := 
  λ x y z, 
    ( (5 * x + 2 * y - 4 * z) / 9, 
      (2 * x + 10 * y - 2 * z) / 9, 
      (-4 * x + 2 * y + 5 * z) / 9 )

-- Theorem to prove the matrix Q correctly projects any vector onto the plane.
theorem projection_matrix_correct (v : ℝ × ℝ × ℝ) : 
    let ⟨x, y, z⟩ := v in 
    Q_matrix x y z = ((v.1) * (5/9) + (v.2) * (2/9) - (v.3) * (4/9), 
                      (v.1) * (2/9) + (v.2) * (10/9) - (v.3) * (2/9), 
                      (v.1) * (-4/9) + (v.2) * (2/9) + (v.3) * (5/9)) := 
by sorry

end projection_matrix_correct_l487_487063


namespace projection_onto_plane_l487_487062

open Matrix

noncomputable def normal_vector : Vector ℝ 3 := ![2, -1, 2]

noncomputable def projection_matrix : Matrix (Fin 3) (Fin 3) ℝ := 
![![5 / 9, 2 / 9, -4 / 9], 
  ![2 / 9, 8 / 9, 2 / 9], 
  ![-4 / 9, 2 / 9, 5 / 9]]

theorem projection_onto_plane (v : Vector ℝ 3) :
  let Q := proj ℝ (span ℝ (Set.range ![normal_vector])) in 
  projection_matrix.mul_vec v = Q v :=
sorry

end projection_onto_plane_l487_487062


namespace sector_arc_length_120_degrees_radius_3_l487_487013

noncomputable def arc_length (θ : ℝ) (r : ℝ) : ℝ :=
  (θ / 360) * 2 * Real.pi * r

theorem sector_arc_length_120_degrees_radius_3 :
  arc_length 120 3 = 2 * Real.pi :=
by
  sorry

end sector_arc_length_120_degrees_radius_3_l487_487013


namespace sum_reciprocals_factors_12_l487_487936

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487936


namespace greatest_distance_centers_of_circles_in_rectangle_l487_487561

/--
Two circles are drawn in a 20-inch by 16-inch rectangle,
each circle with a diameter of 8 inches.
Prove that the greatest possible distance between 
the centers of the two circles without extending beyond the 
rectangular region is 4 * sqrt 13 inches.
-/
theorem greatest_distance_centers_of_circles_in_rectangle :
  let diameter := 8
  let width := 20
  let height := 16
  let radius := diameter / 2
  let reduced_width := width - 2 * radius
  let reduced_height := height - 2 * radius
  let distance := Real.sqrt ((reduced_width ^ 2) + (reduced_height ^ 2))
  distance = 4 * Real.sqrt 13 := by
    sorry

end greatest_distance_centers_of_circles_in_rectangle_l487_487561


namespace partA_l487_487240

theorem partA (x y z : ℂ) 
  (hx : abs x = 1) 
  (hy : abs y = 1) 
  (harg : (Real.pi / 3) ≤ complex.arg x - complex.arg y ∧ complex.arg x - complex.arg y ≤ 5 * Real.pi / 3) : 
  abs z + abs (z - x) + abs (z - y) ≥ abs (z * x - y) :=
sorry

end partA_l487_487240


namespace tangent_line_equation_min_t_value_ln_n_inequality_l487_487364

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (x + 1)

theorem tangent_line_equation :
  let f'(x : ℝ) := (λ x, (x + 1 - x * Real.log x) / (x * (x + 1) ^ 2))
  ∃ x y : ℝ, x = 1 ∧ y = 0 ∧ f'(1) = 1 / 2 ∧ (x - 2 * y - 1 = 0) :=
sorry

theorem min_t_value :
  ∃ t : ℝ, (∀ x : ℝ, 0 < x → f(x) + t / x ≥ 2 / (x + 1)) ∧ (t = 1) :=
sorry

theorem ln_n_inequality (n : ℕ) (hn : 2 ≤ n) :
  Real.log n > ∑ i in Finset.range n \ {0, 1}, (1 / (i + 2 : ℝ)) :=
sorry

end tangent_line_equation_min_t_value_ln_n_inequality_l487_487364


namespace sum_of_reciprocals_factors_12_l487_487817

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487817


namespace sum_of_reciprocals_factors_12_l487_487827

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487827


namespace zero_points_of_g_l487_487314

noncomputable def f (a : ℝ) : ℝ → ℝ := λ x,
  if x >= a then x^2 - 2
  else x + 2

noncomputable def h : ℝ → ℝ := λ x, Real.log x + 1/x

noncomputable def g (a : ℝ) : ℝ → ℝ := λ x, f a (h x) - a

theorem zero_points_of_g (a : ℝ) : 
  (∃ x : ℝ, g a x = 0) ↔ a ∈ Set.Icc (-1 : ℝ) 2 ∪ Set.Ici 3 :=
by sorry

end zero_points_of_g_l487_487314


namespace sum_of_reciprocals_factors_12_l487_487721

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487721


namespace length_of_second_train_l487_487205

/-- 
The length of the second train can be determined given the length and speed of the first train,
the speed of the second train, and the time they take to cross each other.
-/
theorem length_of_second_train (speed1_kmph : ℝ) (length1_m : ℝ) (speed2_kmph : ℝ) (time_s : ℝ) :
  (speed1_kmph = 120) →
  (length1_m = 230) →
  (speed2_kmph = 80) →
  (time_s = 9) →
  let relative_speed_m_per_s := (speed1_kmph * 1000 / 3600) + (speed2_kmph * 1000 / 3600)
  let total_distance := relative_speed_m_per_s * time_s
  let length2_m := total_distance - length1_m
  length2_m = 269.95 :=
by
  intros h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃, h₄]
  let relative_speed_m_per_s := (120 * 1000 / 3600) + (80 * 1000 / 3600)
  let total_distance := relative_speed_m_per_s * 9
  let length2_m := total_distance - 230
  exact sorry

end length_of_second_train_l487_487205


namespace smallest_positive_multiple_l487_487592

theorem smallest_positive_multiple :
  ∃ (a : ℕ), (47 * a ≡ 7 [MOD 97]) ∧ (47 * a ≡ -3 [MOD 31]) ∧ (47 * a = 79618) :=
by
  sorry

end smallest_positive_multiple_l487_487592


namespace calc_value_l487_487344

noncomputable def f : ℝ → ℝ := sorry 

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom non_const_zero : ∃ x : ℝ, f x ≠ 0
axiom functional_eq : ∀ x : ℝ, x * f (x + 1) = (x + 1) * f x

theorem calc_value : f (f (5 / 2)) = 0 :=
sorry

end calc_value_l487_487344


namespace expand_poly_product_l487_487281

noncomputable def poly1 : Polynomial ℤ := 7 * X ^ 2 + 5 * X + 3
noncomputable def poly2 : Polynomial ℤ := 3 * X ^ 3 + 2 * X ^ 2 + 1
noncomputable def expected : Polynomial ℤ := 21 * X ^ 5 + 29 * X ^ 4 + 19 * X ^ 3 + 13 * X ^ 2 + 5 * X + 3

theorem expand_poly_product : poly1 * poly2 = expected := by
  sorry

end expand_poly_product_l487_487281


namespace minimize_wage_l487_487499

def totalWorkers : ℕ := 150
def wageA : ℕ := 2000
def wageB : ℕ := 3000

theorem minimize_wage : ∃ (a : ℕ), a = 50 ∧ (totalWorkers - a) ≥ 2 * a ∧ 
  (wageA * a + wageB * (totalWorkers - a) = 400000) := sorry

end minimize_wage_l487_487499


namespace shaded_area_correct_l487_487528

-- Definitions based on the conditions
def radius : ℝ := 60
noncomputable def pi_approx : ℝ := 3.14

-- The proof statement
theorem shaded_area_correct :
  let circle_area := pi_approx * radius^2 in
  let total_structural_area : ℝ := 14400 in
  total_structural_area - circle_area = 3096 :=
by
  -- Definitions from total structural area (2 squares and 1 octagon)
  let total_structural_area := 14400
  -- Calculate the circle area using given π approximation
  let circle_area := pi_approx * radius^2

  -- Goal is to assert the shaded area calculation
  have H : total_structural_area - circle_area = 3096, from
    calc
      total_structural_area - circle_area
        = 14400 - (3.14 * 60^2) : by rw [radius, pi_approx]
    ... = 14400 - 11304           : by norm_num
    ... = 3096                  : by norm_num,

  exact H

end shaded_area_correct_l487_487528


namespace modulus_conjugate_of_2_plus_3i_l487_487405

theorem modulus_conjugate_of_2_plus_3i : 
  ∀ z : Complex, z = Complex.conj (2 + 3 * Complex.I) → Complex.abs z = Real.sqrt 13 :=
by
  intro z h
  sorry

end modulus_conjugate_of_2_plus_3i_l487_487405


namespace initial_balance_before_check_deposit_l487_487197

theorem initial_balance_before_check_deposit (new_balance : ℝ) (initial_balance : ℝ) : 
  (50 = 1 / 4 * new_balance) → (initial_balance = new_balance - 50) → initial_balance = 150 :=
by
  sorry

end initial_balance_before_check_deposit_l487_487197


namespace solve_l487_487153

noncomputable def num_students_in_A : ℝ :=
  let x := 
    50 * (70 * 70) / 
    (61.67 * 70 - 50 * 70) in
  x

theorem solve : num_students_in_A = 50 := 
  by sorry

end solve_l487_487153


namespace arrangement_count_l487_487158

-- Define the problem conditions
def students : ℕ := 3
def villages : ℕ := 2
def total_arrangements : ℕ := 6

-- Define the property we want to prove
theorem arrangement_count :
  ∃ (s v : ℕ), s = students ∧ v = villages ∧ ∑ k in finset.range ((nat.choose students 2) * (nat.choose (students-2) 1) * nat.factorial villages), k = total_arrangements :=
by
  sorry

end arrangement_count_l487_487158


namespace sum_reciprocals_factors_12_l487_487704

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487704


namespace percent_of_x_eq_21_percent_l487_487977

theorem percent_of_x_eq_21_percent (x : Real) : (0.21 * x = 0.30 * 0.70 * x) := by
  sorry

end percent_of_x_eq_21_percent_l487_487977


namespace sum_reciprocals_factors_12_l487_487938

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487938


namespace sum_of_arcs_approaches_l487_487212

theorem sum_of_arcs_approaches (D : ℝ) (n : ℕ) :
  (∑ i in Finset.range n, (π * (D / n) / 2) + (π * (D / (2 * n)) / 2)) → (3 * π * D / 4) as n → ∞ :=
by
  sorry

end sum_of_arcs_approaches_l487_487212


namespace centroid_ratio_inequality_l487_487415

-- Definitions and conditions from the problem statement
variable {A B C G D E F A' B' C' : Point}
variable {circumcircle : Circle ABC}
variable {triangle : Triangle ABC}

-- Our required theorem statement
theorem centroid_ratio_inequality
  (h1 : G = centroid A B C)
  (h2 : incidence AG D BC ∧ incidence BG E CA ∧ incidence CG F AB)
  (h3 : incidence (extendLineThrough AG) A' circumcircle ∧ incidence (extendLineThrough BG) B' circumcircle ∧ incidence (extendLineThrough CG) C' circumcircle) 
  : (A'.d / D.A) + (B'.e / E.B) + (C'.f / F.C) ≥ 1 :=
sorry

end centroid_ratio_inequality_l487_487415


namespace sum_of_reciprocals_factors_12_l487_487832

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487832


namespace bus_routes_arrangement_l487_487434

-- Define the lines and intersection points (stops).
def routes := Fin 10
def stops (r1 r2 : routes) : Prop := r1 ≠ r2 -- Representing intersection

-- First condition: Any subset of 9 routes will cover all stops.
def covers_all_stops (routes_subset : Finset routes) : Prop :=
  routes_subset.card = 9 → ∀ r1 r2 : routes, r1 ≠ r2 → stops r1 r2

-- Second condition: Any subset of 8 routes will miss at least one stop.
def misses_at_least_one_stop (routes_subset : Finset routes) : Prop :=
  routes_subset.card = 8 → ∃ r1 r2 : routes, r1 ≠ r2 ∧ ¬stops r1 r2

-- The theorem to prove that this arrangement is possible.
theorem bus_routes_arrangement : 
  (∃ stops_scheme : routes → routes → Prop, 
    (∀ subset_9 : Finset routes, covers_all_stops subset_9) ∧ 
    (∀ subset_8 : Finset routes, misses_at_least_one_stop subset_8)) :=
by
  sorry

end bus_routes_arrangement_l487_487434


namespace number_of_ways_to_place_numbers_l487_487041

theorem number_of_ways_to_place_numbers (a b c : ℕ) : 
    (4 * 14 * a = 14 * 6 * c) ∧ 
    (4 * 14 * a = a * b * c) →
    b = 28 / c →
    6 =
    if (c = 1 ∨ c = 2 ∨ c = 4 ∨ c = 7 ∨ c = 14 ∨ c = 28) 
    then 6 
    else 0 :=
begin
    sorry
end

end number_of_ways_to_place_numbers_l487_487041


namespace roberto_valid_outfits_l487_487500

-- Definitions based on the conditions
def total_trousers : ℕ := 6
def total_shirts : ℕ := 8
def total_jackets : ℕ := 4
def restricted_jacket : ℕ := 1
def restricted_shirts : ℕ := 2

-- Theorem statement
theorem roberto_valid_outfits : 
  total_trousers * total_shirts * total_jackets - total_trousers * restricted_shirts * restricted_jacket = 180 := 
by
  sorry

end roberto_valid_outfits_l487_487500


namespace mass_percentage_Al_in_Al2CO33_l487_487292
-- Importing the required libraries

-- Define the necessary constants for molar masses
def molar_mass_Al : ℝ := 26.98
def molar_mass_C : ℝ := 12.01
def molar_mass_O : ℝ := 16.00
def molar_mass_Al2CO33 : ℝ := 2 * molar_mass_Al + 3 * molar_mass_C + 9 * molar_mass_O
def mass_Al_in_Al2CO33 : ℝ := 2 * molar_mass_Al

-- Define the main theorem to prove the mass percentage of Al in Al2(CO3)3
theorem mass_percentage_Al_in_Al2CO33 :
  (mass_Al_in_Al2CO33 / molar_mass_Al2CO33) * 100 = 23.05 :=
by
  simp [molar_mass_Al, molar_mass_C, molar_mass_O, molar_mass_Al2CO33, mass_Al_in_Al2CO33]
  -- Calculation result based on given molar masses
  sorry

end mass_percentage_Al_in_Al2CO33_l487_487292


namespace sum_of_reciprocals_of_factors_of_12_l487_487771

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487771


namespace sum_of_reciprocals_of_factors_of_12_l487_487761

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487761


namespace solve_for_m_l487_487297

-- Define the operation ◎ for real numbers a and b
def op (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

-- Lean statement for the proof problem
theorem solve_for_m (m : ℝ) (h : op (m + 1) (m - 2) = 16) : m = 3 ∨ m = -2 :=
sorry

end solve_for_m_l487_487297


namespace possible_values_of_m_l487_487329

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = m ∧ 
  (∀ n, a (n + 1) = if a n % 2 = 0 then a n / 2 else 3 * a n + 1)

theorem possible_values_of_m (m : ℕ) (a : ℕ → ℕ) (h : sequence a) (h_a6 : a 6 = 1) :
  m = 4 ∨ m = 5 ∨ m = 32 :=
sorry

end possible_values_of_m_l487_487329


namespace sum_of_reciprocals_of_factors_of_12_l487_487668

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487668


namespace sum_reciprocals_factors_12_l487_487885

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487885


namespace sum_reciprocal_factors_12_l487_487856

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487856


namespace P_sufficient_but_not_necessary_for_Q_l487_487308

-- Definitions based on given conditions
def P (x : ℝ) : Prop := abs (2 * x - 3) < 1
def Q (x : ℝ) : Prop := x * (x - 3) < 0

-- The theorem to prove that P is sufficient but not necessary for Q
theorem P_sufficient_but_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ (∃ x : ℝ, Q x ∧ ¬P x) :=
by
  sorry

end P_sufficient_but_not_necessary_for_Q_l487_487308


namespace find_slope_of_chord_l487_487037

def ellipse_eq (x y : ℝ) : Prop :=
  (x^2) / 16 + (y^2) / 9 = 1

def is_midpoint (M : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  2 * M.1 = A.1 + B.1 ∧ 2 * M.2 = A.2 + B.2

theorem find_slope_of_chord (x1 y1 x2 y2 : ℝ) (hA : ellipse_eq x1 y1) (hB : ellipse_eq x2 y2)
  (hM : is_midpoint (-2, 1) (x1, y1) (x2, y2)) :
  let k := -(9 * (x1 + x2)) / (16 * (y1 + y2)) in
  k = 9 / 8 :=
by
  dsimp [is_midpoint] at hM
  have h_sum_x : x1 + x2 = -4 := hM.1
  have h_sum_y : y1 + y2 = 2 := hM.2
  let k := -(9 * (x1 + x2)) / (16 * (y1 + y2))
  rw [h_sum_x, h_sum_y]
  norm_num
  exact sorry

end find_slope_of_chord_l487_487037


namespace minimize_distance_l487_487349

-- Definition of the condition
def point_lies_on_line (a b : ℝ) : Prop :=
  3 * a - 4 * b = 10

-- The actual theorem statement
theorem minimize_distance (a b : ℝ) (h : point_lies_on_line a b) : 
  ∃ p : ℝ, p = 2 ∧ sqrt (a^2 + b^2) = p :=
begin
  sorry
end

end minimize_distance_l487_487349


namespace sum_of_reciprocals_factors_12_l487_487729

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487729


namespace eagles_win_at_least_three_matches_l487_487511

-- Define the conditions
def n : ℕ := 5
def p : ℝ := 0.5

-- Binomial coefficient function
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Probability function for the binomial distribution
noncomputable def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial n k) * p^k * (1 - p)^(n - k)

-- Theorem stating the main result
theorem eagles_win_at_least_three_matches :
  (binomial_prob n 3 p + binomial_prob n 4 p + binomial_prob n 5 p) = 1 / 2 :=
by
  sorry

end eagles_win_at_least_three_matches_l487_487511


namespace value_of_a_l487_487473

noncomputable def f (a x : ℝ) := Real.log x - a * x
noncomputable def g (a x : ℝ) := Real.exp x - 3 * a * x

theorem value_of_a (a : ℝ) :
  (∀ x > 1, deriv (λ x, f a x) x ≤ 0) ∧ (∃ x > 1, deriv (λ x, g a x) x = 0) ↔ 1 ≤ a := 
sorry

end value_of_a_l487_487473


namespace dropped_participants_did_not_play_each_other_l487_487275

theorem dropped_participants_did_not_play_each_other (n : ℕ) (total_games : ℕ) (dropped_games : ℕ) 
  (played_eq_number : ℕ → ℕ) (round_robin : (n Choose 2) = total_games) 
  (eq_number_condition : ∀ (i j : ℕ), (i ≠ j) → played_eq_number i = played_eq_number j) 
  (total_games_eq : total_games = 23) : 
  ¬ (playd_eq_number 1 = played_eq_number 2) :=
sorry

end dropped_participants_did_not_play_each_other_l487_487275


namespace sum_of_reciprocals_of_factors_of_12_l487_487663

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487663


namespace sum_of_reciprocals_factors_12_l487_487731

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487731


namespace length_of_other_train_l487_487564

variable (speed_train1_kmph : ℝ) 
variable (speed_train2_kmph : ℝ)
variable (length_train1_m : ℝ)
variable (crossing_time_s : ℝ)

noncomputable def speed_train1_mps := speed_train1_kmph * 1000 / 3600
noncomputable def speed_train2_mps := speed_train2_kmph * 1000 / 3600
noncomputable def relative_speed_mps := speed_train1_mps + speed_train2_mps
noncomputable def total_distance_covered := relative_speed_mps * crossing_time_s
noncomputable def length_train2_m := total_distance_covered - length_train1_m

theorem length_of_other_train :
  length_train2_m 60 40 190 11.879049676025918 = 140 :=
by sorry

end length_of_other_train_l487_487564


namespace count_valid_permutations_l487_487031

-- Definitions based on the conditions
def is_increasing (a b c : ℕ) : Prop := a < b ∧ b < c
def is_decreasing (a b c : ℕ) : Prop := a > b ∧ b > c

-- Definition of valid sequence based on conditions
def is_valid_sequence (l : List ℕ) : Prop :=
  ∀ (a b c : ℕ) (h1 : a ∈ l) (h2 : b ∈ l) (h3 : c ∈ l), 
  is_increasing a b c ∨ is_decreasing a b c → 
  ¬((l.indexOf a < l.indexOf b) 
  ∧ (l.indexOf b < l.indexOf c) 
  ∧ l.indexOf c - l.indexOf a = 2)

-- Main theorem statement based on the proof problem
theorem count_valid_permutations : 
  Finset.card { perm : List ℕ | perm ~ [1, 2, 3, 4, 5, 6] ∧ is_valid_sequence perm } = 16 :=
by sorry

end count_valid_permutations_l487_487031


namespace ratio_of_surface_areas_l487_487408

theorem ratio_of_surface_areas (r1 r2 : ℝ) (h : r1 / r2 = 1 / 2) :
  (4 * Real.pi * r1^2) / (4 * Real.pi * r2^2) = 1 / 4 :=
by
  sorry

end ratio_of_surface_areas_l487_487408


namespace sum_of_reciprocals_of_factors_of_12_l487_487910

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487910


namespace find_f_of_functions_l487_487000

theorem find_f_of_functions
  (f g : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = - f x)
  (h_even : ∀ x, g (-x) = g x)
  (h_eq : ∀ x, f x + g x = x^3 - x^2 + x - 3) :
  ∀ x, f x = x^3 + x := 
sorry

end find_f_of_functions_l487_487000


namespace central_angle_of_sector_l487_487010

theorem central_angle_of_sector 
  (A : ℝ)
  (r : ℝ)
  (a : ℝ)
  (h1 : A = (3 * real.pi) / 8)
  (h2 : r = 1) 
  (h3 : A = (1 / 2) * a * r^2) :
  a = (3 * real.pi) / 4 :=
sorry

end central_angle_of_sector_l487_487010


namespace sum_of_possible_k_values_l487_487079

theorem sum_of_possible_k_values (a b c : ℝ) (n : ℕ) (h : n > 2) :
  let f := λ x : ℝ, x^n + a * x^2 + b * x + c in
  let possible_k := if n % 2 = 0 then {0, 1, 2, 3, 4} else {0, 1, 2, 3} in
  (∑ k in possible_k, k) = 10 :=
sorry

end sum_of_possible_k_values_l487_487079


namespace sum_reciprocals_of_factors_12_l487_487839

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487839


namespace sum_of_reciprocals_factors_12_l487_487820

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487820


namespace sum_of_reciprocals_factors_12_l487_487792

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487792


namespace greatest_3_digit_base7_divisible_by_7_l487_487168

def base7ToDec (a b c : ℕ) : ℕ := a * 7^2 + b * 7^1 + c * 7^0

theorem greatest_3_digit_base7_divisible_by_7 :
  ∃ (a b c : ℕ), a ≠ 0 ∧ a < 7 ∧ b < 7 ∧ c < 7 ∧
  base7ToDec a b c % 7 = 0 ∧ base7ToDec a b c = 342 :=
begin
  use [6, 6, 6],
  split, { repeat { norm_num } }, -- a ≠ 0
  split, { norm_num }, -- a < 7
  split, { norm_num }, -- b < 7
  split, { norm_num }, -- c < 7
  split,
  { norm_num },
  norm_num,
end

end greatest_3_digit_base7_divisible_by_7_l487_487168


namespace find_num_tables_l487_487096

-- Definitions based on conditions
def num_students_in_class : ℕ := 47
def num_girls_bathroom : ℕ := 3
def num_students_canteen : ℕ := 3 * 3
def num_students_new_groups : ℕ := 2 * 4
def num_students_exchange : ℕ := 3 * 3 + 3 * 3 + 3 * 3

-- Calculation of the number of tables (corresponding to the answer)
def num_missing_students : ℕ := num_girls_bathroom + num_students_canteen + num_students_new_groups + num_students_exchange

def num_students_currently_in_class : ℕ := num_students_in_class - num_missing_students
def students_per_table : ℕ := 3

def num_tables : ℕ := num_students_currently_in_class / students_per_table

-- The theorem we want to prove
theorem find_num_tables : num_tables = 6 := by
  -- Proof steps would go here
  sorry

end find_num_tables_l487_487096


namespace product_of_odd_last_digit_is_5_l487_487171

theorem product_of_odd_last_digit_is_5 (n : ℕ) (hn : n ≥ 99 ∨ n ≥ 199) :
    let odd_product := ∏ i in (finset.range (n + 1)).filter (λ k, k % 2 = 1), i
    in (odd_product % 10) = 5 :=
by
  sorry

end product_of_odd_last_digit_is_5_l487_487171


namespace sum_reciprocals_factors_12_l487_487881

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487881


namespace max_intersections_two_circles_three_lines_l487_487570

theorem max_intersections_two_circles_three_lines :
  ∀ (C1 C2 : ℝ × ℝ × ℝ) (L1 L2 L3 : ℝ × ℝ × ℝ), 
  C1 ≠ C2 → L1 ≠ L2 → L2 ≠ L3 → L1 ≠ L3 →
  ∃ (P : ℕ), P = 17 :=
by 
  sorry

end max_intersections_two_circles_three_lines_l487_487570


namespace angle_condition_AC_implies_equivalence_l487_487099

theorem angle_condition_AC_implies_equivalence (A B C D E : Point) (h1: collinear A B C D) 
(h2: dist A B = dist C D) (h3: dist C E = dist D E) : 
(∠ C E D = 2 * ∠ A E B) ↔ (dist A C = dist E C) :=
sorry

end angle_condition_AC_implies_equivalence_l487_487099


namespace ravi_made_overall_profit_l487_487994

-- Variables for costs
def cost_refrigerator : ℝ := 15000
def cost_mobile : ℝ := 8000

-- Variables for loss and profit percentages
def loss_percentage_refrigerator : ℝ := 2 / 100
def profit_percentage_mobile : ℝ := 10 / 100

-- Definitions
def loss_refrigerator := loss_percentage_refrigerator * cost_refrigerator
def selling_price_refrigerator := cost_refrigerator - loss_refrigerator
def profit_mobile := profit_percentage_mobile * cost_mobile
def selling_price_mobile := cost_mobile + profit_mobile

-- Overall calculations
def total_cost_price := cost_refrigerator + cost_mobile
def total_selling_price := selling_price_refrigerator + selling_price_mobile
def overall_profit := total_selling_price - total_cost_price

theorem ravi_made_overall_profit : overall_profit = 500 := by sorry

end ravi_made_overall_profit_l487_487994


namespace suitable_for_census_l487_487985

-- Definitions based on the conditions in a)
def survey_A := "The service life of a batch of batteries"
def survey_B := "The height of all classmates in the class"
def survey_C := "The content of preservatives in a batch of food"
def survey_D := "The favorite mathematician of elementary and middle school students in the city"

-- The main statement to prove
theorem suitable_for_census : survey_B = "The height of all classmates in the class" := by
  -- We assert that the height of all classmates is the suitable survey for a census based on given conditions
  sorry

end suitable_for_census_l487_487985


namespace sum_of_reciprocals_factors_12_l487_487789

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487789


namespace at_least_binom_sum_nonneg_l487_487452

theorem at_least_binom_sum_nonneg {n d : ℕ} {x : fin n → ℝ}
  (h₁ : n ≥ 2) 
  (h₂ : d ≥ 1) 
  (h₃ : d ∣ n) 
  (h_sum : ∑ i, x i = 0) : 
  ∃ (I : finset (fin n)), I.card = d ∧ 0 ≤ ∑ i in I, x i ∧ finset.card (finset.choose (n - 1) (d - 1)) :=
sorry

end at_least_binom_sum_nonneg_l487_487452


namespace sum_of_reciprocals_factors_12_l487_487784

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487784


namespace sum_of_reciprocals_of_factors_of_12_l487_487813

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487813


namespace sum_reciprocals_factors_12_l487_487889

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487889


namespace time_to_cross_bridge_l487_487237

-- Definitions for conditions
def length_train := 100 -- in meters
def length_bridge := 300 -- in meters
def speed_train_kmh := 60 -- in km/h
def speed_train_ms := speed_train_kmh * 1000 / 3600 -- converting to m/s

-- Theorem statement
theorem time_to_cross_bridge : 
  let total_distance := length_train + length_bridge in
  let time := total_distance / speed_train_ms in
  time = 24 := sorry

end time_to_cross_bridge_l487_487237


namespace piesEatenWithForksPercentage_l487_487210

def totalPies : ℕ := 2000
def notEatenWithForks : ℕ := 640
def eatenWithForks : ℕ := totalPies - notEatenWithForks

def percentageEatenWithForks := (eatenWithForks : ℚ) / totalPies * 100

theorem piesEatenWithForksPercentage : percentageEatenWithForks = 68 := by
  sorry

end piesEatenWithForksPercentage_l487_487210


namespace sum_reciprocals_factors_12_l487_487886

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487886


namespace constant_term_expansion_l487_487427

theorem constant_term_expansion : 
  (∃ r : ℕ, (Binomial (8:ℕ) r) * ((1 / 2)^(8 - r)) * ((-1)^r) * ((2^(r - 8)) * 1) = 28) :=
sorry

end constant_term_expansion_l487_487427


namespace sequence_properties_l487_487328

section
variables {a : ℕ → ℕ} {b : ℕ → ℕ}

-- Given conditions
def a_n (n : ℕ) : ℕ :=
  if n = 1 then 1
  else n * a (n + 1) = 2 * (n + 1) * a n

-- Define b_n
def b_n (n : ℕ) : ℕ := a n / n

-- Prove the specific values and general form
theorem sequence_properties :
  a 1 = 1 ∧
  (∀ n : ℕ, n * a (n + 1) = 2 * (n + 1) * a n) →
  b 1 = 1 ∧ b 2 = 2 ∧ b 3 = 4 ∧
  (∀ n : ℕ, b (n + 1) / b n = 2) ∧
  (∀ n : ℕ, a n = n * 2^(n-1)) :=
by
  sorry
end

end sequence_properties_l487_487328


namespace max_intersections_l487_487582

-- Define the number of circles and lines
def num_circles : ℕ := 2
def num_lines : ℕ := 3

-- Define the maximum number of intersection points of circles
def max_circle_intersections : ℕ := 2

-- Define the number of intersection points between each line and each circle
def max_line_circle_intersections : ℕ := 2

-- Define the number of intersection points among lines (using the combination formula)
def num_line_intersections : ℕ := (num_lines.choose 2)

-- Define the greatest number of points of intersection
def total_intersections : ℕ :=
  max_circle_intersections + (num_lines * num_circles * max_line_circle_intersections) + num_line_intersections

-- Prove the greatest number of points of intersection is 17
theorem max_intersections : total_intersections = 17 := by
  -- Calculating individual parts for clarity
  have h1: max_circle_intersections = 2 := rfl
  have h2: num_lines * num_circles * max_line_circle_intersections = 12 := by
    calc
      num_lines * num_circles * max_line_circle_intersections
        = 3 * 2 * 2 := by rw [num_lines, num_circles, max_line_circle_intersections]
        ... = 12 := by norm_num
  have h3: num_line_intersections = 3 := by
    calc
      num_line_intersections = (3.choose 2) := rfl
      ... = 3 := by norm_num

  -- Adding the parts to get the total intersections
  calc
    total_intersections
      = max_circle_intersections + (num_lines * num_circles * max_line_circle_intersections) + num_line_intersections := rfl
      ... = 2 + 12 + 3 := by rw [h1, h2, h3]
      ... = 17 := by norm_num

end max_intersections_l487_487582


namespace sum_reciprocals_factors_12_l487_487892

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487892


namespace equal_integral_pairs_count_l487_487311

def f1 (x : ℝ) := 2 * |x|
def g1 (x : ℝ) := x + 1

def f2 (x : ℝ) := Real.sin x
def g2 (x : ℝ) := Real.cos x

noncomputable def f3 (x : ℝ) := Real.sqrt (1 - x ^ 2)
def g3 (x : ℝ) := (π / 4) * x ^ 2

def f4 (x : ℝ) := x   -- example of an odd function
def g4 (x : ℝ) := -x  -- example of an odd function

def isEqualIntegral (f g : ℝ → ℝ) : Prop :=
  ∫ x in -1..1, f x = ∫ x in -1..1, g x

theorem equal_integral_pairs_count : 
  let pairs := [(f1, g1), (f2, g2), (f3, g3), (f4, g4)] 
  in (pairs.map (λ p, isEqualIntegral p.1 p.2)).count true = 3 :=
sorry

end equal_integral_pairs_count_l487_487311


namespace work_completion_days_l487_487003

-- We assume D is a certain number of days and W is some amount of work
variables (D W : ℕ)

-- Define the rate at which 3 people can do 3W work in D days
def rate_3_people : ℚ := 3 * W / D

-- Define the rate at which 5 people can do 5W work in D days
def rate_5_people : ℚ := 5 * W / D

-- The problem states that both rates must be equal
theorem work_completion_days : (3 * D) = D / 3 :=
by sorry

end work_completion_days_l487_487003


namespace sum_reciprocals_factors_12_l487_487952

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487952


namespace sum_of_reciprocals_of_factors_of_12_l487_487798

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487798


namespace find_eccentricity_l487_487335

variables {a b : ℝ} (hb : 0 < b) (ha : b < a)
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

variables {x₀ y₀ : ℝ} (hA : ellipse x₀ y₀)
def point_A := (x₀, y₀)

def tangent_line_at_A (x y : ℝ) : Prop := (x₀ * x / a^2) + (y₀ * y / b^2) = 1

variables {c : ℝ} (hc : c = Real.sqrt (a^2 - b^2))
def focal_distance := (c, 0)

variables {y₁ : ℝ} (hQ : tangent_line_at_A 0 y₁)
def point_Q := (0, y₁)

variables (angle_QFO : ℝ) (angle_QFA : ℝ)
def angle1 : angle_QFO = π / 4
def angle2 : angle_QFA = π / 6

noncomputable def eccentricity : ℝ := Real.sqrt (1 - (b^2 / a^2))

theorem find_eccentricity (h : angle1 : angle_QFO = π / 4 ∧ angle2 : angle_QFA = π / 6) : 
  eccentricity a b = (Real.sqrt 6) / 3 := 
sorry

end find_eccentricity_l487_487335


namespace number_of_wedges_per_potato_l487_487267

theorem number_of_wedges_per_potato
  (total_potatoes : ℕ)
  (potatoes_cut_into_wedges : ℕ)
  (chips_per_potato : ℕ)
  (extra_chips : ℕ)
  (remaining_potatoes_halved : total_potatoes - potatoes_cut_into_wedges = 54)
  (chips_more_than_wedges : extra_chips = 436) :
  (W : ℕ) := 
  13 * W = (54 / 2) * chips_per_potato - extra_chips
  → W = 8 :=
by
  sorry

end number_of_wedges_per_potato_l487_487267


namespace sum_of_reciprocals_factors_12_l487_487830

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487830


namespace partI_partII_partIII_l487_487083

-- Define the basic sets and their operations
variable {α : Type*}

-- Defining set A and conditions
noncomputable def A : Set ℝ := {-1, 1}
noncomputable def Aplus (A : Set ℝ) : Set ℝ := {x | ∃ (a b : ℝ), a ∈ A ∧ b ∈ A ∧ x = a + b}
noncomputable def Aminus (A : Set ℝ) : Set ℝ := {x | ∃ (a b : ℝ), a ∈ A ∧ b ∈ A ∧ x = abs (a - b)}

-- Part (I)
theorem partI :
  A = {-1, 1} →
  Aplus A = {-2, 0, 2} ∧ 
  Aminus A = {0, 2} := 
by 
  intro h,
  sorry

-- Defining A and ordering for Part (II)
variable {x1 x2 x3 x4 : ℝ}
variable h_order : x1 < x2 ∧ x2 < x3 ∧ x3 < x4

-- Part (II)
theorem partII (A : Set ℝ) :
  A = {x1, x2, x3, x4} ∧
  Aminus A = A →
  x1 + x4 = x2 + x3 := 
by 
  intro h,
  sorry

-- Defining the universal set and conditions for Part (III)
noncomputable def U : Set ℕ := {x | 0 ≤ x ∧ x ≤ 2023}
noncomputable def Amax (A : Set ℕ) : ℕ := A.card

-- Part (III)
theorem partIII (A : Set ℕ) :
  A ⊆ U ∧ Aplus A ∩ Aminus A = ∅ →
  Amax A ≤ 1349 := 
by 
  intro h,
  sorry

end partI_partII_partIII_l487_487083


namespace word_problem_points_l487_487236

theorem word_problem_points
  (num_problems : ℕ) (points_per_comp_problem : ℕ) (total_points : ℕ) (num_comp_problems : ℕ)
  (h1 : num_problems = 30) (h2 : points_per_comp_problem = 3) (h3 : total_points = 110) (h4 : num_comp_problems = 20) :
  let num_word_problems := num_problems - num_comp_problems,
      total_comp_points := num_comp_problems * points_per_comp_problem,
      total_word_points := total_points - total_comp_points,
      points_per_word_problem := total_word_points / num_word_problems
  in points_per_word_problem = 5 :=
by 
  sorry

end word_problem_points_l487_487236


namespace greatest_base7_3_digit_divisible_by_7_l487_487167

theorem greatest_base7_3_digit_divisible_by_7 :
  ∃ n : ℕ, n < 7^3 ∧ n ≥ 7^2 ∧ 7 ∣ n ∧ nat.to_digits 7 n = [6, 6, 0] := 
sorry

end greatest_base7_3_digit_divisible_by_7_l487_487167


namespace sum_reciprocals_of_factors_12_l487_487845

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487845


namespace sum_of_reciprocals_of_factors_of_12_l487_487900

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487900


namespace sum_reciprocal_factors_12_l487_487866

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487866


namespace congruence_solution_count_l487_487348

theorem congruence_solution_count :
  ∀ y : ℕ, y < 150 → (y ≡ 20 + 110 [MOD 46]) → y = 38 ∨ y = 84 ∨ y = 130 :=
by
  intro y
  intro hy
  intro hcong
  sorry

end congruence_solution_count_l487_487348


namespace trigonometric_identity_equiv_l487_487989

theorem trigonometric_identity_equiv (α β : ℝ) (h : α = 2 * β) :
  3.413 * (sin α)^3 * cos (3 * α) + (cos α)^3 * sin (3 * α) =
  3.413 * (sin (2 * β))^3 * cos (6 * β) + (cos (2 * β))^3 * sin (6 * β) :=
by
  sorry

end trigonometric_identity_equiv_l487_487989


namespace solution_set_l487_487345

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 - 4*x else -(x^2 - 4*(-x))

theorem solution_set :
  (is_odd f) →
  (∀ x, f x = f_definition x) →
  { x : ℝ | f x > x } = { x : ℝ | -5 < x ∧ x < 0 } ∪ { x : ℝ | x > 5 } :=
by
  sorry

def f_definition (x : ℝ) : ℝ :=
  if x > 0 then x^2 - 4*x else -(x^2 - 4*(-x))

example (h : (is_odd f) ∧ (∀ x, f x = f_definition x)) :
  ( { x : ℝ | f x > x } = { x : ℝ | -5 < x ∧ x < 0 } ∪ { x : ℝ | x > 5 } ) :=
sorry

end solution_set_l487_487345


namespace solve_equation_l487_487182

theorem solve_equation (x : ℝ) : x * (x + 1) = 12 → (x = -4 ∨ x = 3) :=
by
  sorry

end solve_equation_l487_487182


namespace angle_BDC_l487_487202

theorem angle_BDC (A B C D : Type) [IsTriangle A B C] 
  (h1 : right_triangle A B C ∧ angle_A = 30 ∧ D_on_line_AC D A C BD_is_bisector)
  : angle_BDC = 60 :=
by
  sorry

end angle_BDC_l487_487202


namespace sum_reciprocals_12_l487_487928

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487928


namespace find_side_length_of_cut_out_square_l487_487225

noncomputable def cardboard_box (x : ℝ) : Prop :=
  let length_initial := 80
  let width_initial := 60
  let area_base := 1500
  let length_final := length_initial - 2 * x
  let width_final := width_initial - 2 * x
  length_final * width_final = area_base

theorem find_side_length_of_cut_out_square : ∃ x : ℝ, cardboard_box x ∧ 0 ≤ x ∧ (80 - 2 * x) > 0 ∧ (60 - 2 * x) > 0 ∧ x = 15 :=
by
  sorry

end find_side_length_of_cut_out_square_l487_487225


namespace range_of_fx_minus_x_l487_487087

noncomputable def f (x : ℝ) : ℝ :=
  if -5 ≤ x ∧ x ≤ -1 then -x
  else if -1 < x ∧ x ≤ 1 then -1
  else if 1 < x ∧ x ≤ 5 then x
  else 0

theorem range_of_fx_minus_x : 
  set.range (λ x : ℝ, f x - x) = set.Ioo (-2) 0 ∪ set.Icc 2 10 ∪ {0} :=
begin
  sorry
end

end range_of_fx_minus_x_l487_487087


namespace sequence_terms_l487_487374

/-- Given the sequence {a_n} with the sum of the first n terms S_n = n^2 - 3, 
    prove that a_1 = -2 and a_n = 2n - 1 for n ≥ 2. --/
theorem sequence_terms (a : ℕ → ℤ) (S : ℕ → ℤ)
  (hS : ∀ n : ℕ, S n = n^2 - 3)
  (h1 : ∀ n : ℕ, a n = S n - S (n - 1)) :
  a 1 = -2 ∧ (∀ n : ℕ, n ≥ 2 → a n = 2 * n - 1) :=
by {
  sorry
}

end sequence_terms_l487_487374


namespace filling_methods_count_l487_487493

def number_of_filling_methods : ℕ :=
  let total_ways := 72 * 72 in
  let invalid_case1 := 72 in
  let invalid_case2 := 16 * 72 in
  total_ways - invalid_case1 - invalid_case2

theorem filling_methods_count : number_of_filling_methods = 3960 := by
  sorry

end filling_methods_count_l487_487493


namespace max_points_of_intersection_l487_487573

theorem max_points_of_intersection (circles : Fin 2 → Circle) (lines : Fin 3 → Line) :
  number_of_intersections circles lines = 17 :=
sorry

end max_points_of_intersection_l487_487573


namespace sum_of_reciprocals_factors_12_l487_487730

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by sorry

end sum_of_reciprocals_factors_12_l487_487730


namespace exists_dense_H_with_chords_disjoint_from_K_l487_487055

open Set

variable (K : Set (EuclideanSpace ℝ 3)) (S2 : Set (EuclideanSpace ℝ 3))

-- Assume K is a closed subset of the closed unit ball in R^3
axiom closed_K : IsClosed K
axiom K_subset_closed_unit_ball : K ⊆ Metric.ball (0 : EuclideanSpace ℝ 3) 1

-- Define the property of the family of chords Ω
variable (Ω : Set (Set (EuclideanSpace ℝ 3)))

axiom chord_property : ∀ X Y ∈ S2, ∃ (X' Y' : EuclideanSpace ℝ 3), X' ∈ S2 ∧ Y' ∈ S2 ∧ Metric.dist X X' < 1 ∧ Metric.dist Y Y' < 1 ∧ ({X', Y'} ∈ Ω) ∧ Disjoint ({X', Y'} : Set (EuclideanSpace ℝ 3)) K

-- Define \( H \) is dense in \( S^2 \)
def dense_in_S2 (H : Set (EuclideanSpace ℝ 3)) : Prop :=
  ∀ x ∈ S2, ∀ ε > 0, ∃ y ∈ H, Metric.dist x y < ε

-- Define property of \( H \)
def chords_disjoint_from_K (H : Set (EuclideanSpace ℝ 3)) : Prop :=
  ∀ x y ∈ H, Disjoint ({x, y} : Set (EuclideanSpace ℝ 3)) K

-- Main theorem
theorem exists_dense_H_with_chords_disjoint_from_K : 
  ∃ H ⊆ S2, dense_in_S2 S2 H ∧ chords_disjoint_from_K K H := sorry

end exists_dense_H_with_chords_disjoint_from_K_l487_487055


namespace gcd_840_1764_l487_487532

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l487_487532


namespace solve_system_of_equations_l487_487121

theorem solve_system_of_equations :
  ∀ x y z : ℝ, (x = sqrt (2 * y + 3)) ∧ (y = sqrt (2 * z + 3)) ∧ (z = sqrt (2 * x + 3)) →
  (x = 3 ∧ y = 3 ∧ z = 3) :=
by
  sorry

end solve_system_of_equations_l487_487121


namespace min_sum_squares_distinct_elements_l487_487467

theorem min_sum_squares_distinct_elements :
  ∃ (p q r s t u v w : ℤ), 
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
    q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
    r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
    s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
    t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
    u ≠ v ∧ u ≠ w ∧
    v ≠ w ∧
    p ∈ {-8, -6, -4, -1, 1, 3, 5, 12} ∧
    q ∈ {-8, -6, -4, -1, 1, 3, 5, 12} ∧
    r ∈ {-8, -6, -4, -1, 1, 3, 5, 12} ∧
    s ∈ {-8, -6, -4, -1, 1, 3, 5, 12} ∧
    t ∈ {-8, -6, -4, -1, 1, 3, 5, 12} ∧
    u ∈ {-8, -6, -4, -1, 1, 3, 5, 12} ∧
    v ∈ {-8, -6, -4, -1, 1, 3, 5, 12} ∧
    w ∈ {-8, -6, -4, -1, 1, 3, 5, 12} ∧
    (p + q + r + s)^2 + (t + u + v + w)^2 = 2 :=
sorry

end min_sum_squares_distinct_elements_l487_487467


namespace smallest_three_digit_integer_solution_l487_487593

theorem smallest_three_digit_integer_solution :
  ∃ n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    (∃ a b c : ℕ,
      n = 100 * a + 10 * b + c ∧
      1 ≤ a ∧ a ≤ 9 ∧
      0 ≤ b ∧ b ≤ 9 ∧ 
      0 ≤ c ∧ c ≤ 9 ∧
      2 * n = 100 * c + 10 * b + a + 5) ∧ 
    n = 102 := by
{
  sorry
}

end smallest_three_digit_integer_solution_l487_487593


namespace sum_a_values_l487_487085

open Real

noncomputable def curve (n : ℕ) : ℝ → ℝ := fun x => x^(n+1)
noncomputable def tangent_intersection_x (n : ℕ) : ℝ := n / (n + 1)
noncomputable def a (n : ℕ) : ℝ := log10 (tangent_intersection_x n)

theorem sum_a_values :
  (∑ n in Finset.range 99, a (n + 1)) = -2 := by
  sorry

end sum_a_values_l487_487085


namespace D_180_equals_43_l487_487455

-- Define D(n) as the number of ways to express the positive integer n
-- as a product of integers strictly greater than 1, where the order of factors matters.
def D (n : Nat) : Nat := sorry  -- The actual implementation is not provided, as per instructions.

theorem D_180_equals_43 : D 180 = 43 :=
by
  sorry  -- The proof is omitted as the task specifies.

end D_180_equals_43_l487_487455


namespace sum_of_reciprocals_of_factors_of_12_l487_487655

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487655


namespace sum_of_reciprocals_of_factors_of_12_l487_487657

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487657


namespace main_theorem_l487_487456

-- Definitions based on conditions
def a : ℕ → ℝ := sorry -- Sequence of non-negative real numbers

axiom nonnegative_seq (n : ℕ) : 0 ≤ a n

axiom condition1 (k : ℕ) : a k - 2 * a (k + 1) + a (k + 2) ≥ 0

axiom condition2 (k : ℕ) : (∑ i in Finset.range k, a (i + 1)) ≤ 1

-- Prove the main statement
theorem main_theorem (k : ℕ) (hk : 0 < k) : 
  0 ≤ a k - a (k + 1) ∧ a k - a (k + 1) < 2 / k ^ 2 := 
sorry

end main_theorem_l487_487456


namespace sum_reciprocals_of_factors_12_l487_487843

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487843


namespace james_additional_votes_l487_487026

theorem james_additional_votes (votes_cast : ℕ) (percent_received : ℝ) (majority_percent : ℝ) (votes_james : ℕ) (needed_votes : ℕ) (additional_votes : ℕ) :
  votes_cast = 2000 →
  percent_received = 0.5 →
  majority_percent = 50 →
  votes_james = (percent_received / 100) * votes_cast →
  needed_votes = ((majority_percent / 100) * votes_cast).to_nat + 1 →
  additional_votes = needed_votes - votes_james →
  additional_votes = 991 :=
by sorry

end james_additional_votes_l487_487026


namespace spices_totals_spices_remaining_total_remaining_weight_proportions_used_total_weight_used_l487_487254

noncomputable def initial_pepper := 24
noncomputable def initial_salt := 12
noncomputable def initial_paprika := 8
noncomputable def initial_cumin := 5
noncomputable def initial_oregano := 10

noncomputable def used_pepper := 0.08
noncomputable def used_salt := 0.03
noncomputable def used_paprika := 0.05
noncomputable def used_cumin := 0.04
noncomputable def used_oregano := 0.06

theorem spices_totals :
  initial_pepper + initial_salt + initial_paprika + initial_cumin + initial_oregano = 59 :=
by
  sorry

theorem spices_remaining :
  initial_pepper - used_pepper = 23.92 ∧
  initial_salt - used_salt = 11.97 ∧
  initial_paprika - used_paprika = 7.95 ∧
  initial_cumin - used_cumin = 4.96 ∧
  initial_oregano - used_oregano = 9.94 :=
by
  sorry

theorem total_remaining_weight :
  initial_pepper - used_pepper +
  initial_salt - used_salt +
  initial_paprika - used_paprika +
  initial_cumin - used_cumin +
  initial_oregano - used_oregano = 58.74 :=
by
  sorry

theorem proportions_used :
  (used_pepper / initial_pepper) * 100 = 0.3333 ∧
  (used_salt / initial_salt) * 100 = 0.25 ∧
  (used_paprika / initial_paprika) * 100 = 0.625 ∧
  (used_cumin / initial_cumin) * 100 = 0.8 ∧
  (used_oregano / initial_oregano) * 100 = 0.6 :=
by
  sorry

theorem total_weight_used :
  used_pepper + used_salt + used_paprika + used_cumin + used_oregano = 0.26 :=
by
  sorry

end spices_totals_spices_remaining_total_remaining_weight_proportions_used_total_weight_used_l487_487254


namespace simplify_and_evaluate_l487_487118

noncomputable def a := 2 * Real.sin (Real.pi / 4) + (1 / 2) ^ (-1 : ℤ)

theorem simplify_and_evaluate :
  (a^2 - 4) / a / ((4 * a - 4) / a - a) + 2 / (a - 2) = -1 - Real.sqrt 2 := by
  sorry

end simplify_and_evaluate_l487_487118


namespace sum_of_reciprocals_of_factors_of_12_l487_487768

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487768


namespace max_possible_reward_zero_l487_487155

-- Let's define the problem in Lean 4

variables (a b c : ℕ) -- initial number of stones in three piles

-- Define the reward function when moving a stone from pile x to pile y
def reward (x y : ℕ) : ℤ := y - x + 1

-- The total reward should be 0 when all stones are back in their initial piles
theorem max_possible_reward_zero (moves : list (ℕ × ℕ)) :
  let total_reward := moves.foldl (λ acc (xy : ℕ × ℕ), acc + reward xy.1 xy.2) 0 in
  -- Condition: When all stones are back to their initial piles, total reward must be zero
  total_reward = 0 :=
sorry

end max_possible_reward_zero_l487_487155


namespace simplify_expression1_simplify_expression2_l487_487506

-- Problem 1
theorem simplify_expression1 (x y : ℤ) :
  (-3) * x + 2 * y - 5 * x - 7 * y = -8 * x - 5 * y :=
by sorry

-- Problem 2
theorem simplify_expression2 (a b : ℤ) :
  5 * (3 * a^2 * b - a * b^2) - 4 * (-a * b^2 + 3 * a^2 * b) = 3 * a^2 * b - a * b^2 :=
by sorry

end simplify_expression1_simplify_expression2_l487_487506


namespace sum_reciprocals_12_l487_487924

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487924


namespace problem_a_l487_487996

theorem problem_a (f : ℕ → ℕ) (h1 : f 1 = 2) (h2 : ∀ n, f (f n) = f n + 3 * n) : f 26 = 59 := 
sorry

end problem_a_l487_487996


namespace sum_reciprocal_factors_of_12_l487_487594

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487594


namespace sum_of_reciprocals_factors_12_l487_487821

theorem sum_of_reciprocals_factors_12 : (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 := sorry

end sum_of_reciprocals_factors_12_l487_487821


namespace max_intersection_points_of_lines_l487_487478

def are_distinct (lines : List Line) : Prop :=
  ∀ i j, i ≠ j → lines[i] ≠ lines[j]

def all_parallel (lines : List Line) : Prop :=
  ∀ i j, lines[i] ∥ lines[j]

def all_pass_through (lines : List Line) (P : Point) : Prop :=
  ∀ i, P ∈ lines[i]

noncomputable def maximum_points_of_intersection (lines : List Line) : ℕ := sorry

theorem max_intersection_points_of_lines (L : List Line) (A : Point) (h_distinct : are_distinct L)
  (h_parallel : ∀ n, 0 < n → 4 * n ≤ L.length → L[4 * n - 4] ∥ L[4 * n - 4])
  (h_pass_through : ∀ n, 0 < n → 4 * n - 3 ≤ L.length → A ∈ L[4 * n - 3]) :
  maximum_points_of_intersection L = 4351 :=
sorry

end max_intersection_points_of_lines_l487_487478


namespace sum_reciprocals_factors_12_l487_487695

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487695


namespace common_ratio_halves_l487_487332

variable {a : ℕ → ℝ}  -- Define the arithmetic sequence as a function

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem common_ratio_halves
  (h_arith_seq : is_arithmetic_sequence a d)
  (h1 : a 1 + a 3 = 10)
  (h2 : a 4 + a 6 = 5 / 4) :
  let q := (5 / 4 * 1 / 10) ^ (1 / 3) in
  q = 1 / 2 := by
  sorry  -- The proof is omitted

end common_ratio_halves_l487_487332


namespace sum_reciprocals_factors_12_l487_487625

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487625


namespace only_f_D_is_monotonically_increasing_on_0_2_l487_487190

-- Definitions of the given functions
def f_A (x : ℝ) : ℝ := (x - 2)^2
def f_B (x : ℝ) : ℝ := 1 / (x - 2)
def f_C (x : ℝ) : ℝ := sin (x - 2)
def f_D (x : ℝ) : ℝ := cos (x - 2)

-- Definition of a monotonically increasing function on an interval
def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  ∀ (x y : ℝ), a < x ∧ x < y ∧ y < b → f x ≤ f y

-- The main theorem to be proved
theorem only_f_D_is_monotonically_increasing_on_0_2 :
  (is_monotonically_increasing f_D 0 2) ∧ 
  ¬(is_monotonically_increasing f_A 0 2) ∧ 
  ¬(is_monotonically_increasing f_B 0 2) ∧ 
  ¬(is_monotonically_increasing f_C 0 2) :=
by
  sorry

end only_f_D_is_monotonically_increasing_on_0_2_l487_487190


namespace product_geometric_sequence_inserted_numbers_l487_487433

-- Conditions of the problem
variables {a : ℕ → ℝ} (a_1 : a 1 = 1) (a_10 : a 10 = 3)
           (geo_seq : ∀ m n p q, m + n = p + q → a m * a n = a p * a q)

-- Define a statement for the product of the inserted numbers
theorem product_geometric_sequence_inserted_numbers :
  (a 2) * (a 3) * (a 4) * (a 5) * (a 6) * (a 7) * (a 8) * (a 9) = 81 :=
by
  -- Inserting conditions as assumptions
  have h1 : a 2 * a 9 = 3, by { apply geo_seq 2 9 1 10, rw [a_1, a_10], linarith },
  have h2 : a 3 * a 8 = 3, by { apply geo_seq 3 8 1 10, rw [a_1, a_10], linarith },
  have h3 : a 4 * a 7 = 3, by { apply geo_seq 4 7 1 10, rw [a_1, a_10], linarith },
  have h4 : a 5 * a 6 = 3, by { apply geo_seq 5 6 1 10, rw [a_1, a_10], linarith },

  -- Using the properties of the geometric sequence
  calc
    (a 2) * (a 3) * (a 4) * (a 5) * (a 6) * (a 7) * (a 8) * (a 9)
      = (a 2 * a 9) * (a 3 * a 8) * (a 4 * a 7) * (a 5 * a 6) : by ring
  ... = 3 * 3 * 3 * 3 : by rw [h1, h2, h3, h4]
  ... = 81 : by norm_num
  sorry

end product_geometric_sequence_inserted_numbers_l487_487433


namespace function_behavior_l487_487525

def f (x : ℝ) : ℝ := (x + 2) / (x - 1)

def domain (x : ℝ) : Prop := 2 ≤ x ∧ x < 5 ∧ x ≠ 1

theorem function_behavior :
  (∀ x ∈ Icc 2 5, x ≠ 1 → ∃ y, y = f 2 ∧ y = 4) ∧
  (∀ y, y ∈ set.range (λ x, f x) → ∀ (ε : ℝ) (hε : ε > 0), ∃ x, x ∈ Ico 2 5 ∧ abs (f x - y) < ε ∧ abs (x - 5) < ε) :=
by sorry

end function_behavior_l487_487525


namespace ratio_of_speeds_l487_487044

noncomputable def jack_time_on_inclined_course : ℝ := 6
noncomputable def jill_time_on_inclined_course : ℝ := 5.7
def marathon_distance : ℝ := 60

-- Jack's average speed on the inclined course
noncomputable def jack_avg_speed : ℝ := marathon_distance / jack_time_on_inclined_course

-- Jill's average speed on the inclined course
noncomputable def jill_avg_speed : ℝ := marathon_distance / jill_time_on_inclined_course

-- Ratio of Jack's average speed to Jill's average speed
noncomputable def speed_ratio : ℝ := jack_avg_speed / jill_avg_speed

theorem ratio_of_speeds : speed_ratio = 1000 / 1053 := by
  have hjack : jack_avg_speed = 10 := by
    exact div_div_eq_div_mul marathon_distance jack_time_on_inclined_course
  have hjill : jill_avg_speed = 60 / 5.7 := by
    exact div_eq_div_iff_mul_eq_mul marathon_distance jill_time_on_inclined_course
  have hrat : 10 / (60 / 5.7) = 1000 / 1053 := by sorry -- Hand-waving the proof details for simplification
  rw [hjack, hjill] at hrat
  exact hrat

end ratio_of_speeds_l487_487044


namespace proof_bug_at_A_after_8_meters_l487_487058

noncomputable def P : ℕ → ℝ := sorry

axiom initial_conditions : P 0 = 1

axiom transition_probabilities (n : ℕ) : 
  P (n + 1) = 1 / 2 * P n + 1 / 4 * P n + 1 / 4 * P n

theorem proof_bug_at_A_after_8_meters :
  P 8 = 401 / 2187 :=
begin
  sorry
end

end proof_bug_at_A_after_8_meters_l487_487058


namespace range_of_a_l487_487370

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, 
    1 ≤ x ∧ x ≤ 2 ∧ 
    2 ≤ y ∧ y ≤ 3 → 
    x * y ≤ a * x^2 + 2 * y^2) ↔ 
  a ≥ -1 :=
by
  sorry

end range_of_a_l487_487370


namespace sum_reciprocals_factors_of_12_l487_487680

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487680


namespace probability_interval_xi_l487_487472

theorem probability_interval_xi (k : ℝ) (xi : ℕ → ℝ) :
  (∀ x, x ∈ Finset.range 6 → xi x = x * k / 15) →
  (∑ x in Finset.range 6, xi x) = 1 →
  (xi 1 + xi 2) = 1 / 5 :=
by
  intro h1 h2
  sorry

end probability_interval_xi_l487_487472


namespace sum_reciprocals_of_factors_of_12_l487_487963

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487963


namespace sqrt_equation_solution_l487_487286

variable (x : Real)

theorem sqrt_equation_solution
  (h1 : x >= 1) :
  (√(x + 5 - 6 * √(x - 1)) + √(x + 10 - 8 * √(x - 1)) = 3) ↔ (x = 5 ∨ x = 26) := 
sorry

end sqrt_equation_solution_l487_487286


namespace digits_base8_2015_l487_487386

theorem digits_base8_2015 : ∃ n : Nat, (8^n ≤ 2015 ∧ 2015 < 8^(n+1)) ∧ n + 1 = 4 := 
by 
  sorry

end digits_base8_2015_l487_487386


namespace common_root_l487_487288

noncomputable def f (λ x : ℝ) : ℝ := λ * x^3 - x^2 - x + (λ + 1)
noncomputable def g (λ x : ℝ) : ℝ := λ * x^2 - x - (λ + 1)

theorem common_root (λ x0 : ℝ) (h1 : f λ x0 = 0) (h2 : g λ x0 = 0) 
    : λ = -1 ∧ x0 = 0 := 
sorry

end common_root_l487_487288


namespace power_of_two_l487_487320

theorem power_of_two (b m n : ℕ) (h1 : b > 1) (h2 : m > n)
  (h3 : ∀ p : ℕ, p.prime → (p ∣ b^m - 1 ↔ p ∣ b^n - 1)) :
  ∃ k : ℕ, b + 1 = 2^k := 
sorry

end power_of_two_l487_487320


namespace sum_of_reciprocals_of_factors_of_12_l487_487773

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487773


namespace sum_of_reciprocals_of_factors_of_12_l487_487664

theorem sum_of_reciprocals_of_factors_of_12 : 
  (1 : ℚ) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 := 
by 
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487664


namespace sum_reciprocals_factors_12_l487_487745

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487745


namespace solve_for_q_l487_487120

theorem solve_for_q 
  (n m q : ℕ)
  (h1 : 5 / 6 = n / 60)
  (h2 : 5 / 6 = (m + n) / 90)
  (h3 : 5 / 6 = (q - m) / 150) : 
  q = 150 :=
sorry

end solve_for_q_l487_487120


namespace sum_first_40_terms_l487_487330

variable {a : ℕ → ℤ}

-- Conditions: The sequence satisfies a specified relation
def sequence_relation := ∀ n : ℕ, a (n + 1) + (-1:ℤ) ^ n * a n = 2 * n - 1

-- To Prove: The sum of the first 40 terms is 820
theorem sum_first_40_terms (h : sequence_relation a) :
  (∑ i in Finset.range 40, a i) = 820 := 
sorry

end sum_first_40_terms_l487_487330


namespace maria_total_money_l487_487093

theorem maria_total_money (Rene Florence Isha : ℕ) (hRene : Rene = 300)
  (hFlorence : Florence = 3 * Rene) (hIsha : Isha = Florence / 2) :
  Isha + Florence + Rene = 1650 := by
  sorry

end maria_total_money_l487_487093


namespace sum_reciprocals_12_l487_487930

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487930


namespace sum_of_reciprocals_of_factors_of_12_l487_487807

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487807


namespace sum_reciprocals_of_factors_of_12_l487_487970

theorem sum_reciprocals_of_factors_of_12 :
  (∑ n in {n | n ∣ 12 ∧ n > 0}.to_finset, (1 : ℚ) / n) = 7 / 3 :=
by
  sorry

end sum_reciprocals_of_factors_of_12_l487_487970


namespace sum_reciprocals_factors_12_l487_487623

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487623


namespace detect_two_counterfeits_detect_three_counterfeits_l487_487436

-- Definitions for counterfeit detection conditions and problem
def distinct_natural_denominations (banknotes : List ℕ) : Prop :=
  (banknotes.nodup)

def sum_of_genuine (detector : List ℕ → ℕ) (genuine_notes : List ℕ) : ℕ :=
  detector genuine_notes

-- Prove for N=2
theorem detect_two_counterfeits (banknotes : List ℕ) (detector : List ℕ → ℕ) (N : ℕ) 
  (h₁ : distinct_natural_denominations banknotes) (h₂ : N = 2) :
  ∃ (S₀ S₁ : List ℕ), detector S₀ = detector banknotes ∧ detector S₁ = detector S₀ - (sum_of_genuine detector S₀ - detector banknotes) :=
sorry

-- Prove for N=3
theorem detect_three_counterfeits (banknotes : List ℕ) (detector : List ℕ → ℕ) (N : ℕ) 
  (h₁ : distinct_natural_denominations banknotes) (h₂ : N = 3) :
  ∃ (S₀ S₁ S₂ : List ℕ), detector S₀ = detector banknotes ∧ detector S₁ = detector S₀ - (sum_of_genuine detector S₀ - detector banknotes) ∧ detector S₂ = detector S₁ - (sum_of_genuine detector S₁ - detector banknotes) :=
sorry

end detect_two_counterfeits_detect_three_counterfeits_l487_487436


namespace sum_reciprocals_factors_of_12_l487_487681

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487681


namespace sum_reciprocals_factors_12_l487_487637

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487637


namespace gain_percent_l487_487216

theorem gain_percent (cp sp : ℝ) (h_cp : cp = 900) (h_sp : sp = 1080) :
    ((sp - cp) / cp) * 100 = 20 :=
by
    sorry

end gain_percent_l487_487216


namespace sum_of_reciprocals_factors_12_l487_487779

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487779


namespace sum_reciprocals_factors_12_l487_487616

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487616


namespace sum_reciprocals_factors_12_l487_487622

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487622


namespace sufficient_condition_range_a_l487_487316

theorem sufficient_condition_range_a (a : ℝ) :
  (∀ x, (2 * a ≤ x ∧ x ≤ a^2 + 1) → (x^2 - 3 * (a + 1) * x + 6 * a + 2 ≤ 0)) ↔
  (1 ≤ a ∧ a ≤ 3) ∨ (a = -1) := by
  sorry

end sufficient_condition_range_a_l487_487316


namespace greatest_multiple_of_8_remainder_eq_120_l487_487059

def no_repeating_digits (n : ℕ) : Prop :=
  let digits := (n.digits 10) in
  digits.nodup

def greatest_multiple_of_8 (n : ℕ) : Prop :=
  ∃ m, no_repeating_digits m ∧ m % 8 = 0 ∧ m ≤ n

theorem greatest_multiple_of_8_remainder_eq_120 :
  ∀ N : ℕ, greatest_multiple_of_8 N → N % 1000 = 120 :=
by
  intro N hN
  sorry

end greatest_multiple_of_8_remainder_eq_120_l487_487059


namespace max_points_of_intersection_l487_487583

-- Definitions from the conditions
def circles := 2
def lines := 3

-- Define the problem of the greatest intersection number
theorem max_points_of_intersection (c : ℕ) (l : ℕ) (h_c : c = circles) (h_l : l = lines) : 
  (2 + (l * 2 * c) + (l * (l - 1) / 2)) = 17 :=
by
  rw [h_c, h_l]
  -- We have 2 points from circle intersections
  -- 12 points from lines intersections with circles
  -- 3 points from lines intersections with lines
  -- Hence, 2 + 12 + 3 = 17
  exact Eq.refl 17

end max_points_of_intersection_l487_487583


namespace projection_matrix_correct_l487_487065

-- Define the normal vector and the projection matrix.
def normal_vector : ℝ × ℝ × ℝ := (2, -1, 2)
def Q_matrix : ℝ → ℝ → ℝ → ℝ × ℝ × ℝ := 
  λ x y z, 
    ( (5 * x + 2 * y - 4 * z) / 9, 
      (2 * x + 10 * y - 2 * z) / 9, 
      (-4 * x + 2 * y + 5 * z) / 9 )

-- Theorem to prove the matrix Q correctly projects any vector onto the plane.
theorem projection_matrix_correct (v : ℝ × ℝ × ℝ) : 
    let ⟨x, y, z⟩ := v in 
    Q_matrix x y z = ((v.1) * (5/9) + (v.2) * (2/9) - (v.3) * (4/9), 
                      (v.1) * (2/9) + (v.2) * (10/9) - (v.3) * (2/9), 
                      (v.1) * (-4/9) + (v.2) * (2/9) + (v.3) * (5/9)) := 
by sorry

end projection_matrix_correct_l487_487065


namespace triangle_nature_l487_487411

theorem triangle_nature (a b c : ℝ) (h : (a - b) * (a^2 + b^2 - c^2) = 0) :
  (a = b ∨ a^2 + b^2 = c^2) :=
begin
  sorry
end

end triangle_nature_l487_487411


namespace median_is_a1012_std_dev_new_data2_l487_487509

variable {α : Type*} [LinearOrderedField α]

-- Define the data set and their properties
variables (a : Fin 2023 → α) 
variable (strictly_increasing : ∀ i j : Fin 2023, i < j → a i < a j)

-- Define mean, median, and standard deviation
noncomputable def mean (a : Fin 2023 → α) : α := (Finset.univ.sum (λ i, a i)) / 2023

noncomputable def median (a : Fin 2023 → α) : α := a ⟨1011, by simp⟩  -- Using ⟨1011, by simp⟩ to denote Fin 2023

noncomputable def std_dev (a : Fin 2023 → α) (μ : α) : α := 
  Real.sqrt ((Finset.univ.sum (λ i, (a i - μ)^2)) / 2023)

-- Defining the new data set transformations
def new_data1 (a : Fin 2023 → α) : Fin 2023 → α := λ i, a i + 2

def new_data2 (a : Fin 2023 → α) : Fin 2023 → α := λ i, 2 * a i + 1

-- Main theorems to prove
theorem median_is_a1012 : median a = a ⟨1011, by simp⟩ :=
  sorry

theorem std_dev_new_data2 (s : α) (h : std_dev a (mean a) = s) : 
  std_dev (new_data2 a) (mean (new_data2 a)) = 2 * s :=
  sorry

end median_is_a1012_std_dev_new_data2_l487_487509


namespace rearrange_digits_to_perfect_square_l487_487113

theorem rearrange_digits_to_perfect_square :
  ∃ n : ℤ, 2601 = n ^ 2 ∧ (∃ (perm : List ℤ), perm = [2, 0, 1, 6] ∧ perm.permutations ≠ List.nil) :=
by
  sorry

end rearrange_digits_to_perfect_square_l487_487113


namespace sum_reciprocals_factors_12_l487_487883

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487883


namespace sum_reciprocal_factors_of_12_l487_487609

theorem sum_reciprocal_factors_of_12 :
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 : ℚ) = 7/3 :=
sorry

end sum_reciprocal_factors_of_12_l487_487609


namespace compare_radii_l487_487211

noncomputable def radius_of_circle_A : ℝ := 3
noncomputable def radius_of_circle_B : ℝ := 4
def radius_of_circle_C : ℝ := 2

theorem compare_radii :
  (radius_of_circle_C < radius_of_circle_A) ∧ (radius_of_circle_A < radius_of_circle_B) :=
by {
  have r_A : radius_of_circle_A = 3, by sorry,
  have r_B : radius_of_circle_B = 4, by sorry,
  have r_C : radius_of_circle_C = 2, by sorry,
  sorry
}

end compare_radii_l487_487211


namespace part1_price_reduction_5_part2_price_reduction_2100_part3_impossible_2200_l487_487234

variable (original_profit_per_item : ℝ)
variable (original_daily_sales_volume : ℝ)
variable (price_reduction_per_unit : ℝ)
variable (additional_items_per_dollar : ℝ)

/-- Part (1): Proving new sales volume and profit after a $5 reduction -/
theorem part1_price_reduction_5 (original_profit_per_item : ℝ) (original_daily_sales_volume : ℝ)
(additional_items_per_dollar : ℝ) :
  let reduced_price := 5 in
  let new_sales_volume := original_daily_sales_volume + additional_items_per_dollar * reduced_price in
  let total_daily_profit := (original_profit_per_item - reduced_price) * new_sales_volume in
  new_sales_volume = 40 ∧ total_daily_profit = 1800 :=
sorry

/-- Part (2): Proving price reduction for $2100 daily profit -/
theorem part2_price_reduction_2100 (original_profit_per_item : ℝ) (original_daily_sales_volume : ℝ)
(additional_items_per_dollar : ℝ) :
  ∃ (reduced_price : ℝ), 
    let new_sales_volume := original_daily_sales_volume + additional_items_per_dollar * reduced_price in
    let total_daily_profit := (original_profit_per_item - reduced_price) * new_sales_volume in
    total_daily_profit = 2100 :=
sorry

/-- Part (3): Proving impossibility of achieving $2200 daily profit -/
theorem part3_impossible_2200 (original_profit_per_item : ℝ) (original_daily_sales_volume : ℝ)
(additional_items_per_dollar : ℝ) :
  ¬ ∃ (reduced_price : ℝ), 
    let new_sales_volume := original_daily_sales_volume + additional_items_per_dollar * reduced_price in
    let total_daily_profit := (original_profit_per_item - reduced_price) * new_sales_volume in
    total_daily_profit = 2200 :=
sorry

end part1_price_reduction_5_part2_price_reduction_2100_part3_impossible_2200_l487_487234


namespace sum_reciprocals_factors_12_l487_487647

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487647


namespace sum_reciprocals_factors_12_l487_487941

theorem sum_reciprocals_factors_12 : 
  let factors := [1, 2, 3, 4, 6, 12] in
  (factors.map (fun x => (1:ℚ)/x)).sum = 7/3 := 
by
  let factors := [1, 2, 3, 4, 6, 12]
  let reciprocals := factors.map (fun x => (1:ℚ) / x)
  have h : reciprocals = [1, 1/2, 1/3, 1/4, 1/6, 1/12] := by simp
  rw [h]
  have sum_reciprocals : (reciprocals).sum = 1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 := by simp
  rw [sum_reciprocals]
  norm_num
  exact sorry

end sum_reciprocals_factors_12_l487_487941


namespace total_water_volume_correct_l487_487228

noncomputable def compute_total_water_volume : ℝ :=
let flow_rate_first_section := (3 * 1000) / 60 in
let flow_rate_second_section := (4 * 1000) / 60 in
let flow_rate_third_section := (5 * 1000) / 60 in
let volume_first_section := flow_rate_first_section * 5 * 25 in
let volume_second_section := flow_rate_second_section * 7 * 25 in
let volume_third_section := flow_rate_third_section * 9 * 25 in
volume_first_section + volume_second_section + volume_third_section

theorem total_water_volume_correct :
  compute_total_water_volume = 36666.75 := by
sorry

end total_water_volume_correct_l487_487228


namespace sum_reciprocals_factors_12_l487_487621

theorem sum_reciprocals_factors_12 :
  (1 / 1) + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487621


namespace min_value_fraction_sum_l487_487409

theorem min_value_fraction_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (1 / a + 4 / b) ≥ 9 :=
begin
  sorry
end

end min_value_fraction_sum_l487_487409


namespace sum_reciprocals_of_factors_12_l487_487840

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487840


namespace sum_reciprocals_factors_12_l487_487890

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l487_487890


namespace probability_sum_23_correct_l487_487219

-- Definitions for the faces of the dice
def die1_faces : Finset ℕ := Finset.range 19  -- Faces 1 through 18 plus a blank face (0)
def die2_faces : Finset ℕ := Finset.union (Finset.range' 2 9) (Finset.range' 11 21)  -- Faces 2 through 9 plus 11 through 21 plus a blank face (0)

-- Probability that the sum of the numbers showing on two dice is 23
def probability_sum_23 : ℚ :=
  let valid_combinations := (die1_faces.product die2_faces).filter (λ (p : ℕ × ℕ), p.1 + p.2 = 23)
  (valid_combinations.card : ℚ) / ((die1_faces.card) * (die2_faces.card))

theorem probability_sum_23_correct :
  probability_sum_23 = 2 / 25 :=
sorry

end probability_sum_23_correct_l487_487219


namespace maze_paths_count_l487_487250

-- Define the DP table and the grid size
def maze_paths (n : ℕ) : ℕ × ℕ → ℕ
| (0, 0) := 1
| (i, j) :=
  let up := if i > 0 then maze_paths (i - 1, j) else 0
  let left := if j > 0 then maze_paths (i, j - 1) else 0
  let right := if j < n - 1 then maze_paths (i, j + 1) else 0
  up + left + right

theorem maze_paths_count : maze_paths 5 (4, 4) = 16 := 
sorry

end maze_paths_count_l487_487250


namespace Sue_made_22_buttons_l487_487447

-- Definitions of the conditions and the goal
def Mari_buttons : ℕ := 8
def Kendra_buttons := 4 + 5 * Mari_buttons
def Sue_buttons := Kendra_buttons / 2

-- Theorem statement
theorem Sue_made_22_buttons : Sue_buttons = 22 :=
by
  -- Definitions used here
  unfold Mari_buttons Kendra_buttons Sue_buttons
  -- Calculation
  rw [show 4 + 5 * 8 = 44, by norm_num, show 44 / 2 = 22, by norm_num]
  rfl

end Sue_made_22_buttons_l487_487447


namespace sum_of_reciprocals_of_factors_of_12_l487_487912

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487912


namespace navigation_critical_height_l487_487230

-- Conditions
def parabola (x : ℝ) : ℝ := -(1 / 25) * x^2
def water_surface_width := 20
def apex_height := 4
def normal_water_depth := 2
def safe_navigation_width := 18

-- Define d(h)
def d (h : ℝ) : ℝ :=
  10 * sqrt (4 - h)

-- Problem statements
theorem navigation_critical_height :
  ∀ (h : ℝ), d(h) < safe_navigation_width  ↔ normal_water_depth + h = 2.76 :=
by
  intro h
  sorry

end navigation_critical_height_l487_487230


namespace arithmetic_progression_cubic_eq_l487_487502

theorem arithmetic_progression_cubic_eq (x y z u : ℤ) (d : ℤ) :
  (x, y, z, u) = (3 * d, 4 * d, 5 * d, 6 * d) →
  x^3 + y^3 + z^3 = u^3 →
  ∃ d : ℤ, x = 3 * d ∧ y = 4 * d ∧ z = 5 * d ∧ u = 6 * d :=
by sorry

end arithmetic_progression_cubic_eq_l487_487502


namespace sum_reciprocal_factors_12_l487_487858

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487858


namespace sum_reciprocals_factors_12_l487_487651

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487651


namespace sum_of_reciprocals_of_factors_of_12_l487_487757

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487757


namespace not_necessarily_form_rectangle_l487_487103

-- Let T be the type representing identical non-isosceles right triangles
constant T : Type

-- Condition: There is a finite set of identical non-isosceles right triangles
constant identical_non_isosceles_right_triangles : Set T

-- Defining the notion of forming a larger rectangle without gaps or overlaps
def forms_larger_rectangle (S : Set T) : Prop := sorry

-- Proposition: The necessity of any two triangles forming a rectangle
constant forms_rectangle : T → T → Prop

-- The main theorem to state the problem
theorem not_necessarily_form_rectangle
  (H : forms_larger_rectangle identical_non_isosceles_right_triangles) :
  ∃ t1 t2 ∈ identical_non_isosceles_right_triangles, ¬ forms_rectangle t1 t2 :=
sorry

end not_necessarily_form_rectangle_l487_487103


namespace sum_reciprocals_factors_of_12_l487_487677

-- Define the set of natural-number factors of 12.
def factors_of_12 := {1, 2, 3, 4, 6, 12}

-- Definition of reciprocal sum calculation for a set of numbers.
def sum_of_reciprocals (s : Set ℕ) : ℚ :=
  s.to_finset.sum (λ n, (1 : ℚ) / n)

-- Prove that the sum of reciprocals of factors of 12 is 7/3.
theorem sum_reciprocals_factors_of_12 : sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_reciprocals_factors_of_12_l487_487677


namespace mike_earnings_after_sales_tax_l487_487095

theorem mike_earnings_after_sales_tax :
  (let total_videos := 20 in
   let non_working_videos := 11 in
   let price_per_game := 8 in
   let sales_tax_rate := 0.12 in
   let working_videos := total_videos - non_working_videos in
   let revenue_before_tax := working_videos * price_per_game in
   revenue_before_tax = 72) :=
by
  let total_videos := 20
  let non_working_videos := 11
  let price_per_game := 8
  let sales_tax_rate := 0.12
  let working_videos := total_videos - non_working_videos
  let revenue_before_tax := working_videos * price_per_game
  sorry

end mike_earnings_after_sales_tax_l487_487095


namespace population_net_increase_l487_487028

-- Definitions of conditions
def birth_rate := 7 / 2 -- 7 people every 2 seconds
def death_rate := 1 / 2 -- 1 person every 2 seconds
def seconds_in_a_day := 86400 -- Number of seconds in one day

-- Definition of the total births in one day
def total_births_per_day := birth_rate * seconds_in_a_day

-- Definition of the total deaths in one day
def total_deaths_per_day := death_rate * seconds_in_a_day

-- Proposition to prove the net population increase in one day
theorem population_net_increase : total_births_per_day - total_deaths_per_day = 259200 := by
  sorry

end population_net_increase_l487_487028


namespace sum_reciprocals_12_l487_487920

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487920


namespace max_total_profit_max_average_profit_l487_487220

-- Conditions of the problem
def initial_cost : ℕ := 980000
def first_year_cost : ℕ := 120000
def annual_increase : ℕ := 40000
def annual_income : ℕ := 500000

-- Total cost over n years
def total_cost (n : ℕ) : ℕ :=
  (n * (2 * first_year_cost + (n - 1) * annual_increase)) / 2

-- Total income over n years
def total_income (n : ℕ) : ℕ :=
  annual_income * n

-- Total profit after n years
def total_profit (n : ℕ) : ℕ :=
  total_income n - total_cost n - initial_cost

-- Annual average profit after n years
def annual_average_profit (n : ℕ) : ℚ :=
  (total_profit n : ℚ) / n

-- Problem 1: Prove that the total profit is maximized after 10 years and the maximum profit is 1,020,000 yuan
theorem max_total_profit : total_profit 10 = 1020000 := sorry

-- Problem 2: Prove that the maximum annual average profit is achieved after 9 years and is approximately 22,222.22 yuan
theorem max_average_profit : annual_average_profit 9 ≈ 22222.22 := sorry

end max_total_profit_max_average_profit_l487_487220


namespace projection_matrix_correct_l487_487068

noncomputable def Q_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  !![
    [7/9, 1/9, -2/9],
    [1/9, 11/18, 1/9],
    [-2/9, 1/9, 7/9]
  ]

def normal_vector : Vector (Fin 3) ℝ :=
  !![
    [2],
    [-1],
    [2]
  ]

def projection_on_plane (v : Vector (Fin 3) ℝ) : Vector (Fin 3) ℝ :=
  Q_matrix ⬝ v

theorem projection_matrix_correct (v : Vector (Fin 3) ℝ) :
  projection_on_plane v = Q_matrix ⬝ v :=
sorry

end projection_matrix_correct_l487_487068


namespace sum_of_reciprocals_of_factors_of_12_l487_487895

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487895


namespace total_height_of_three_pipes_l487_487024

/-- Given three identical cylindrical pipes each with a diameter of 12 cm, 
   stacked in a triangular formation such that they touch each other, 
   prove that the total height of the stack is 12 + 12\sqrt{3} cm.
-/
noncomputable def total_height_stack := ∀ (d : ℝ) (n : ℕ),
  d = 12 ∧ n = 3 → 
  ∃ (h : ℝ), h = 12 + 12 * Real.sqrt 3

theorem total_height_of_three_pipes :
  total_height_stack 12 3 :=
begin
  sorry
end

end total_height_of_three_pipes_l487_487024


namespace second_increase_is_40_l487_487141

variable (P : ℝ) (x : ℝ)

def second_increase (P : ℝ) (x : ℝ) : Prop :=
  1.30 * P * (1 + x / 100) = 1.82 * P

theorem second_increase_is_40 (P : ℝ) : ∃ x, second_increase P x ∧ x = 40 := by
  use 40
  sorry

end second_increase_is_40_l487_487141


namespace sum_reciprocals_factors_12_l487_487698

theorem sum_reciprocals_factors_12 : ∑ (n : ℕ) in {1, 2, 3, 4, 6, 12}, (1 / n : ℚ) = 7 / 3 := 
by 
  sorry

end sum_reciprocals_factors_12_l487_487698


namespace combined_height_is_120_l487_487501

-- Definitions for conditions
def SaraHeight (S : ℕ) : Prop :=
  ∃ J, J = 2 * S + 6 ∧ J = 82

-- Statement of the problem
theorem combined_height_is_120 (S : ℕ) :
  SaraHeight S → (S + 82 = 120) :=
by
  intro h
  cases h with J hJ
  cases hJ with hJ1 hJ2
  rw [hJ2] at hJ1
  have : 2 * S + 6 = 82 := hJ1
  have : 2 * S = 76 := by linarith
  have : S = 38 := by linarith
  rw [this]
  norm_num
  exact sorry

end combined_height_is_120_l487_487501


namespace john_final_pace_l487_487444

noncomputable def john_pace (john_behind steve_speed john_ahead race_duration : ℝ) : ℝ :=
  (steve_speed * race_duration + john_behind + john_ahead) / race_duration

theorem john_final_pace :
  john_pace 12 3.7 2 28 = 4.2 :=
by
  -- Definitions and conditions
  let john_behind := 12 -- John is 12 meters behind Steve
  let steve_speed := 3.7 -- Steve's speed is 3.7 m/s
  let john_ahead := 2 -- John finishes 2 meters ahead of Steve
  let race_duration := 28 -- Race lasts 28 seconds
  let J := (steve_speed * race_duration + john_behind + john_ahead) / race_duration

  -- The actual proof
  sorry

end john_final_pace_l487_487444


namespace steps_in_five_days_l487_487109

def steps_to_school : ℕ := 150
def daily_steps : ℕ := steps_to_school * 2
def days : ℕ := 5

theorem steps_in_five_days : daily_steps * days = 1500 := by
  sorry

end steps_in_five_days_l487_487109


namespace sum_of_reciprocals_factors_12_l487_487777

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487777


namespace sum_of_reciprocals_factors_12_l487_487786

theorem sum_of_reciprocals_factors_12 : 
  (1 : ℚ) + 1/2 + 1/3 + 1/4 + 1/6 + 1/12 = 7/3 := 
by
  sorry

end sum_of_reciprocals_factors_12_l487_487786


namespace sum_of_reciprocals_of_factors_of_12_l487_487906

-- Define the relationship that a number is a factor of 12
def is_factor_of_12 (d : ℕ) : Prop := 12 % d = 0

-- The set of all natural-number factors of 12
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}.toFinset

-- The sum of the reciprocals of the elements of a set of natural numbers
noncomputable def sum_of_reciprocals (s : Finset ℕ) : ℚ :=
  ∑ d in s, (1 : ℚ) / d

-- Statement that needs to be proven
theorem sum_of_reciprocals_of_factors_of_12 :
  sum_of_reciprocals factors_of_12 = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487906


namespace sin_negative_300_eq_l487_487974

theorem sin_negative_300_eq : Real.sin (-(300 * Real.pi / 180)) = Real.sqrt 3 / 2 :=
by
  -- Periodic property of sine function: sin(theta) = sin(theta + 360 * n)
  have periodic_property : ∀ θ n : ℤ, Real.sin θ = Real.sin (θ + n * 2 * Real.pi) :=
    by sorry
  -- Known value: sin(60 degrees) = sqrt(3)/2
  have sin_60 : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  -- Apply periodic_property to transform sin(-300 degrees) to sin(60 degrees)
  sorry

end sin_negative_300_eq_l487_487974


namespace johns_final_push_duration_l487_487993

noncomputable def time_of_johns_final_push
  (initial_gap : ℝ)
  (john_speed : ℝ)
  (steve_speed : ℝ)
  (john_final_lead : ℝ) : ℝ :=
  (initial_gap + john_final_lead) / (john_speed - steve_speed) 

theorem johns_final_push_duration : 
  time_of_johns_final_push 15 4.2 3.8 2 = 42.5 :=
by
  unfold time_of_johns_final_push
  simp
  norm_num
  sorry

end johns_final_push_duration_l487_487993


namespace sum_reciprocals_of_factors_12_l487_487835

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487835


namespace maximal_label_value_l487_487453

theorem maximal_label_value (n : ℕ) (h : n ≥ 3) :
  ∃ r : ℕ, (∀ (edges : ℕ → ℕ)
  (h_edges : ∀ e, edges e ∈ Finset.range r)
  (h_occurrence : ∀ i, ∃ e, edges e = i)
  (h_triangle : ∀ (a b c : ℕ), 
    let labels := [edges a, edges b, edges c] in 
    ∃ x y z, list.perm labels [x, y, x] ∧ x > y),
    r = n - 1 ∧ 
    ∃ k : ℤ, k = (nat.factorial n * nat.factorial (n-1)) / 2^(n-1))) := sorry

end maximal_label_value_l487_487453


namespace brick_length_cm_l487_487209

theorem brick_length_cm
    (L : ℕ)
    (brick_volume : ℕ := L * 10 * 8)
    (wall_volume : ℕ := 10 * 100 * 8 * 100 * 24.5 * 100)
    (bricks_needed : ℕ := 12250)
    (total_bricks_volume : ℕ := bricks_needed * brick_volume) :
  total_bricks_volume = wall_volume → L = 2000 := by
  sorry

end brick_length_cm_l487_487209


namespace max_points_of_intersection_l487_487575

theorem max_points_of_intersection (circles : Fin 2 → Circle) (lines : Fin 3 → Line) :
  number_of_intersections circles lines = 17 :=
sorry

end max_points_of_intersection_l487_487575


namespace integers_between_cubes_l487_487391

theorem integers_between_cubes (a b : ℝ) (ha : a = 9.2) (hb : b = 9.3) : 
  let x := a^3,
      y := b^3
  in (⌊y⌋ - ⌊x⌋) = 26 := by
  sorry

end integers_between_cubes_l487_487391


namespace difference_pencils_l487_487253

theorem difference_pencils (x : ℕ) (h1 : 162 = x * n_g) (h2 : 216 = x * n_f) : n_f - n_g = 3 :=
by
  sorry

end difference_pencils_l487_487253


namespace smallest_integer_value_of_m_l487_487014

theorem smallest_integer_value_of_m (x y m : ℝ) 
  (h1 : 3*x + y = m + 8) 
  (h2 : 2*x + 2*y = 2*m + 5) 
  (h3 : x - y < 1) : 
  m >= 3 := 
sorry

end smallest_integer_value_of_m_l487_487014


namespace perimeter_of_polygon_l487_487278

theorem perimeter_of_polygon :
  ∃ (ABCDEFGH : Polygon) (side_lengths : Fin 8 → ℕ), 
  (∀ i, side_lengths i ∈ ℕ) ∧
  (divide_into_rectangle_and_square ABCDEFGH) ∧
  let s : ℕ := side_length_of_square ABCDEFGH in
  let r_area : ℕ := area_of_rectangle ABCDEFGH in
  let s_area : ℕ := s * s in
  s_area > r_area ∧
  s_area * r_area = 98 ∧
  perimeter_of ABCDEFGH = 32 :=
sorry

end perimeter_of_polygon_l487_487278


namespace largest_invertible_interval_l487_487263

open Set

noncomputable def g (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 9
def x_vertex : ℝ := -1

theorem largest_invertible_interval (x : ℝ) (h : x ≤ x_vertex) :
  g ⁻¹' (g '' Icc (x, x_vertex)) = Icc (x, x_vertex) := sorry

end largest_invertible_interval_l487_487263


namespace orange_juice_usage_l487_487125

theorem orange_juice_usage :
  let total_oranges := 8 -- in million tons
  let export_percent := 0.30
  let orange_juice_percent := 0.60
  let remaining_oranges := total_oranges * (1 - export_percent)
  let orange_juice_used := remaining_oranges * orange_juice_percent
  (Float.round (orange_juice_used * 10) / 10) = 3.4 :=
by
  let total_oranges := 8 -- in million tons
  let export_percent := 0.30
  let orange_juice_percent := 0.60
  let remaining_oranges := total_oranges * (1 - export_percent)
  let orange_juice_used := remaining_oranges * orange_juice_percent
  have orange_juice_used_val : orange_juice_used = 3.36 := by
    calc
      remaining_oranges = 8 * 0.70 := by norm_num
      orange_juice_used = 5.6 * 0.60 := by norm_num
  show (Float.round (3.36 * 10) / 10) = 3.4
  sorry

end orange_juice_usage_l487_487125


namespace sum_reciprocals_of_factors_12_l487_487848

theorem sum_reciprocals_of_factors_12 : 
  (1 + (1 / 2) + (1 / 3) + (1 / 4) + (1 / 6) + (1 / 12) = 7 / 3) :=
by
  sorry

end sum_reciprocals_of_factors_12_l487_487848


namespace primes_in_range_l487_487078

open Nat

def f (x p : ℕ) : ℕ := x^2 + x + p

theorem primes_in_range (p : ℕ) (hp : ∀ n ≤ ⌊sqrt (p / 3)⌋, Prime (f n p)) :
  ∀ x < p - 1, Prime (f x p) := by
  sorry

end primes_in_range_l487_487078


namespace bubble_sort_prob_l487_487124

theorem bubble_sort_prob :
  ∃ p q : ℕ, Nat.gcd p q = 1 ∧ (↑p / ↑q : ℚ) = 1 / 132 ∧ (p + q = 133) :=
by
  sorry

end bubble_sort_prob_l487_487124


namespace sum_reciprocals_factors_12_l487_487736

theorem sum_reciprocals_factors_12 :
  let factors := {1, 2, 3, 4, 6, 12}
  let reciprocals := (λ x, 1 / x : (finset ℤ))
  let sum_reciprocals := (∑ x in factors, reciprocals x)
  sum_reciprocals = 2.333 := 
by
  sorry

end sum_reciprocals_factors_12_l487_487736


namespace blue_highlighters_l487_487020

theorem blue_highlighters (pink yellow total blue : Nat) (h₁ : pink = 4) (h₂ : yellow = 2) (h₃ : total = 11) :
  blue = total - (pink + yellow) → blue = 5 :=
by
  intros h₄
  rw [h₁, h₂, h₃] at h₄
  exact h₄

-- Proving the given problem: Total number of blue highlighters is 5 given the conditions
example :
  blue_highlighters 4 2 11 5 (by rfl) (by rfl) (by rfl) (by rfl) :=
by
  refl

end blue_highlighters_l487_487020


namespace proj_matrix_det_zero_l487_487457

open Matrix

def projection_matrix (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let den := a ^ 2 + b ^ 2
  Matrix.of ![
    ![a^2 / den, a * b / den],
    ![a * b / den, b^2 / den]
  ]

theorem proj_matrix_det_zero :
  det (projection_matrix 3 2) = 0 :=
by
  sorry

end proj_matrix_det_zero_l487_487457


namespace greatest_base7_3_digit_divisible_by_7_l487_487166

theorem greatest_base7_3_digit_divisible_by_7 :
  ∃ n : ℕ, n < 7^3 ∧ n ≥ 7^2 ∧ 7 ∣ n ∧ nat.to_digits 7 n = [6, 6, 0] := 
sorry

end greatest_base7_3_digit_divisible_by_7_l487_487166


namespace min_children_see_ear_l487_487033

theorem min_children_see_ear (n : ℕ) : ∃ (k : ℕ), k = n + 2 :=
by
  sorry

end min_children_see_ear_l487_487033


namespace dot_product_eq_neg29_l487_487377

-- Given definitions and conditions
variables (a b : ℝ × ℝ)

-- Theorem to prove the dot product condition.
theorem dot_product_eq_neg29 (h1 : a + b = (2, -4)) (h2 : 3 • a - b = (-10, 16)) :
  a.1 * b.1 + a.2 * b.2 = -29 :=
sorry

end dot_product_eq_neg29_l487_487377


namespace sum_reciprocals_12_l487_487929

-- Define the natural-number factors of 12
def factors_of_12 := [1, 2, 3, 4, 6, 12]

-- Define the sum of the reciprocals of these factors
def sum_of_reciprocals (l : List ℕ) : ℚ :=
  l.foldl (λ acc x → acc + (1 / x : ℚ)) 0

theorem sum_reciprocals_12 : 
  sum_of_reciprocals factors_of_12 = 7 / 3 := 
by
  sorry

end sum_reciprocals_12_l487_487929


namespace emily_made_amount_l487_487276

theorem emily_made_amount (chocolate_cost : ℕ) (initial_bars : ℕ) (bars_sold : ℕ) (bars_left : ℕ) : 
  chocolate_cost = 7 → 
  initial_bars = 15 → 
  bars_left = 4 → 
  bars_sold = initial_bars - bars_left → 
  (bars_sold * 7) = 77 := 
by 
  intros h1 h2 h3 h4 
  rw [h1, h2, h3, ← h4] 
  sorry

end emily_made_amount_l487_487276


namespace sum_reciprocals_factors_12_l487_487638

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487638


namespace vertical_distance_to_Charlie_l487_487247

-- Define points for Annie, Barbara, and Charlie
def point := (ℝ × ℝ)
def Annie : point := (10, -15)
def Barbara : point := (0, 20)
def Charlie : point := (5, 12)

-- Function to calculate the midpoint of two points
def midpoint (p1 p2 : point) : point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Midpoint between Annie and Barbara
def meeting_point : point := midpoint Annie Barbara

-- Vertical distance calculation
def vertical_distance (p1 p2 : point) : ℝ :=
  abs (p2.2 - p1.2)

theorem vertical_distance_to_Charlie :
  vertical_distance meeting_point Charlie = 9.5 :=
by
  sorry

end vertical_distance_to_Charlie_l487_487247


namespace max_intersections_l487_487580

-- Define the number of circles and lines
def num_circles : ℕ := 2
def num_lines : ℕ := 3

-- Define the maximum number of intersection points of circles
def max_circle_intersections : ℕ := 2

-- Define the number of intersection points between each line and each circle
def max_line_circle_intersections : ℕ := 2

-- Define the number of intersection points among lines (using the combination formula)
def num_line_intersections : ℕ := (num_lines.choose 2)

-- Define the greatest number of points of intersection
def total_intersections : ℕ :=
  max_circle_intersections + (num_lines * num_circles * max_line_circle_intersections) + num_line_intersections

-- Prove the greatest number of points of intersection is 17
theorem max_intersections : total_intersections = 17 := by
  -- Calculating individual parts for clarity
  have h1: max_circle_intersections = 2 := rfl
  have h2: num_lines * num_circles * max_line_circle_intersections = 12 := by
    calc
      num_lines * num_circles * max_line_circle_intersections
        = 3 * 2 * 2 := by rw [num_lines, num_circles, max_line_circle_intersections]
        ... = 12 := by norm_num
  have h3: num_line_intersections = 3 := by
    calc
      num_line_intersections = (3.choose 2) := rfl
      ... = 3 := by norm_num

  -- Adding the parts to get the total intersections
  calc
    total_intersections
      = max_circle_intersections + (num_lines * num_circles * max_line_circle_intersections) + num_line_intersections := rfl
      ... = 2 + 12 + 3 := by rw [h1, h2, h3]
      ... = 17 := by norm_num

end max_intersections_l487_487580


namespace sequence_a_n_l487_487355

variables (a : ℕ → ℕ) (b : ℕ → ℕ)

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def is_geometric_sequence (b : ℕ → ℕ) : Prop :=
  ∃ q, ∀ n, b (n + 1) = b n * q

def sum_first_n (f : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in finset.range n, f (i + 1)

axiom sum_of_product (a b : ℕ → ℕ) (n : ℕ) :
  sum_first_n (λ i, a i * b i) n = (2 * n + 1) * (3^n) - 1

theorem sequence_a_n (h1 : is_arithmetic_sequence a)
                     (h2 : is_geometric_sequence b)
                     (h3 : a 1 = 2)
                     (h4 : ∀ n, sum_first_n (λ i, a i * b i) n = (2 * n + 1) * (3^n) - 1) :
  ∀ n, a n = n + 1 :=
by sorry

end sequence_a_n_l487_487355


namespace proof_problem_l487_487384

variable {R : Type*} [LinearOrderedField R]

theorem proof_problem 
  (a1 a2 a3 b1 b2 b3 : R)
  (h1 : a1 < a2) (h2 : a2 < a3) (h3 : b1 < b2) (h4 : b2 < b3)
  (h_sum : a1 + a2 + a3 = b1 + b2 + b3)
  (h_pair_sum : a1 * a2 + a1 * a3 + a2 * a3 = b1 * b2 + b1 * b3 + b2 * b3)
  (h_a1_lt_b1 : a1 < b1) :
  (b2 < a2) ∧ (a3 < b3) ∧ (a1 * a2 * a3 < b1 * b2 * b3) ∧ ((1 - a1) * (1 - a2) * (1 - a3) > (1 - b1) * (1 - b2) * (1 - b3)) :=
by {
  sorry
}

end proof_problem_l487_487384


namespace sum_of_reciprocals_of_factors_of_12_l487_487794

theorem sum_of_reciprocals_of_factors_of_12 : 
  (∑ x in {1, 2, 3, 4, 6, 12}, (1 / (x : ℚ))) = 7 / 3 :=
by
  sorry

end sum_of_reciprocals_of_factors_of_12_l487_487794


namespace sum_reciprocal_factors_12_l487_487864

/-- The sum of the reciprocals of the natural-number factors of 12 equals 7/3. -/
theorem sum_reciprocal_factors_12 : 
  let factors := {1, 2, 3, 4, 6, 12} : Set ℕ
  let sum_reciprocals := ∑ n in factors, (1 / (n : ℚ))
  sum_reciprocals = (7 / 3) := by
  sorry

end sum_reciprocal_factors_12_l487_487864


namespace cycling_route_length_l487_487524

-- Conditions (segment lengths)
def segment1 : ℝ := 4
def segment2 : ℝ := 7
def segment3 : ℝ := 2
def segment4 : ℝ := 6
def segment5 : ℝ := 7

-- Specify the total length calculation
noncomputable def total_length : ℝ :=
  2 * (segment1 + segment2 + segment3) + 2 * (segment4 + segment5)

-- The theorem we want to prove
theorem cycling_route_length :
  total_length = 52 :=
by
  sorry

end cycling_route_length_l487_487524


namespace probability_of_product_divisible_by_4_l487_487162

-- Define a 6-sided die
def die := {1, 2, 3, 4, 5, 6}

-- Define the event that the product of 5 rolls is divisible by 4
def product_divisible_by_4 (rolls : Vector ℕ 5) : Prop :=
  (rolls.toList.prod % 4 = 0)

-- Calculate the probability of the event that the product is divisible by 4
def probability_divisible_by_4 : ℚ :=
  -- Here, we should compute the probability
  sorry

-- Statement of the theorem
theorem probability_of_product_divisible_by_4 :
  probability_divisible_by_4 = 11 / 12 :=
sorry

end probability_of_product_divisible_by_4_l487_487162


namespace sum_reciprocals_factors_12_l487_487649

theorem sum_reciprocals_factors_12 :
  (1:ℚ) + (1/2) + (1/3) + (1/4) + (1/6) + (1/12) = (7/3:ℚ) := 
by
  sorry

end sum_reciprocals_factors_12_l487_487649


namespace num_complementary_sets_correct_l487_487224

noncomputable def num_complementary_sets : ℕ := 13888

-- Shapes
inductive Shape | circle | square | triangle | star
-- Colors
inductive Color | red | blue | green | yellow
-- Shades
inductive Shade | light | medium | dark | very_dark

-- Define a card with shape, color, and shade
structure Card :=
  (shape : Shape)
  (color : Color)
  (shade : Shade)

-- Define conditions for complementary sets
def complementary (s : List Card) : Prop :=
  (∀ c ∈ s, c.shape = (s.head!).shape) ∨ (∀ (c₁ c₂ c₃ c₄ : Card),
    c₁ ∈ s → c₂ ∈ s → c₃ ∈ s → c₄ ∈ s →
    [c₁.shape, c₂.shape, c₃.shape, c₄.shape].nodup) ∧
  (∀ c ∈ s, c.color = (s.head!).color) ∨ (∀ (c₁ c₂ c₃ c₄ : Card),
    c₁ ∈ s → c₂ ∈ s → c₃ ∈ s → c₄ ∈ s →
    [c₁.color, c₂.color, c₃.color, c₄.color].nodup) ∧
  (∀ c ∈ s, c.shade = (s.head!).shade) ∨ (∀ (c₁ c₂ c₃ c₄ : Card),
    c₁ ∈ s → c₂ ∈ s → c₃ ∈ s → c₄ ∈ s →
    [c₁.shade, c₂.shade, c₃.shade, c₄.shade].nodup)

-- Theorem statement
theorem num_complementary_sets_correct : num_complementary_sets = 13888 :=
by sorry

end num_complementary_sets_correct_l487_487224


namespace arithmetic_sequence_ratio_l487_487351

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ (d : ℝ), ∀ n, a (n + 1) = a n + d

variable {a b : ℕ → ℝ}
variable {S T : ℕ → ℝ}

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(n + 1) * (a 0 + a n) / 2

variable (S_eq_k_mul_n_plus_2 : ∀ n, S n = (n + 2) * (S 0 / (n + 2)))
variable (T_eq_k_mul_n_plus_1 : ∀ n, T n = (n + 1) * (T 0 / (n + 1)))

theorem arithmetic_sequence_ratio (h₁ : arithmetic_sequence a) (h₂ : arithmetic_sequence b)
  (h₃ : ∀ n, S n = sum_first_n_terms a n)
  (h₄ : ∀ n, T n = sum_first_n_terms b n)
  (h₅ : ∀ n, (S n) / (T n) = (n + 2) / (n + 1))
  : a 6 / b 8 = 13 / 16 := 
sorry

end arithmetic_sequence_ratio_l487_487351
