import Mathlib

namespace smallest_possible_value_of_c_l2248_224867

/-- 
Given three integers \(a, b, c\) with \(a < b < c\), 
such that they form an arithmetic progression (AP) with the property that \(2b = a + c\), 
and form a geometric progression (GP) with the property that \(c^2 = ab\), 
prove that \(c = 2\) is the smallest possible value of \(c\).
-/
theorem smallest_possible_value_of_c :
  ∃ a b c : ℤ, a < b ∧ b < c ∧ 2 * b = a + c ∧ c^2 = a * b ∧ c = 2 :=
by
  sorry

end smallest_possible_value_of_c_l2248_224867


namespace find_angle_MON_l2248_224870

-- Definitions of conditions
variables {A B O C M N : Type} -- Points in a geometric space
variables (angle_AOB : ℝ) (ray_OC : Prop) (bisects_OM : Prop) (bisects_ON : Prop)
variables (angle_MOB : ℝ) (angle_MON : ℝ)

-- Conditions
-- Angle AOB is 90 degrees
def angle_AOB_90 (angle_AOB : ℝ) : Prop := angle_AOB = 90

-- OC is a ray (using a placeholder property for ray, as Lean may not have geometric entities)
def OC_is_ray (ray_OC : Prop) : Prop := ray_OC

-- OM bisects angle BOC
def OM_bisects_BOC (bisects_OM : Prop) : Prop := bisects_OM

-- ON bisects angle AOC
def ON_bisects_AOC (bisects_ON : Prop) : Prop := bisects_ON

-- The problem statement as a theorem in Lean
theorem find_angle_MON
  (h1 : angle_AOB_90 angle_AOB)
  (h2 : OC_is_ray ray_OC)
  (h3 : OM_bisects_BOC bisects_OM)
  (h4 : ON_bisects_AOC bisects_ON) :
  angle_MON = 45 ∨ angle_MON = 135 :=
sorry

end find_angle_MON_l2248_224870


namespace construct_1_degree_l2248_224863

def canConstruct1DegreeUsing19Degree : Prop :=
  ∃ (n : ℕ), n * 19 = 360 + 1

theorem construct_1_degree (h : ∃ (x : ℕ), x * 19 = 360 + 1) : canConstruct1DegreeUsing19Degree := by
  sorry

end construct_1_degree_l2248_224863


namespace inequality_proof_l2248_224872

-- Given conditions
variables {a b : ℝ} (ha_lt_b : a < b) (hb_lt_0 : b < 0)

-- Question statement we want to prove
theorem inequality_proof : ab < 0 → a < b → b < 0 → ab > b^2 :=
by
  sorry

end inequality_proof_l2248_224872


namespace initial_rope_length_l2248_224807

variable (R₀ R₁ R₂ R₃ : ℕ)
variable (h_cut1 : 2 * R₀ = R₁) -- Josh cuts the original rope in half
variable (h_cut2 : 2 * R₁ = R₂) -- He cuts one of the halves in half again
variable (h_cut3 : 5 * R₂ = R₃) -- He cuts one of the resulting pieces into fifths
variable (h_held_piece : R₃ = 5) -- The piece Josh is holding is 5 feet long

theorem initial_rope_length:
  R₀ = 100 :=
by
  sorry

end initial_rope_length_l2248_224807


namespace circles_are_externally_tangent_l2248_224838

noncomputable def circleA : Prop := ∀ x y : ℝ, x^2 + y^2 + 4 * x + 2 * y + 1 = 0
noncomputable def circleB : Prop := ∀ x y : ℝ, x^2 + y^2 - 2 * x - 6 * y + 1 = 0

theorem circles_are_externally_tangent (hA : circleA) (hB : circleB) : 
  ∃ P Q : ℝ, (P = 5) ∧ (Q = 5) := 
by 
  -- start proving with given conditions
  sorry

end circles_are_externally_tangent_l2248_224838


namespace verify_addition_by_subtraction_l2248_224855

theorem verify_addition_by_subtraction (a b c : ℤ) (h : a + b = c) : (c - a = b) ∧ (c - b = a) :=
by
  sorry

end verify_addition_by_subtraction_l2248_224855


namespace highest_point_difference_l2248_224831

theorem highest_point_difference :
  let A := -112
  let B := -80
  let C := -25
  max A (max B C) - min A (min B C) = 87 :=
by
  sorry

end highest_point_difference_l2248_224831


namespace tom_bought_8_kg_of_apples_l2248_224856

/-- 
   Given:
   - The cost of apples is 70 per kg.
   - 9 kg of mangoes at a rate of 55 per kg.
   - Tom paid a total of 1055.

   Prove that Tom purchased 8 kg of apples.
 -/
theorem tom_bought_8_kg_of_apples 
  (A : ℕ) 
  (h1 : 70 * A + 55 * 9 = 1055) : 
  A = 8 :=
sorry

end tom_bought_8_kg_of_apples_l2248_224856


namespace gem_stone_necklaces_sold_l2248_224897

theorem gem_stone_necklaces_sold (total_earned total_cost number_bead number_gem total_necklaces : ℕ) 
    (h1 : total_earned = 36) 
    (h2 : total_cost = 6) 
    (h3 : number_bead = 3) 
    (h4 : total_necklaces = total_earned / total_cost) 
    (h5 : total_necklaces = number_bead + number_gem) : 
    number_gem = 3 := 
sorry

end gem_stone_necklaces_sold_l2248_224897


namespace find_d_l2248_224859

theorem find_d (a b c d : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
    (h5 : a^2 = c * (d + 29)) (h6 : b^2 = c * (d - 29)) :
    d = 421 :=
    sorry

end find_d_l2248_224859


namespace no_valid_C_for_2C4_multiple_of_5_l2248_224895

theorem no_valid_C_for_2C4_multiple_of_5 :
  ¬ (∃ C : ℕ, C < 10 ∧ (2 * 100 + C * 10 + 4) % 5 = 0) :=
by
  sorry

end no_valid_C_for_2C4_multiple_of_5_l2248_224895


namespace xy_difference_l2248_224815

theorem xy_difference (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 10) (h3 : x = 15) : x - y = 10 :=
by
  sorry

end xy_difference_l2248_224815


namespace compare_abc_l2248_224813

noncomputable def a : ℝ := ∫ x in (0:ℝ)..1, x ^ (-1/3 : ℝ)
noncomputable def b : ℝ := 1 - ∫ x in (0:ℝ)..1, x ^ (1/2 : ℝ)
noncomputable def c : ℝ := ∫ x in (0:ℝ)..1, x ^ (3 : ℝ)

theorem compare_abc : a > b ∧ b > c := by
  sorry

end compare_abc_l2248_224813


namespace subscriptions_to_grandfather_l2248_224825

/-- 
Maggie earns $5.00 for every magazine subscription sold. 
She sold 4 subscriptions to her parents, 2 to the next-door neighbor, 
and twice that amount to another neighbor. Maggie earned $55 in total. 
Prove that the number of subscriptions Maggie sold to her grandfather is 1.
-/
theorem subscriptions_to_grandfather (G : ℕ) 
  (h1 : 5 * (4 + G + 2 + 4) = 55) : 
  G = 1 :=
by {
  sorry
}

end subscriptions_to_grandfather_l2248_224825


namespace range_of_m_l2248_224868

noncomputable def p (m : ℝ) : Prop :=
  (m^2 - 4 > 0) ∧ (m < 0)

noncomputable def q (m : ℝ) : Prop :=
  (16*(m-2)^2 - 16 < 0)

theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) → (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by
  intro h
  sorry

end range_of_m_l2248_224868


namespace trajectory_eq_l2248_224816

-- Define the points O, A, and B
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (-1, -2)

-- Define the vector equation for point C given the parameters s and t
def C (s t : ℝ) : ℝ × ℝ := (s * 2 + t * -1, s * 1 + t * -2)

-- Prove the equation of the trajectory of C given s + t = 1
theorem trajectory_eq (s t : ℝ) (h : s + t = 1) : ∃ x y : ℝ, C s t = (x, y) ∧ x - y - 1 = 0 := by
  -- The proof will be added here
  sorry

end trajectory_eq_l2248_224816


namespace sufficient_not_necessary_l2248_224842

variables {a : ℕ → ℝ} {S : ℕ → ℝ}
variables (h_seq : ∀ n, a (n + 1) = a n + (a 1 - a 0))
variables (h_sum : ∀ n, S n = n * (a 1 + a n) / 2)
variables (h_condition : 3 * a 2 = a 5 + 4)

theorem sufficient_not_necessary (h1 : a 1 < 1) : S 4 < 10 :=
sorry

end sufficient_not_necessary_l2248_224842


namespace quadrilateral_inequality_l2248_224844

-- Definitions based on conditions in a)
variables {A B C D : Type}
variables (AB AC AD BC CD : ℝ)
variable (angleA angleC: ℝ)
variable (convex := angleA + angleC < 180)

-- Lean statement that encodes the problem
theorem quadrilateral_inequality 
  (Hconvex : convex = true)
  : AB * CD + AD * BC < AC * (AB + AD) := 
sorry

end quadrilateral_inequality_l2248_224844


namespace journey_time_difference_journey_time_difference_in_minutes_l2248_224882

-- Define the constant speed of the bus
def speed : ℕ := 60

-- Define distances of journeys
def distance_1 : ℕ := 360
def distance_2 : ℕ := 420

-- Define the time calculation function
def time (d : ℕ) (s : ℕ) : ℕ := d / s

-- State the theorem
theorem journey_time_difference :
  time distance_2 speed - time distance_1 speed = 1 :=
by
  sorry

-- Convert the time difference from hours to minutes
theorem journey_time_difference_in_minutes :
  (time distance_2 speed - time distance_1 speed) * 60 = 60 :=
by
  sorry

end journey_time_difference_journey_time_difference_in_minutes_l2248_224882


namespace surveys_completed_total_l2248_224851

variable (regular_rate cellphone_rate total_earnings cellphone_surveys total_surveys : ℕ)
variable (h_regular_rate : regular_rate = 10)
variable (h_cellphone_rate : cellphone_rate = 13) -- 30% higher than regular_rate
variable (h_total_earnings : total_earnings = 1180)
variable (h_cellphone_surveys : cellphone_surveys = 60)
variable (h_total_surveys : total_surveys = cellphone_surveys + (total_earnings - (cellphone_surveys * cellphone_rate)) / regular_rate)

theorem surveys_completed_total :
  total_surveys = 100 :=
by
  sorry

end surveys_completed_total_l2248_224851


namespace perfect_square_trinomial_l2248_224864

theorem perfect_square_trinomial (a : ℝ) :
  (∃ m : ℝ, (x^2 + (a-1)*x + 9) = (x + m)^2) → (a = 7 ∨ a = -5) :=
by
  sorry

end perfect_square_trinomial_l2248_224864


namespace pos_integers_divisible_by_2_3_5_7_less_than_300_l2248_224879

theorem pos_integers_divisible_by_2_3_5_7_less_than_300 : 
  ∃ n : ℕ, n = 1 ∧ ∀ k : ℕ, k < 300 → 2 ∣ k ∧ 3 ∣ k ∧ 5 ∣ k ∧ 7 ∣ k → k = n * (210 : ℕ) :=
by
  sorry

end pos_integers_divisible_by_2_3_5_7_less_than_300_l2248_224879


namespace fraction_dad_roasted_l2248_224835

theorem fraction_dad_roasted :
  ∀ (dad_marshmallows joe_marshmallows joe_roast total_roast dad_roast : ℕ),
    dad_marshmallows = 21 →
    joe_marshmallows = 4 * dad_marshmallows →
    joe_roast = joe_marshmallows / 2 →
    total_roast = 49 →
    dad_roast = total_roast - joe_roast →
    (dad_roast : ℚ) / (dad_marshmallows : ℚ) = 1 / 3 :=
by
  intros dad_marshmallows joe_marshmallows joe_roast total_roast dad_roast
  intro h_dad_marshmallows
  intro h_joe_marshmallows
  intro h_joe_roast
  intro h_total_roast
  intro h_dad_roast
  sorry

end fraction_dad_roasted_l2248_224835


namespace maximum_value_of_f_l2248_224836

noncomputable def f (t : ℝ) : ℝ := ((3^t - 4 * t) * t) / (9^t)

theorem maximum_value_of_f : ∃ t : ℝ, f t = 1/16 :=
sorry

end maximum_value_of_f_l2248_224836


namespace ratio_of_rats_l2248_224804

theorem ratio_of_rats (x y : ℝ) (h : (0.56 * x) / (0.84 * y) = 1 / 2) : x / y = 3 / 4 :=
sorry

end ratio_of_rats_l2248_224804


namespace computation_problem_points_l2248_224824

/-- A teacher gives out a test of 30 problems. Each computation problem is worth some points, and
each word problem is worth 5 points. The total points you can receive on the test is 110 points,
and there are 20 computation problems. How many points is each computation problem worth? -/

theorem computation_problem_points (x : ℕ) (total_problems : ℕ := 30) (word_problem_points : ℕ := 5)
    (total_points : ℕ := 110) (computation_problems : ℕ := 20) :
    20 * x + (total_problems - computation_problems) * word_problem_points = total_points → x = 3 :=
by
  intro h
  sorry

end computation_problem_points_l2248_224824


namespace stationery_sales_other_l2248_224862

theorem stationery_sales_other (p e n : ℝ) (h_p : p = 25) (h_e : e = 30) (h_n : n = 20) :
    100 - (p + e + n) = 25 :=
by
  sorry

end stationery_sales_other_l2248_224862


namespace same_type_monomials_l2248_224896

theorem same_type_monomials (a b : ℤ) (h1 : 1 = a - 2) (h2 : b + 1 = 3) : (a - b) ^ 2023 = 1 := by
  sorry

end same_type_monomials_l2248_224896


namespace inequality_of_thirds_of_ordered_triples_l2248_224803

variable (a1 a2 a3 b1 b2 b3 : ℝ)

theorem inequality_of_thirds_of_ordered_triples 
  (h1 : a1 ≤ a2) 
  (h2 : a2 ≤ a3) 
  (h3 : b1 ≤ b2)
  (h4 : b2 ≤ b3)
  (h5 : a1 + a2 + a3 = b1 + b2 + b3)
  (h6 : a1 * a2 + a2 * a3 + a1 * a3 = b1 * b2 + b2 * b3 + b1 * b3)
  (h7 : a1 ≤ b1) : 
  a3 ≤ b3 := 
by 
  sorry

end inequality_of_thirds_of_ordered_triples_l2248_224803


namespace triangle_is_isosceles_or_right_l2248_224811

theorem triangle_is_isosceles_or_right (A B C a b : ℝ) (h : a * Real.cos (π - A) + b * Real.sin (π / 2 + B) = 0)
  (h1 : a = 2 * R * Real.sin A) 
  (h2 : b = 2 * R * Real.sin B) : 
  (A = B ∨ A + B = π / 2) := 
sorry

end triangle_is_isosceles_or_right_l2248_224811


namespace students_decrement_l2248_224834

theorem students_decrement:
  ∃ d : ℕ, ∃ A : ℕ, 
  (∃ n1 n2 n3 n4 n5 : ℕ, n1 = A ∧ n2 = A - d ∧ n3 = A - 2 * d ∧ n4 = A - 3 * d ∧ n5 = A - 4 * d) ∧
  (5 = 5) ∧
  (n1 + n2 + n3 + n4 + n5 = 115) ∧
  (A = 27) → d = 2 :=
by {
  sorry
}

end students_decrement_l2248_224834


namespace parabolas_intersect_at_single_point_l2248_224826

theorem parabolas_intersect_at_single_point (p q : ℝ) (h : -2 * p + q = 2023) :
  ∃ (x0 y0 : ℝ), (∀ p q : ℝ, y0 = x0^2 + p * x0 + q → -2 * p + q = 2023) ∧ x0 = -2 ∧ y0 = 2027 :=
by
  -- Proof to be filled in
  sorry

end parabolas_intersect_at_single_point_l2248_224826


namespace inverse_proposition_l2248_224823

theorem inverse_proposition (x : ℝ) : 
  (¬ (x > 2) → ¬ (x > 1)) ↔ ((x > 1) → (x > 2)) := 
by 
  sorry

end inverse_proposition_l2248_224823


namespace solution_set_inequality_l2248_224861

theorem solution_set_inequality : {x : ℝ | (x + 3) * (1 - x) ≥ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 1} :=
by
  sorry

end solution_set_inequality_l2248_224861


namespace max_value_fraction_sum_l2248_224845

theorem max_value_fraction_sum (a b c : ℝ) (h : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_sum : a + b + c = 3) :
  (ab / (a + b + 1) + ac / (a + c + 1) + bc / (b + c + 1) ≤ 3 / 2) :=
sorry

end max_value_fraction_sum_l2248_224845


namespace max_value_of_expression_l2248_224883

theorem max_value_of_expression (x y : ℝ) 
  (h : Real.sqrt (x * y) + Real.sqrt ((1 - x) * (1 - y)) = Real.sqrt (7 * x * (1 - y)) + (Real.sqrt (y * (1 - x)) / Real.sqrt 7)) :
  x + 7 * y ≤ 57 / 8 :=
sorry

end max_value_of_expression_l2248_224883


namespace number_of_federal_returns_sold_l2248_224886

/-- Given conditions for revenue calculations at the Kwik-e-Tax Center -/
structure TaxCenter where
  price_federal : ℕ
  price_state : ℕ
  price_quarterly : ℕ
  num_state : ℕ
  num_quarterly : ℕ
  total_revenue : ℕ

/-- The specific instance of the TaxCenter for this problem -/
def KwikETaxCenter : TaxCenter :=
{ price_federal := 50,
  price_state := 30,
  price_quarterly := 80,
  num_state := 20,
  num_quarterly := 10,
  total_revenue := 4400 }

/-- Proof statement for the number of federal returns sold -/
theorem number_of_federal_returns_sold (F : ℕ) :
  KwikETaxCenter.price_federal * F + 
  KwikETaxCenter.price_state * KwikETaxCenter.num_state + 
  KwikETaxCenter.price_quarterly * KwikETaxCenter.num_quarterly = 
  KwikETaxCenter.total_revenue → 
  F = 60 :=
by
  intro h
  /- Proof is skipped -/
  sorry

end number_of_federal_returns_sold_l2248_224886


namespace range_of_abs_function_l2248_224871

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 1)

theorem range_of_abs_function : Set.range f = Set.Ici 2 := by
  sorry

end range_of_abs_function_l2248_224871


namespace angle_B_in_triangle_l2248_224876

theorem angle_B_in_triangle (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = Real.pi / 5) :
  B = 3 * Real.pi / 10 :=
sorry

end angle_B_in_triangle_l2248_224876


namespace cost_of_childrens_ticket_l2248_224849

theorem cost_of_childrens_ticket (x : ℝ) 
  (h1 : ∀ A C : ℝ, A = 2 * C) 
  (h2 : 152 = 2 * 76)
  (h3 : ∀ A C : ℝ, 5.50 * A + x * C = 1026) 
  (h4 : 152 = 152) : 
  x = 2.50 :=
by
  sorry

end cost_of_childrens_ticket_l2248_224849


namespace silver_nitrate_mass_fraction_l2248_224839

variable (n : ℝ) (M : ℝ) (m_total : ℝ)
variable (m_agno3 : ℝ) (omega_agno3 : ℝ)

theorem silver_nitrate_mass_fraction 
  (h1 : n = 0.12) 
  (h2 : M = 170) 
  (h3 : m_total = 255)
  (h4 : m_agno3 = n * M) 
  (h5 : omega_agno3 = (m_agno3 * 100) / m_total) : 
  m_agno3 = 20.4 ∧ omega_agno3 = 8 :=
by
  -- insert proof here eventually 
  sorry

end silver_nitrate_mass_fraction_l2248_224839


namespace tan_alpha_sub_2pi_over_3_two_sin_sq_alpha_sub_cos_sq_alpha_l2248_224812

variable (α : ℝ)

theorem tan_alpha_sub_2pi_over_3 (h : Real.tan (α + π / 3) = 2 * Real.sqrt 3) :
    Real.tan (α - 2 * π / 3) = 2 * Real.sqrt 3 :=
sorry

theorem two_sin_sq_alpha_sub_cos_sq_alpha (h : Real.tan (α + π / 3) = 2 * Real.sqrt 3) :
    2 * (Real.sin α) ^ 2 - (Real.cos α) ^ 2 = -43 / 52 :=
sorry

end tan_alpha_sub_2pi_over_3_two_sin_sq_alpha_sub_cos_sq_alpha_l2248_224812


namespace glass_volume_230_l2248_224817

variable (V : ℝ) -- Define the total volume of the glass

-- Define the conditions
def pessimist_glass_volume (V : ℝ) := 0.40 * V
def optimist_glass_volume (V : ℝ) := 0.60 * V
def volume_difference (V : ℝ) := optimist_glass_volume V - pessimist_glass_volume V

theorem glass_volume_230 
  (h1 : volume_difference V = 46) : V = 230 := 
sorry

end glass_volume_230_l2248_224817


namespace perpendicular_slope_solution_l2248_224884

theorem perpendicular_slope_solution (a : ℝ) :
  (∀ x y : ℝ, ax + (3 - a) * y + 1 = 0) →
  (∀ x y : ℝ, x - 2 * y = 0) →
  (l1_perp_l2 : ∀ x y : ℝ, ax + (3 - a) * y + 1 = 0 → x - 2 * y = 0 → False) →
  a = 2 :=
sorry

end perpendicular_slope_solution_l2248_224884


namespace problem_B_false_l2248_224810

def diamondsuit (x y : ℝ) : ℝ := abs (x + y - 1)

theorem problem_B_false : ∀ x y : ℝ, 2 * (diamondsuit x y) ≠ diamondsuit (2 * x) (2 * y) :=
by
  intro x y
  dsimp [diamondsuit]
  sorry

end problem_B_false_l2248_224810


namespace cost_of_500_pencils_is_15_dollars_l2248_224829

-- Defining the given conditions
def cost_per_pencil_cents : ℕ := 3
def pencils_count : ℕ := 500
def cents_to_dollars : ℕ := 100

-- The proof problem: statement only, no proof provided
theorem cost_of_500_pencils_is_15_dollars :
  (cost_per_pencil_cents * pencils_count) / cents_to_dollars = 15 :=
by
  sorry

end cost_of_500_pencils_is_15_dollars_l2248_224829


namespace geo_seq_property_l2248_224808

theorem geo_seq_property (a : ℕ → ℤ) (r : ℤ) (h_geom : ∀ n, a (n+1) = r * a n)
  (h4_8 : a 4 + a 8 = -3) : a 6 * (a 2 + 2 * a 6 + a 10) = 9 := 
sorry

end geo_seq_property_l2248_224808


namespace expansion_coefficient_a2_l2248_224874

theorem expansion_coefficient_a2 (z x : ℂ) 
  (h : z = 1 + I) : 
  ∃ a_0 a_1 a_2 a_3 a_4 : ℂ,
    (z + x)^4 = a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0
    ∧ a_2 = 12 * I :=
by
  sorry

end expansion_coefficient_a2_l2248_224874


namespace smallest_n_l2248_224890

theorem smallest_n (n : ℕ) (h₁ : ∃ k₁ : ℕ, 5 * n = k₁ ^ 2) (h₂ : ∃ k₂ : ℕ, 4 * n = k₂ ^ 3) : n = 1600 :=
sorry

end smallest_n_l2248_224890


namespace largest_possible_three_day_success_ratio_l2248_224889

noncomputable def beta_max_success_ratio : ℝ :=
  let (a : ℕ) := 33
  let (b : ℕ) := 50
  let (c : ℕ) := 225
  let (d : ℕ) := 300
  let (e : ℕ) := 100
  let (f : ℕ) := 200
  a / b + c / d + e / f

theorem largest_possible_three_day_success_ratio :
  beta_max_success_ratio = (358 / 600 : ℝ) :=
by
  sorry

end largest_possible_three_day_success_ratio_l2248_224889


namespace percentage_of_male_students_solved_l2248_224857

variable (M F : ℝ)
variable (M_25 F_25 : ℝ)
variable (prob_less_25 : ℝ)

-- Conditions from the problem
def graduation_class_conditions (M F M_25 F_25 prob_less_25 : ℝ) : Prop :=
  M + F = 100 ∧
  M_25 = 0.50 * M ∧
  F_25 = 0.30 * F ∧
  (1 - 0.50) * M + (1 - 0.30) * F = prob_less_25 * 100

-- Theorem to prove
theorem percentage_of_male_students_solved (M F : ℝ) (M_25 F_25 prob_less_25 : ℝ) :
  graduation_class_conditions M F M_25 F_25 prob_less_25 → prob_less_25 = 0.62 → M = 40 :=
by
  sorry

end percentage_of_male_students_solved_l2248_224857


namespace emma_still_missing_fraction_l2248_224893

variable (x : ℕ)  -- Total number of coins Emma received 

-- Conditions
def emma_lost_half (x : ℕ) : ℕ := x / 2
def emma_found_four_fifths (lost : ℕ) : ℕ := 4 * lost / 5

-- Question to prove
theorem emma_still_missing_fraction :
  (x - (x / 2 + emma_found_four_fifths (emma_lost_half x))) / x = 1 / 10 := 
by
  sorry

end emma_still_missing_fraction_l2248_224893


namespace two_d_minus_c_zero_l2248_224801

theorem two_d_minus_c_zero :
  ∃ (c d : ℕ), (∀ x : ℕ, x^2 - 18 * x + 72 = (x - c) * (x - d)) ∧ c > d ∧ (2 * d - c = 0) := 
sorry

end two_d_minus_c_zero_l2248_224801


namespace factor_expression_l2248_224866

theorem factor_expression (a : ℝ) :
  (8 * a^3 + 105 * a^2 + 7) - (-9 * a^3 + 16 * a^2 - 14) = a^2 * (17 * a + 89) + 21 :=
by
  sorry

end factor_expression_l2248_224866


namespace quadratic_factors_l2248_224809

theorem quadratic_factors {a b c : ℝ} (h : a = 1) (h_roots : (1:ℝ) + 2 = b ∧ (-1:ℝ) * 2 = c) :
  (x^2 - b * x + c) = (x - 1) * (x - 2) := by
  sorry

end quadratic_factors_l2248_224809


namespace sum_of_first_4_terms_arithmetic_sequence_l2248_224843

variable {a : ℕ → ℝ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d a1, (∀ n, a n = a1 + n * d) ∧ (a 3 - a 1 = 2) ∧ (a 5 = 5)

-- Define the sum S4 for the first 4 terms of the sequence
def sum_first_4_terms (a : ℕ → ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3

-- Define the Lean statement for the problem
theorem sum_of_first_4_terms_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a → sum_first_4_terms a = 10 :=
by
  sorry

end sum_of_first_4_terms_arithmetic_sequence_l2248_224843


namespace determine_swimming_day_l2248_224805

def practices_sport_each_day (sports : ℕ → ℕ → Prop) : Prop :=
  ∀ (d : ℕ), ∃ s, sports d s

def runs_four_days_no_consecutive (sports : ℕ → ℕ → Prop) : Prop :=
  ∃ (days : ℕ → ℕ), (∀ i, sports (days i) 0) ∧ 
    (∀ i j, i ≠ j → days i ≠ days j) ∧ 
    (∀ i j, (days i + 1 = days j) → false)

def plays_basketball_tuesday (sports : ℕ → ℕ → Prop) : Prop :=
  sports 2 1

def plays_golf_friday_after_tuesday (sports : ℕ → ℕ → Prop) : Prop :=
  sports 5 2

def swims_and_plays_tennis_condition (sports : ℕ → ℕ → Prop) : Prop :=
  ∃ (swim_day tennis_day : ℕ), swim_day ≠ tennis_day ∧ 
    sports swim_day 3 ∧ 
    sports tennis_day 4 ∧ 
    ∀ (d : ℕ), (sports d 3 → sports (d + 1) 4 → false) ∧ 
    (∀ (d : ℕ), sports d 3 → ∀ (r : ℕ), sports (d + 2) 0 → false)

theorem determine_swimming_day (sports : ℕ → ℕ → Prop) : 
  practices_sport_each_day sports → 
  runs_four_days_no_consecutive sports → 
  plays_basketball_tuesday sports → 
  plays_golf_friday_after_tuesday sports → 
  swims_and_plays_tennis_condition sports → 
  ∃ (d : ℕ), d = 7 := 
sorry

end determine_swimming_day_l2248_224805


namespace find_value_less_than_twice_l2248_224848

def value_less_than_twice_another (x y v : ℕ) : Prop :=
  y = 2 * x - v ∧ x + y = 51 ∧ y = 33

theorem find_value_less_than_twice (x y v : ℕ) (h : value_less_than_twice_another x y v) : v = 3 := by
  sorry

end find_value_less_than_twice_l2248_224848


namespace least_value_b_l2248_224830

-- Defining the conditions of the problem
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

variables (a b c : ℕ)

-- Conditions
axiom angle_sum : a + b + c = 180
axiom primes : is_prime a ∧ is_prime b ∧ is_prime c
axiom order : a > b ∧ b > c

-- The statement to be proved
theorem least_value_b (h : a + b + c = 180) (hp : is_prime a ∧ is_prime b ∧ is_prime c) (ho : a > b ∧ b > c) : b = 5 :=
sorry

end least_value_b_l2248_224830


namespace ticket_cost_proof_l2248_224828

def adult_ticket_price : ℕ := 55
def child_ticket_price : ℕ := 28
def senior_ticket_price : ℕ := 42

def num_adult_tickets : ℕ := 4
def num_child_tickets : ℕ := 2
def num_senior_tickets : ℕ := 1

def total_ticket_cost : ℕ :=
  (num_adult_tickets * adult_ticket_price) + (num_child_tickets * child_ticket_price) + (num_senior_tickets * senior_ticket_price)

theorem ticket_cost_proof : total_ticket_cost = 318 := by
  sorry

end ticket_cost_proof_l2248_224828


namespace intersection_complement_R_M_and_N_l2248_224847

open Set

def universalSet := ℝ
def M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def complementR (S : Set ℝ) := {x : ℝ | x ∉ S}
def N := {x : ℝ | x < 1}

theorem intersection_complement_R_M_and_N:
  (complementR M ∩ N) = {x : ℝ | x < -2} := by
  sorry

end intersection_complement_R_M_and_N_l2248_224847


namespace solve_equation_l2248_224832

theorem solve_equation : ∃! x : ℕ, 3^x = x + 2 := by
  sorry

end solve_equation_l2248_224832


namespace arithmetic_sequence_a9_l2248_224819

theorem arithmetic_sequence_a9 (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = n * (2 * a 0 + (n - 1))) →
  S 6 = 3 * S 3 →
  a 9 = 10 := by
  sorry

end arithmetic_sequence_a9_l2248_224819


namespace square_diff_theorem_l2248_224840

theorem square_diff_theorem
  (a b c p x : ℝ)
  (h1 : a + b + c = 2 * p)
  (h2 : x = (b^2 + c^2 - a^2) / (2 * c))
  (h3 : c ≠ 0) :
  b^2 - x^2 = 4 / c^2 * (p * (p - a) * (p - b) * (p - c)) := by
  sorry

end square_diff_theorem_l2248_224840


namespace arrangement_ways_count_l2248_224894

theorem arrangement_ways_count:
  let n := 10
  let k := 4
  (Nat.choose n k) = 210 :=
by
  sorry

end arrangement_ways_count_l2248_224894


namespace radius_of_spheres_in_cone_l2248_224846

theorem radius_of_spheres_in_cone :
  ∀ (r : ℝ),
    let base_radius := 6
    let height := 15
    let distance_from_vertex := (2 * Real.sqrt 3 / 3) * r
    let total_height := height - r
    (total_height = distance_from_vertex) →
    r = 27 - 6 * Real.sqrt 3 :=
by
  intros r base_radius height distance_from_vertex total_height H
  sorry -- The proof of the theorem will be filled here.

end radius_of_spheres_in_cone_l2248_224846


namespace solve_x_given_y_l2248_224802

theorem solve_x_given_y (x : ℝ) (h : 2 = 2 / (5 * x + 3)) : x = -2 / 5 :=
sorry

end solve_x_given_y_l2248_224802


namespace incorrect_observation_value_l2248_224821

theorem incorrect_observation_value
  (mean : ℕ → ℝ)
  (n : ℕ)
  (observed_mean : ℝ)
  (incorrect_value : ℝ)
  (correct_value : ℝ)
  (corrected_mean : ℝ)
  (H1 : n = 50)
  (H2 : observed_mean = 36)
  (H3 : correct_value = 43)
  (H4 : corrected_mean = 36.5)
  (H5 : mean n = observed_mean)
  (H6 : mean (n - 1 + 1) = corrected_mean - correct_value + incorrect_value) :
  incorrect_value = 18 := sorry

end incorrect_observation_value_l2248_224821


namespace parallel_lines_l2248_224875

theorem parallel_lines (a : ℝ) : (∀ x y : ℝ, 2 * x + a * y + 1 = 0 ↔ x - 4 * y - 1 = 0) → a = -8 :=
by
  intro h -- Introduce the hypothesis that lines are parallel
  sorry -- Skip the proof

end parallel_lines_l2248_224875


namespace large_block_volume_correct_l2248_224877

def normal_block_volume (w d l : ℝ) : ℝ := w * d * l

def large_block_volume (w d l : ℝ) : ℝ := (2 * w) * (2 * d) * (3 * l)

theorem large_block_volume_correct (w d l : ℝ) (h : normal_block_volume w d l = 3) :
  large_block_volume w d l = 36 :=
by sorry

end large_block_volume_correct_l2248_224877


namespace total_fruits_l2248_224899

theorem total_fruits (total_baskets apples_baskets oranges_baskets apples_per_basket oranges_per_basket pears_per_basket : ℕ)
  (h1 : total_baskets = 127)
  (h2 : apples_baskets = 79)
  (h3 : oranges_baskets = 30)
  (h4 : apples_per_basket = 75)
  (h5 : oranges_per_basket = 143)
  (h6 : pears_per_basket = 56)
  : 79 * 75 + 30 * 143 + (127 - (79 + 30)) * 56 = 11223 := by
  sorry

end total_fruits_l2248_224899


namespace add_gold_coins_l2248_224898

open Nat

theorem add_gold_coins (G S X : ℕ) 
  (h₁ : G = S / 3) 
  (h₂ : (G + X) / S = 1 / 2) 
  (h₃ : G + X + S = 135) : 
  X = 15 := 
sorry

end add_gold_coins_l2248_224898


namespace minimum_strips_cover_circle_l2248_224853

theorem minimum_strips_cover_circle (l R : ℝ) (hl : l > 0) (hR : R > 0) :
  ∃ (k : ℕ), (k : ℝ) * l ≥ 2 * R ∧ ((k - 1 : ℕ) : ℝ) * l < 2 * R :=
sorry

end minimum_strips_cover_circle_l2248_224853


namespace sum_of_powers_mod_7_l2248_224822

theorem sum_of_powers_mod_7 :
  ((1^1 + 2^2 + 3^3 + 4^4 + 5^5 + 6^6 + 7^7) % 7 = 1) := by
  sorry

end sum_of_powers_mod_7_l2248_224822


namespace rotated_parabola_equation_l2248_224800

def parabola_equation (x y : ℝ) : Prop := y = x^2 - 4 * x + 3

def standard_form (x y : ℝ) : Prop := y = (x - 2)^2 - 1

def after_rotation (x y : ℝ) : Prop := (y + 1)^2 = x - 2

theorem rotated_parabola_equation (x y : ℝ) (h : standard_form x y) : after_rotation x y :=
sorry

end rotated_parabola_equation_l2248_224800


namespace josh_total_money_l2248_224827

-- Define the initial conditions
def initial_wallet : ℝ := 300
def initial_investment : ℝ := 2000
def stock_increase_rate : ℝ := 0.30

-- The expected total amount Josh will have after selling his stocks
def expected_total_amount : ℝ := 2900

-- Define the problem: that the total money in Josh's wallet after selling all stocks equals $2900
theorem josh_total_money :
  let increased_value := initial_investment * stock_increase_rate
  let new_investment := initial_investment + increased_value
  let total_money := new_investment + initial_wallet
  total_money = expected_total_amount :=
by
  sorry

end josh_total_money_l2248_224827


namespace mean_computation_l2248_224885

theorem mean_computation (x y : ℝ) 
  (h1 : (28 + x + 70 + 88 + 104) / 5 = 67)
  (h2 : (if x < 50 ∧ x < 62 then if y < 62 then ((28 + y) / 2 = 81) else ((62 + x) / 2 = 81) else if y < 50 then ((y + 50) / 2 = 81) else if y < 62 then ((50 + y) / 2 = 81) else ((50 + x) / 2 = 81)) -- conditions for median can be simplified and expanded as necessary
) : (50 + 62 + 97 + 124 + x + y) / 6 = 82.5 :=
sorry

end mean_computation_l2248_224885


namespace calc_value_exponents_l2248_224806

theorem calc_value_exponents :
  (3^3) * (5^3) * (3^5) * (5^5) = 15^8 :=
by sorry

end calc_value_exponents_l2248_224806


namespace angle_F_after_decrease_l2248_224869

theorem angle_F_after_decrease (D E F : ℝ) (h1 : D = 60) (h2 : E = 60) (h3 : F = 60) (h4 : E = D) :
  F - 20 = 40 := by
  simp [h3]
  sorry

end angle_F_after_decrease_l2248_224869


namespace expected_prize_money_l2248_224860

theorem expected_prize_money :
  let a1 := 1 / 7
  let prob1 := a1
  let prob2 := 2 * a1
  let prob3 := 4 * a1
  let prize1 := 700
  let prize2 := 700 - 140
  let prize3 := 700 - 140 * 2
  let expected_money := prize1 * prob1 + prize2 * prob2 + prize3 * prob3
  expected_money = 500 := 
by
  -- Definitions
  let a1 := 1 / 7
  let prob1 := a1
  let prob2 := 2 * a1
  let prob3 := 4 * a1
  let prize1 := 700
  let prize2 := 700 - 140
  let prize3 := 700 - 140 * 2
  let expected_money := prize1 * prob1 + prize2 * prob2 + prize3 * prob3

  -- Calculate
  sorry -- Proof to show expected_money equals 500

end expected_prize_money_l2248_224860


namespace large_rect_area_is_294_l2248_224833

-- Define the dimensions of the smaller rectangles
def shorter_side : ℕ := 7
def longer_side : ℕ := 2 * shorter_side

-- Condition 1: Each smaller rectangle has a shorter side measuring 7 feet
axiom smaller_rect_shorter_side : ∀ (r : ℕ), r = shorter_side → r = 7

-- Condition 4: The longer side of each smaller rectangle is twice the shorter side
axiom smaller_rect_longer_side : ∀ (r : ℕ), r = longer_side → r = 2 * shorter_side

-- Condition 2: Three rectangles are aligned vertically
def vertical_height : ℕ := 3 * shorter_side

-- Condition 3: One rectangle is aligned horizontally adjoining them
def horizontal_length : ℕ := longer_side

-- The dimensions of the larger rectangle EFGH
def large_rect_width : ℕ := vertical_height
def large_rect_length : ℕ := horizontal_length

-- Calculate the area of the larger rectangle EFGH
def large_rect_area : ℕ := large_rect_width * large_rect_length

-- Prove that the area of the large rectangle is 294 square feet
theorem large_rect_area_is_294 : large_rect_area = 294 := by
  sorry

end large_rect_area_is_294_l2248_224833


namespace max_lamps_on_road_l2248_224837

theorem max_lamps_on_road (k: ℕ) (lk: ℕ): 
  lk = 1000 → (∀ n: ℕ, n < k → n≥ 1 ∧ ∀ m: ℕ, if m > n then m > 1 else true) → (lk ≤ k) ∧ 
  (∀ i:ℕ,∃ j, (i ≠ j) → (lk < 1000)) → k = 1998 :=
by sorry

end max_lamps_on_road_l2248_224837


namespace perimeter_of_floor_l2248_224858

-- Define the side length of the room's floor
def side_length : ℕ := 5

-- Define the formula for the perimeter of a square
def perimeter_of_square (side : ℕ) : ℕ := 4 * side

-- State the theorem: the perimeter of the floor of the room is 20 meters
theorem perimeter_of_floor : perimeter_of_square side_length = 20 :=
by
  sorry

end perimeter_of_floor_l2248_224858


namespace oranges_to_juice_l2248_224891

theorem oranges_to_juice (oranges: ℕ) (juice: ℕ) (h: oranges = 18 ∧ juice = 27): 
  ∃ x, (juice / oranges) = (9 / x) ∧ x = 6 :=
by
  sorry

end oranges_to_juice_l2248_224891


namespace smallest_sum_l2248_224820

theorem smallest_sum (r s t : ℕ) (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_pos_t : 0 < t) 
  (h_prod : r * s * t = 1230) : r + s + t = 52 :=
sorry

end smallest_sum_l2248_224820


namespace find_remainder_l2248_224850

-- Definitions based on given conditions
def dividend := 167
def divisor := 18
def quotient := 9

-- Statement to prove
theorem find_remainder : dividend = (divisor * quotient) + 5 :=
by
  -- Definitions used in the problem
  unfold dividend divisor quotient
  sorry

end find_remainder_l2248_224850


namespace find_principal_amount_l2248_224888

variable (P : ℝ)

def interestA_to_B (P : ℝ) : ℝ := P * 0.10 * 3
def interestB_from_C (P : ℝ) : ℝ := P * 0.115 * 3
def gain_B (P : ℝ) : ℝ := interestB_from_C P - interestA_to_B P

theorem find_principal_amount (h : gain_B P = 45) : P = 1000 := by
  sorry

end find_principal_amount_l2248_224888


namespace son_age_l2248_224865

theorem son_age (S F : ℕ) (h1 : F = S + 30) (h2 : F + 2 = 2 * (S + 2)) : S = 28 :=
by
  sorry

end son_age_l2248_224865


namespace real_to_fraction_l2248_224878

noncomputable def real_num : ℚ := 3.675

theorem real_to_fraction : real_num = 147 / 40 :=
by
  -- convert 3.675 to a mixed number
  have h1 : real_num = 3 + 675 / 1000 := by sorry
  -- find gcd of 675 and 1000
  have h2 : Nat.gcd 675 1000 = 25 := by sorry
  -- simplify 675/1000 to 27/40
  have h3 : 675 / 1000 = 27 / 40 := by sorry
  -- convert mixed number to improper fraction 147/40
  have h4 : 3 + 27 / 40 = 147 / 40 := by sorry
  -- combine the results to prove the required equality
  exact sorry

end real_to_fraction_l2248_224878


namespace range_of_m_l2248_224873

noncomputable def p (m : ℝ) : Prop :=
  (m > 2)

noncomputable def q (m : ℝ) : Prop :=
  (m > 1)

theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) → (1 < m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l2248_224873


namespace problem1_proof_problem2_proof_l2248_224892

section Problems

variable {x a : ℝ}

-- Problem 1
theorem problem1_proof : 3 * x^2 * x^4 - (-x^3)^2 = 2 * x^6 := by
  sorry

-- Problem 2
theorem problem2_proof : a^3 * a + (-a^2)^3 / a^2 = 0 := by
  sorry

end Problems

end problem1_proof_problem2_proof_l2248_224892


namespace range_of_x_for_function_l2248_224887

theorem range_of_x_for_function :
  ∀ x : ℝ, (2 - x ≥ 0 ∧ x - 1 ≠ 0) ↔ (x ≤ 2 ∧ x ≠ 1) := by
  sorry

end range_of_x_for_function_l2248_224887


namespace outlets_per_room_l2248_224880

theorem outlets_per_room
  (rooms : ℕ)
  (total_outlets : ℕ)
  (h1 : rooms = 7)
  (h2 : total_outlets = 42) :
  total_outlets / rooms = 6 :=
by sorry

end outlets_per_room_l2248_224880


namespace cost_of_toys_target_weekly_price_l2248_224852

-- First proof problem: Cost of Plush Toy and Metal Ornament
theorem cost_of_toys (x : ℝ) (hx : 6400 / x = 2 * (4000 / (x + 20))) : 
  x = 80 :=
by sorry

-- Second proof problem: Price to achieve target weekly profit
theorem target_weekly_price (y : ℝ) (hy : (y - 80) * (10 + (150 - y) / 5) = 720) :
  y = 140 :=
by sorry

end cost_of_toys_target_weekly_price_l2248_224852


namespace imaginary_part_of_complex_number_l2248_224881

def imaginary_unit (i : ℂ) : Prop := i * i = -1

def complex_number (z : ℂ) (i : ℂ) : Prop := z = i * (1 - 3 * i)

theorem imaginary_part_of_complex_number (i z : ℂ) (h1 : imaginary_unit i) (h2 : complex_number z i) : z.im = 1 :=
by
  sorry

end imaginary_part_of_complex_number_l2248_224881


namespace binom_identity1_binom_identity2_l2248_224814

variable (n k : ℕ)

theorem binom_identity1 (hn : n > 0) (hk : k > 0) :
  (Nat.choose n k) + (Nat.choose n (k + 1)) = (Nat.choose (n + 1) (k + 1)) :=
sorry

theorem binom_identity2 (hn : n > 0) (hk : k > 0) :
  (Nat.choose n k) = (n * Nat.choose (n - 1) (k - 1)) / k :=
sorry

end binom_identity1_binom_identity2_l2248_224814


namespace multiples_of_7_units_digit_7_l2248_224818

theorem multiples_of_7_units_digit_7 (n : ℕ) (h1 : n < 150) (h2 : ∃ (k : ℕ), n = 7 * k) (h3 : n % 10 = 7) : 
    ∃ (m : ℕ), m = 2 := 
by
  sorry

end multiples_of_7_units_digit_7_l2248_224818


namespace evaluate_expression_at_neg_two_l2248_224854

noncomputable def complex_expression (a : ℝ) : ℝ :=
  (1 - (a / (a + 1))) / (1 / (1 - a^2))

theorem evaluate_expression_at_neg_two :
  complex_expression (-2) = sorry :=
sorry

end evaluate_expression_at_neg_two_l2248_224854


namespace iggy_total_time_correct_l2248_224841

noncomputable def total_time_iggy_spends : ℕ :=
  let monday_time := 3 * (10 + 1)
  let tuesday_time := 4 * (9 + 1)
  let wednesday_time := 6 * 12
  let thursday_time := 8 * (8 + 2)
  let friday_time := 3 * 10
  monday_time + tuesday_time + wednesday_time + thursday_time + friday_time

theorem iggy_total_time_correct : total_time_iggy_spends = 255 :=
by
  -- sorry at the end indicates the skipping of the actual proof elaboration.
  sorry

end iggy_total_time_correct_l2248_224841
