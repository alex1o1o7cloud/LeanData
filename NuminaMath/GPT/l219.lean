import Mathlib

namespace find_point_B_l219_21925

noncomputable def vector_a : ℝ × ℝ := (1, 1)
noncomputable def point_A : ℝ × ℝ := (-3, -1)
def line_y_eq_2x (x : ℝ) : ℝ × ℝ := (x, 2 * x)
def is_parallel (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1 

theorem find_point_B (B : ℝ × ℝ) (hB : B = line_y_eq_2x B.1) (h_parallel : is_parallel (B.1 + 3, B.2 + 1) vector_a) :
  B = (2, 4) := 
  sorry

end find_point_B_l219_21925


namespace polynomial_abs_sum_l219_21977

theorem polynomial_abs_sum (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) :
  (1 - (2:ℝ) * x) ^ 8 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 →
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| + |a_8| = (3:ℝ) ^ 8 :=
sorry

end polynomial_abs_sum_l219_21977


namespace trigonometric_identity_l219_21914

theorem trigonometric_identity
  (θ : ℝ)
  (h : (2 + (1 / (Real.sin θ) ^ 2)) / (1 + Real.sin θ) = 1) :
  (1 + Real.sin θ) * (2 + Real.cos θ) = 4 :=
sorry

end trigonometric_identity_l219_21914


namespace factor_expression_l219_21945

theorem factor_expression (x : ℝ) : (45 * x^3 - 135 * x^7) = 45 * x^3 * (1 - 3 * x^4) :=
by
  sorry

end factor_expression_l219_21945


namespace DeansCalculatorGame_l219_21933

theorem DeansCalculatorGame (r : ℕ) (c1 c2 c3 : ℤ) (h1 : r = 45) (h2 : c1 = 1) (h3 : c2 = 0) (h4 : c3 = -2) : 
  let final1 := (c1 ^ 3)
  let final2 := (c2 ^ 2)
  let final3 := (-c3)^45
  final1 + final2 + final3 = 3 := 
by
  sorry

end DeansCalculatorGame_l219_21933


namespace number_of_5_digit_numbers_l219_21924

/-- There are 324 five-digit numbers starting with 2 that have exactly three identical digits which are not 2. -/
theorem number_of_5_digit_numbers : ∃ n : ℕ, n = 324 ∧ ∀ (d₁ d₂ : ℕ), 
  (d₁ ≠ 2) ∧ (d₁ ≠ d₂) ∧ (0 ≤ d₁ ∧ d₁ < 10) ∧ (0 ≤ d₂ ∧ d₂ < 10) → 
  n = 4 * 9 * 9 := by
  sorry

end number_of_5_digit_numbers_l219_21924


namespace range_of_m_l219_21908

noncomputable def isEllipse (m : ℝ) : Prop := (m^2 > 2 * m + 8) ∧ (2 * m + 8 > 0)
noncomputable def intersectsXAxisAtTwoPoints (m : ℝ) : Prop := (2 * m - 3)^2 - 1 > 0

theorem range_of_m (m : ℝ) :
  ((m^2 > 2 * m + 8 ∧ 2 * m + 8 > 0 ∨ (2 * m - 3)^2 - 1 > 0) ∧
  ¬ (m^2 > 2 * m + 8 ∧ 2 * m + 8 > 0 ∧ (2 * m - 3)^2 - 1 > 0)) →
  (m ≤ -4 ∨ (-2 ≤ m ∧ m < 1) ∨ (2 < m ∧ m ≤ 4)) :=
by sorry

end range_of_m_l219_21908


namespace largest_integer_solving_inequality_l219_21981

theorem largest_integer_solving_inequality :
  ∃ (x : ℤ), (7 - 5 * x > 22) ∧ ∀ (y : ℤ), (7 - 5 * y > 22) → x ≥ y ∧ x = -4 :=
by
  sorry

end largest_integer_solving_inequality_l219_21981


namespace loss_percentage_is_10_l219_21973

-- Define the conditions
def cost_price (CP : ℝ) : Prop :=
  (550 : ℝ) = 1.1 * CP

def selling_price (SP : ℝ) : Prop :=
  SP = 450

-- Define the main proof statement
theorem loss_percentage_is_10 (CP SP : ℝ) (HCP : cost_price CP) (HSP : selling_price SP) :
  ((CP - SP) / CP) * 100 = 10 :=
by
  -- Translation of the condition into Lean statement
  sorry

end loss_percentage_is_10_l219_21973


namespace solve_for_y_l219_21980

theorem solve_for_y (x y : ℝ) (h : 2 * x - 3 * y = 4) : y = (2 * x - 4) / 3 :=
sorry

end solve_for_y_l219_21980


namespace scientific_notation_of_50000_l219_21986

theorem scientific_notation_of_50000 :
  50000 = 5 * 10^4 :=
sorry

end scientific_notation_of_50000_l219_21986


namespace range_of_a_l219_21968

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a + Real.cos (2 * x) < 5 - 4 * Real.sin x + Real.sqrt (5 * a - 4)) :
  a ∈ Set.Icc (4 / 5) 8 :=
sorry

end range_of_a_l219_21968


namespace find_x_l219_21998

theorem find_x (x : ℝ) (h : x * 1.6 - (2 * 1.4) / 1.3 = 4) : x = 3.846154 :=
sorry

end find_x_l219_21998


namespace total_chickens_l219_21926

-- Definitions from conditions
def ducks : ℕ := 40
def rabbits : ℕ := 30
def hens : ℕ := ducks + 20
def roosters : ℕ := rabbits - 10

-- Theorem statement: total number of chickens
theorem total_chickens : hens + roosters = 80 := 
sorry

end total_chickens_l219_21926


namespace min_value_of_b_minus_2c_plus_1_over_a_l219_21984

theorem min_value_of_b_minus_2c_plus_1_over_a
  (a b c : ℝ)
  (h₁ : (a ≠ 0))
  (h₂ : ∀ x, -1 < x ∧ x < 3 → ax^2 + bx + c < 0) :
  b - 2 * c + (1 / a) = 4 :=
sorry

end min_value_of_b_minus_2c_plus_1_over_a_l219_21984


namespace less_sum_mult_l219_21971

theorem less_sum_mult {a b : ℝ} (h1 : a < 1) (h2 : b > 1) : a * b < a + b :=
sorry

end less_sum_mult_l219_21971


namespace smallest_n_exceeds_15_l219_21956

noncomputable def g (n : ℕ) : ℕ :=
  sorry  -- Define the sum of the digits of 1 / 3^n to the right of the decimal point

theorem smallest_n_exceeds_15 : ∃ n : ℕ, n > 0 ∧ g n > 15 ∧ ∀ m : ℕ, m > 0 ∧ g m > 15 → n ≤ m :=
  sorry  -- Prove the smallest n such that g(n) > 15

end smallest_n_exceeds_15_l219_21956


namespace correct_quotient_is_243_l219_21999

-- Define the given conditions
def mistaken_divisor : ℕ := 121
def mistaken_quotient : ℕ := 432
def correct_divisor : ℕ := 215
def remainder : ℕ := 0

-- Calculate the dividend based on mistaken values
def dividend : ℕ := mistaken_divisor * mistaken_quotient + remainder

-- State the theorem for the correct quotient
theorem correct_quotient_is_243
  (h_dividend : dividend = mistaken_divisor * mistaken_quotient + remainder)
  (h_divisible : dividend % correct_divisor = remainder) :
  dividend / correct_divisor = 243 :=
sorry

end correct_quotient_is_243_l219_21999


namespace football_team_goal_l219_21946

-- Definitions of the conditions
def L1 : ℤ := -5
def G2 : ℤ := 13
def L3 : ℤ := -(L1 ^ 2)
def G4 : ℚ := - (L3 : ℚ) / 2

def total_yardage : ℚ := L1 + G2 + L3 + G4

-- The statement to be proved
theorem football_team_goal : total_yardage < 30 := by
  -- sorry for now since no proof is needed
  sorry

end football_team_goal_l219_21946


namespace solve_quadratic_eq_l219_21987

theorem solve_quadratic_eq (x : ℝ) : x^2 = 4 * x → x = 0 ∨ x = 4 :=
by
  intro h
  sorry

end solve_quadratic_eq_l219_21987


namespace sum_super_cool_rectangle_areas_eq_84_l219_21915

theorem sum_super_cool_rectangle_areas_eq_84 :
  ∀ (a b : ℕ), 
  (a * b = 3 * (a + b)) → 
  ∃ (S : ℕ), 
  S = 84 :=
by
  sorry

end sum_super_cool_rectangle_areas_eq_84_l219_21915


namespace matthew_and_zac_strawberries_l219_21995

theorem matthew_and_zac_strawberries (total_strawberries jonathan_and_matthew_strawberries zac_strawberries : ℕ) (h1 : total_strawberries = 550) (h2 : jonathan_and_matthew_strawberries = 350) (h3 : zac_strawberries = 200) : (total_strawberries - (jonathan_and_matthew_strawberries - zac_strawberries) = 400) :=
by { sorry }

end matthew_and_zac_strawberries_l219_21995


namespace find_length_of_DE_l219_21903

-- Define the setup: five points A, B, C, D, E on a circle
variables (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]

-- Define the given distances 
def AB : ℝ := 7
def BC : ℝ := 7
def AD : ℝ := 10

-- Define the total distance AC
def AC : ℝ := AB + BC

-- Define the length DE to be solved
def DE : ℝ := 0.2

-- State the theorem to be proved given the conditions
theorem find_length_of_DE : 
  DE = 0.2 :=
sorry

end find_length_of_DE_l219_21903


namespace puppy_ratios_l219_21947

theorem puppy_ratios :
  ∀(total_puppies : ℕ)(golden_retriever_females golden_retriever_males : ℕ)
   (labrador_females labrador_males : ℕ)(poodle_females poodle_males : ℕ)
   (beagle_females beagle_males : ℕ),
  total_puppies = golden_retriever_females + golden_retriever_males +
                  labrador_females + labrador_males +
                  poodle_females + poodle_males +
                  beagle_females + beagle_males →
  golden_retriever_females = 2 →
  golden_retriever_males = 4 →
  labrador_females = 1 →
  labrador_males = 3 →
  poodle_females = 3 →
  poodle_males = 2 →
  beagle_females = 1 →
  beagle_males = 2 →
  (golden_retriever_females / golden_retriever_males = 1 / 2) ∧
  (labrador_females / labrador_males = 1 / 3) ∧
  (poodle_females / poodle_males = 3 / 2) ∧
  (beagle_females / beagle_males = 1 / 2) ∧
  (7 / 11 = (golden_retriever_females + labrador_females + poodle_females + beagle_females) / 
            (golden_retriever_males + labrador_males + poodle_males + beagle_males)) :=
by intros;
   sorry

end puppy_ratios_l219_21947


namespace remaining_glazed_correct_remaining_chocolate_correct_remaining_raspberry_correct_l219_21900

section Doughnuts

variable (initial_glazed : Nat := 10)
variable (initial_chocolate : Nat := 8)
variable (initial_raspberry : Nat := 6)

variable (personA_glazed : Nat := 2)
variable (personA_chocolate : Nat := 1)
variable (personB_glazed : Nat := 1)
variable (personC_chocolate : Nat := 3)
variable (personD_glazed : Nat := 1)
variable (personD_raspberry : Nat := 1)
variable (personE_raspberry : Nat := 1)
variable (personF_raspberry : Nat := 2)

def remaining_glazed : Nat :=
  initial_glazed - (personA_glazed + personB_glazed + personD_glazed)

def remaining_chocolate : Nat :=
  initial_chocolate - (personA_chocolate + personC_chocolate)

def remaining_raspberry : Nat :=
  initial_raspberry - (personD_raspberry + personE_raspberry + personF_raspberry)

theorem remaining_glazed_correct :
  remaining_glazed initial_glazed personA_glazed personB_glazed personD_glazed = 6 :=
by
  sorry

theorem remaining_chocolate_correct :
  remaining_chocolate initial_chocolate personA_chocolate personC_chocolate = 4 :=
by
  sorry

theorem remaining_raspberry_correct :
  remaining_raspberry initial_raspberry personD_raspberry personE_raspberry personF_raspberry = 2 :=
by
  sorry

end Doughnuts

end remaining_glazed_correct_remaining_chocolate_correct_remaining_raspberry_correct_l219_21900


namespace largest_value_n_under_100000_l219_21912

theorem largest_value_n_under_100000 :
  ∃ n : ℕ,
    0 ≤ n ∧
    n < 100000 ∧
    (10 * (n - 3)^5 - n^2 + 20 * n - 30) % 7 = 0 ∧
    n = 99999 :=
sorry

end largest_value_n_under_100000_l219_21912


namespace crafts_club_necklaces_l219_21960

theorem crafts_club_necklaces (members : ℕ) (total_beads : ℕ) (beads_per_necklace : ℕ)
  (h1 : members = 9) (h2 : total_beads = 900) (h3 : beads_per_necklace = 50) :
  (total_beads / beads_per_necklace) / members = 2 :=
by
  sorry

end crafts_club_necklaces_l219_21960


namespace find_t_l219_21974

theorem find_t (t a b : ℝ) :
  (∀ x : ℝ, (3 * x^2 - 4 * x + 5) * (5 * x^2 + t * x + 12) = 15 * x^4 - 47 * x^3 + a * x^2 + b * x + 60) →
  t = -9 :=
by
  intros h
  -- We'll skip the proof part
  sorry

end find_t_l219_21974


namespace product_of_three_consecutive_integers_l219_21935

theorem product_of_three_consecutive_integers (x : ℕ) (h1 : x * (x + 1) = 740)
    (x1 : ℕ := x - 1) (x2 : ℕ := x) (x3 : ℕ := x + 1) :
    x1 * x2 * x3 = 17550 :=
by
  sorry

end product_of_three_consecutive_integers_l219_21935


namespace bricks_required_l219_21985

   -- Definitions from the conditions
   def courtyard_length_meters : ℝ := 42
   def courtyard_width_meters : ℝ := 22
   def brick_length_cm : ℝ := 16
   def brick_width_cm : ℝ := 10

   -- The Lean statement to prove
   theorem bricks_required : (courtyard_length_meters * courtyard_width_meters * 10000) / (brick_length_cm * brick_width_cm) = 57750 :=
   by 
       sorry
   
end bricks_required_l219_21985


namespace raman_salary_loss_l219_21988

theorem raman_salary_loss : 
  ∀ (S : ℝ), S > 0 →
  let decreased_salary := S - (0.5 * S) 
  let final_salary := decreased_salary + (0.5 * decreased_salary) 
  let loss := S - final_salary 
  let percentage_loss := (loss / S) * 100
  percentage_loss = 25 := 
by
  intros S hS
  let decreased_salary := S - (0.5 * S)
  let final_salary := decreased_salary + (0.5 * decreased_salary)
  let loss := S - final_salary
  let percentage_loss := (loss / S) * 100
  have h1 : decreased_salary = 0.5 * S := by sorry
  have h2 : final_salary = 0.75 * S := by sorry
  have h3 : loss = 0.25 * S := by sorry
  have h4 : percentage_loss = 25 := by sorry
  exact h4

end raman_salary_loss_l219_21988


namespace color_copies_comparison_l219_21916

theorem color_copies_comparison (n : ℕ) (pX pY : ℝ) (charge_diff : ℝ) 
  (h₀ : pX = 1.20) (h₁ : pY = 1.70) (h₂ : charge_diff = 35) 
  (h₃ : pY * n = pX * n + charge_diff) : n = 70 :=
by
  -- proof steps would go here
  sorry

end color_copies_comparison_l219_21916


namespace rate_of_interest_is_5_percent_l219_21930

-- Defining the conditions as constants
def simple_interest : ℝ := 4016.25
def principal : ℝ := 16065
def time_period : ℝ := 5

-- Proving that the rate of interest is 5%
theorem rate_of_interest_is_5_percent (R : ℝ) : 
  simple_interest = (principal * R * time_period) / 100 → 
  R = 5 :=
by
  intro h
  sorry

end rate_of_interest_is_5_percent_l219_21930


namespace unique_real_root_count_l219_21994

theorem unique_real_root_count :
  ∃! x : ℝ, (x^12 + 1) * (x^10 + x^8 + x^6 + x^4 + x^2 + 1) = 12 * x^11 := by
  sorry

end unique_real_root_count_l219_21994


namespace sum_of_cubes_divisible_by_9n_l219_21918

theorem sum_of_cubes_divisible_by_9n (n : ℕ) (h : n % 3 ≠ 0) : 
  ((n - 1)^3 + n^3 + (n + 1)^3) % (9 * n) = 0 := by
  sorry

end sum_of_cubes_divisible_by_9n_l219_21918


namespace converted_land_eqn_l219_21961

theorem converted_land_eqn (forest_land dry_land converted_dry_land : ℝ)
  (h1 : forest_land = 108)
  (h2 : dry_land = 54)
  (h3 : converted_dry_land = x) :
  (dry_land - converted_dry_land = 0.2 * (forest_land + converted_dry_land)) :=
by
  simp [h1, h2, h3]
  sorry

end converted_land_eqn_l219_21961


namespace area_XMY_l219_21934

-- Definitions
structure Triangle :=
(area : ℝ)

def ratio (a b : ℝ) : Prop := ∃ k : ℝ, (a = k * b)

-- Given conditions
variables {XYZ XMY YZ MY : ℝ}
variables (h1 : ratio XYZ 35)
variables (h2 : ratio (XM / MY) (5 / 2))

-- Theorem to prove
theorem area_XMY (hYZ_ratio : YZ = XM + MY) (hshared_height : true) : XMY = 10 :=
by
  sorry

end area_XMY_l219_21934


namespace geometric_sequence_sum_l219_21951

variable (a : ℕ → ℝ)
variable (q : ℝ)

axiom h1 : a 1 + a 2 = 20
axiom h2 : a 3 + a 4 = 40
axiom h3 : q^2 = 2

theorem geometric_sequence_sum : a 5 + a 6 = 80 :=
by
  sorry

end geometric_sequence_sum_l219_21951


namespace rosemary_leaves_count_l219_21901

-- Define the number of pots for each plant type
def basil_pots : ℕ := 3
def rosemary_pots : ℕ := 9
def thyme_pots : ℕ := 6

-- Define the number of leaves each plant type has
def basil_leaves : ℕ := 4
def thyme_leaves : ℕ := 30
def total_leaves : ℕ := 354

-- Prove that the number of leaves on each rosemary plant is 18
theorem rosemary_leaves_count (R : ℕ) (h : basil_pots * basil_leaves + rosemary_pots * R + thyme_pots * thyme_leaves = total_leaves) : R = 18 :=
by {
  -- Following steps are within the theorem's proof
  sorry
}

end rosemary_leaves_count_l219_21901


namespace binary_to_base5_l219_21964

theorem binary_to_base5 : Nat.digits 5 (Nat.ofDigits 2 [1, 0, 1, 1, 0, 0, 1]) = [4, 2, 3] :=
by
  sorry

end binary_to_base5_l219_21964


namespace kevin_age_l219_21965

theorem kevin_age (x : ℕ) :
  (∃ n : ℕ, x - 2 = n^2) ∧ (∃ m : ℕ, x + 2 = m^3) → x = 6 :=
by
  sorry

end kevin_age_l219_21965


namespace work_completion_l219_21957

theorem work_completion (p q : ℝ) (h1 : p = 1.60 * q) (h2 : (1 / p + 1 / q) = 1 / 16) : p = 1 / 26 := 
by {
  -- This will be followed by the proof steps, but we add sorry since only the statement is required
  sorry
}

end work_completion_l219_21957


namespace spinner_final_direction_l219_21931

-- Define the directions as an enumeration
inductive Direction
| north
| east
| south
| west

-- Convert between revolution fractions to direction
def direction_after_revolutions (initial : Direction) (revolutions : ℚ) : Direction :=
  let quarters := (revolutions * 4) % 4
  match initial with
  | Direction.south => if quarters == 0 then Direction.south
                       else if quarters == 1 then Direction.west
                       else if quarters == 2 then Direction.north
                       else Direction.east
  | Direction.east  => if quarters == 0 then Direction.east
                       else if quarters == 1 then Direction.south
                       else if quarters == 2 then Direction.west
                       else Direction.north
  | Direction.north => if quarters == 0 then Direction.north
                       else if quarters == 1 then Direction.east
                       else if quarters == 2 then Direction.south
                       else Direction.west
  | Direction.west  => if quarters == 0 then Direction.west
                       else if quarters == 1 then Direction.north
                       else if quarters == 2 then Direction.east
                       else Direction.south

-- Final proof statement
theorem spinner_final_direction : direction_after_revolutions Direction.south (4 + 3/4 - (6 + 1/2)) = Direction.east := 
by 
  sorry

end spinner_final_direction_l219_21931


namespace toothpicks_for_10_squares_l219_21950

theorem toothpicks_for_10_squares : (4 + 3 * (10 - 1)) = 31 :=
by 
  sorry

end toothpicks_for_10_squares_l219_21950


namespace product_of_x_and_y_l219_21927

variables (EF FG GH HE : ℕ) (x y : ℕ)

theorem product_of_x_and_y (h1: EF = 42) (h2: FG = 4 * y^3) (h3: GH = 2 * x + 10) (h4: HE = 32) (h5: EF = GH) (h6: FG = HE) :
  x * y = 32 :=
by
  sorry

end product_of_x_and_y_l219_21927


namespace simplify_expression_correct_l219_21919

noncomputable def simplify_expression (α : ℝ) : ℝ :=
    (2 * (Real.cos (2 * α))^2 - 1) / 
    (2 * Real.tan ((Real.pi / 4) - 2 * α) * (Real.sin ((3 * Real.pi / 4) - 2 * α))^2) -
    Real.tan (2 * α) + Real.cos (2 * α) - Real.sin (2 * α)

theorem simplify_expression_correct (α : ℝ) : 
    simplify_expression α = 
    (2 * Real.sqrt 2 * Real.sin ((Real.pi / 4) - 2 * α) * (Real.cos α)^2) /
    Real.cos (2 * α) := by
    sorry

end simplify_expression_correct_l219_21919


namespace parabola_focus_eq_l219_21920

/-- Given the equation of a parabola y = -4x^2 - 8x + 1, prove that its focus is at (-1, 79/16). -/
theorem parabola_focus_eq :
  ∀ x y : ℝ, y = -4 * x ^ 2 - 8 * x + 1 → 
  ∃ h k p : ℝ, y = -4 * (x + 1)^2 + 5 ∧ 
  h = -1 ∧ k = 5 ∧ p = -1 / 16 ∧ (h, k + p) = (-1, 79/16) :=
by
  sorry

end parabola_focus_eq_l219_21920


namespace determine_n_l219_21940

noncomputable def average_value (n : ℕ) : ℚ :=
  (n * (n + 1) * (2 * n + 1) : ℚ) / (6 * (n * (n + 1) / 2))

theorem determine_n :
  ∃ n : ℕ, average_value n = 2020 ∧ n = 3029 :=
sorry

end determine_n_l219_21940


namespace determine_x_l219_21976

theorem determine_x (x : ℚ) (n : ℤ) (d : ℚ) 
  (h_cond : x = n + d)
  (h_floor : n = ⌊x⌋)
  (h_d : 0 ≤ d ∧ d < 1)
  (h_eq : ⌊x⌋ + x = 17 / 4) :
  x = 9 / 4 := sorry

end determine_x_l219_21976


namespace football_total_points_l219_21910

theorem football_total_points :
  let Zach_points := 42.0
  let Ben_points := 21.0
  let Sarah_points := 18.5
  let Emily_points := 27.5
  Zach_points + Ben_points + Sarah_points + Emily_points = 109.0 :=
by
  let Zach_points := 42.0
  let Ben_points := 21.0
  let Sarah_points := 18.5
  let Emily_points := 27.5
  have h : Zach_points + Ben_points + Sarah_points + Emily_points = 42.0 + 21.0 + 18.5 + 27.5 := by rfl
  have total_points := 42.0 + 21.0 + 18.5 + 27.5
  have result := 109.0
  sorry

end football_total_points_l219_21910


namespace mandy_yoga_time_l219_21943

-- Define the conditions
def ratio_swimming := 1
def ratio_running := 2
def ratio_gym := 3
def ratio_biking := 5
def ratio_yoga := 4

def time_biking := 30

-- Define the Lean 4 statement to prove
theorem mandy_yoga_time : (time_biking / ratio_biking) * ratio_yoga = 24 :=
by
  sorry

end mandy_yoga_time_l219_21943


namespace inequality_proof_l219_21969

theorem inequality_proof (x : ℝ) (n : ℕ) (hx : 0 < x) : 
  1 + x^(n+1) ≥ (2*x)^n / (1 + x)^(n-1) := 
by
  sorry

end inequality_proof_l219_21969


namespace find_d_and_a11_l219_21906

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem find_d_and_a11 (a : ℕ → ℤ) (d : ℤ) :
  arithmetic_sequence a d →
  a 5 = 6 →
  a 8 = 15 →
  d = 3 ∧ a 11 = 24 :=
by
  intros h_seq h_a5 h_a8
  sorry

end find_d_and_a11_l219_21906


namespace factor_expression_l219_21949

variable (y : ℝ)

theorem factor_expression : 64 - 16 * y ^ 3 = 16 * (2 - y) * (4 + 2 * y + y ^ 2) := by
  sorry

end factor_expression_l219_21949


namespace gasoline_price_decrease_l219_21938

theorem gasoline_price_decrease (a : ℝ) (h : 0 ≤ a) :
  8.1 * (1 - a / 100) ^ 2 = 7.8 :=
sorry

end gasoline_price_decrease_l219_21938


namespace area_comparison_l219_21992

namespace Quadrilaterals

open Real

-- Define the vertices of both quadrilaterals
def quadrilateral_I_vertices : List (ℝ × ℝ) := [(0, 0), (2, 0), (2, 2), (0, 1)]
def quadrilateral_II_vertices : List (ℝ × ℝ) := [(0, 0), (3, 0), (3, 1), (0, 2)]

-- Area calculation function (example function for clarity)
def area_of_quadrilateral (vertices : List (ℝ × ℝ)) : ℝ :=
  -- This would use the actual geometry to compute the area
  2.5 -- placeholder for the area of quadrilateral I
  -- 4.5 -- placeholder for the area of quadrilateral II

theorem area_comparison :
  (area_of_quadrilateral quadrilateral_I_vertices) < (area_of_quadrilateral quadrilateral_II_vertices) :=
  sorry

end Quadrilaterals

end area_comparison_l219_21992


namespace least_positive_x_l219_21975

theorem least_positive_x (x : ℕ) : ((2 * x) ^ 2 + 2 * 41 * 2 * x + 41 ^ 2) % 53 = 0 ↔ x = 6 := 
sorry

end least_positive_x_l219_21975


namespace probability_exactly_one_each_is_correct_l219_21907

def probability_one_each (total forks spoons knives teaspoons : ℕ) : ℚ :=
  (forks * spoons * knives * teaspoons : ℚ) / ((total.choose 4) : ℚ)

theorem probability_exactly_one_each_is_correct :
  probability_one_each 34 8 9 10 7 = 40 / 367 :=
by sorry

end probability_exactly_one_each_is_correct_l219_21907


namespace clark_family_ticket_cost_l219_21909

theorem clark_family_ticket_cost
  (regular_price children's_price seniors_price : ℝ)
  (number_youngest_gen number_second_youngest_gen number_second_oldest_gen number_oldest_gen : ℕ)
  (h_senior_discount : seniors_price = 0.7 * regular_price)
  (h_senior_ticket_cost : seniors_price = 7)
  (h_child_discount : children's_price = 0.6 * regular_price)
  (h_number_youngest_gen : number_youngest_gen = 3)
  (h_number_second_youngest_gen : number_second_youngest_gen = 1)
  (h_number_second_oldest_gen : number_second_oldest_gen = 2)
  (h_number_oldest_gen : number_oldest_gen = 1)
  : 3 * children's_price + 1 * regular_price + 2 * seniors_price + 1 * regular_price = 52 := by
  sorry

end clark_family_ticket_cost_l219_21909


namespace no_x_intersections_geometric_sequence_l219_21962

theorem no_x_intersections_geometric_sequence (a b c : ℝ) 
  (h1 : b^2 = a * c)
  (h2 : a * c > 0) : 
  (∃ x : ℝ, a * x^2 + b * x + c = 0) = false :=
by
  sorry

end no_x_intersections_geometric_sequence_l219_21962


namespace solve_length_BF_l219_21970

-- Define the problem conditions
def rectangular_paper (short_side long_side : ℝ) : Prop :=
  short_side = 12 ∧ long_side > short_side

def vertex_touch_midpoint (vmp mid : ℝ) : Prop :=
  vmp = mid / 2

def congruent_triangles (triangle1 triangle2 : ℝ) : Prop :=
  triangle1 = triangle2

-- Theorem to prove the length of BF
theorem solve_length_BF (short_side long_side vmp mid triangle1 triangle2 : ℝ) 
  (h1 : rectangular_paper short_side long_side)
  (h2 : vertex_touch_midpoint vmp mid)
  (h3 : congruent_triangles triangle1 triangle2) :
  -- The length of BF is 10
  mid = 6 → 18 - 6 = 12 + 6 - 10 → 10 = 12 - (18 - 10) → vmp = 6 → 6 * 2 = 12 →
  sorry :=
sorry

end solve_length_BF_l219_21970


namespace solve_eq1_solve_eq2_l219_21959

-- Define the first problem statement and the correct answers
theorem solve_eq1 (x : ℝ) (h : (x - 2) ^ 2 = 169) : x = 15 ∨ x = -11 := 
  by sorry

-- Define the second problem statement and the correct answer
theorem solve_eq2 (x : ℝ) (h : 3 * (x - 3) ^ 3 - 24 = 0) : x = 5 := 
  by sorry

end solve_eq1_solve_eq2_l219_21959


namespace percentage_decrease_correct_l219_21978

theorem percentage_decrease_correct :
  ∀ (p : ℝ), (1 + 0.25) * (1 - p) = 1 → p = 0.20 :=
by
  intro p
  intro h
  sorry

end percentage_decrease_correct_l219_21978


namespace min_value_of_angle_function_l219_21944

theorem min_value_of_angle_function (α β γ : ℝ) (h1 : α + β + γ = Real.pi) (h2 : 0 < α) (h3 : α < Real.pi) :
  ∃ α, α = (2 * Real.pi / 3) ∧ (4 / α + 1 / (Real.pi - α)) = (9 / Real.pi) := by
  sorry

end min_value_of_angle_function_l219_21944


namespace complement_intersection_l219_21991

-- Define the universal set
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2 * x > 0}

-- Define complement of A in U
def C_U_A : Set ℝ := U \ A

-- Define set B
def B : Set ℝ := {x | x > 1}

-- State the theorem
theorem complement_intersection (x : ℝ) : x ∈ C_U_A ∩ B ↔ 1 < x ∧ x ≤ 2 :=
by
   sorry

end complement_intersection_l219_21991


namespace range_of_expression_l219_21972

theorem range_of_expression (x y : ℝ) 
  (h1 : x - 2 * y + 2 ≥ 0) 
  (h2 : x ≤ 1) 
  (h3 : x + y - 1 ≥ 0) : 
  3 / 2 ≤ (x + y + 2) / (x + 1) ∧ (x + y + 2) / (x + 1) ≤ 3 :=
by
  sorry

end range_of_expression_l219_21972


namespace master_wang_resting_on_sunday_again_l219_21942

theorem master_wang_resting_on_sunday_again (n : ℕ) 
  (works_days := 8) 
  (rest_days := 2) 
  (week_days := 7) 
  (cycle_days := works_days + rest_days) 
  (initial_rest_saturday_sunday : Prop) : 
  (initial_rest_saturday_sunday → ∃ n : ℕ, (week_days * n) % cycle_days = rest_days) → 
  (∃ n : ℕ, n = 7) :=
by
  sorry

end master_wang_resting_on_sunday_again_l219_21942


namespace min_value_of_reciprocal_sum_l219_21937

noncomputable def arithmetic_sequence_condition (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ ((2016 * (a 1 + a 2016)) / 2 = 1008)

theorem min_value_of_reciprocal_sum (a : ℕ → ℝ) (h : arithmetic_sequence_condition a) :
  ∃ x : ℝ, x = 4 ∧ (∀ y, y = (1 / a 1001 + 1 / a 1016) → x ≤ y) :=
sorry

end min_value_of_reciprocal_sum_l219_21937


namespace evaluate_fraction_sum_l219_21941

variable (a b c : ℝ)

theorem evaluate_fraction_sum
  (h : (a / (30 - a)) + (b / (70 - b)) + (c / (80 - c)) = 9) :
  (6 / (30 - a)) + (14 / (70 - b)) + (16 / (80 - c)) = 2.4 :=
by
  sorry

end evaluate_fraction_sum_l219_21941


namespace laborer_monthly_income_l219_21911

theorem laborer_monthly_income
  (I : ℕ)
  (D : ℕ)
  (h1 : 6 * I + D = 510)
  (h2 : 4 * I - D = 270) : I = 78 := by
  sorry

end laborer_monthly_income_l219_21911


namespace quadratic_prime_roots_l219_21913

theorem quadratic_prime_roots (k : ℕ) (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  p + q = 101 → p * q = k → False :=
by
  sorry

end quadratic_prime_roots_l219_21913


namespace total_area_covered_is_60_l219_21936

-- Declare the dimensions of the strips
def length_strip : ℕ := 12
def width_strip : ℕ := 2
def num_strips : ℕ := 3

-- Define the total area covered without overlaps
def total_area_no_overlap := num_strips * (length_strip * width_strip)

-- Define the area of overlap for each pair of strips
def overlap_area_per_pair := width_strip * width_strip

-- Define the total overlap area given 3 pairs
def total_overlap_area := 3 * overlap_area_per_pair

-- Define the actual total covered area
def total_covered_area := total_area_no_overlap - total_overlap_area

-- Prove that the total covered area is 60 square units
theorem total_area_covered_is_60 : total_covered_area = 60 := by 
  sorry

end total_area_covered_is_60_l219_21936


namespace range_of_m_l219_21958

theorem range_of_m (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2*x + m ≤ 0) →
  (1 < m) :=
by
  sorry

end range_of_m_l219_21958


namespace speed_in_terms_of_time_l219_21967

variable (a b x : ℝ)

-- Conditions
def condition1 : Prop := 1000 = a * x
def condition2 : Prop := 833 = b * x

-- The theorem to prove
theorem speed_in_terms_of_time (h1 : condition1 a x) (h2 : condition2 b x) :
  a = 1000 / x ∧ b = 833 / x :=
by
  sorry

end speed_in_terms_of_time_l219_21967


namespace stratified_sample_l219_21990

theorem stratified_sample 
  (total_households : ℕ) 
  (high_income_households : ℕ) 
  (middle_income_households : ℕ) 
  (low_income_households : ℕ) 
  (sample_size : ℕ)
  (H1 : total_households = 600) 
  (H2 : high_income_households = 150)
  (H3 : middle_income_households = 360)
  (H4 : low_income_households = 90)
  (H5 : sample_size = 100) : 
  (middle_income_households * sample_size / total_households = 60) := 
by 
  sorry

end stratified_sample_l219_21990


namespace drawings_with_colored_pencils_l219_21922

-- Definitions based on conditions
def total_drawings : Nat := 25
def blending_markers_drawings : Nat := 7
def charcoal_drawings : Nat := 4
def colored_pencils_drawings : Nat := total_drawings - (blending_markers_drawings + charcoal_drawings)

-- Theorem to be proven
theorem drawings_with_colored_pencils : colored_pencils_drawings = 14 :=
by
  sorry

end drawings_with_colored_pencils_l219_21922


namespace john_behind_steve_l219_21917

theorem john_behind_steve
  (vJ : ℝ) (vS : ℝ) (ahead : ℝ) (t : ℝ) (d : ℝ)
  (hJ : vJ = 4.2) (hS : vS = 3.8) (hA : ahead = 2) (hT : t = 42.5)
  (h1 : vJ * t = d + ahead)
  (h2 : vS * t + ahead = vJ * t - ahead) :
  d = 15 :=
by
  -- Proof omitted
  sorry

end john_behind_steve_l219_21917


namespace angles_arith_prog_triangle_l219_21905

noncomputable def a : ℕ := 8
noncomputable def b : ℕ := 37
noncomputable def c : ℕ := 0

theorem angles_arith_prog_triangle (y : ℝ) (h1 : y = 8 ∨ y * y = 37) :
  a + b + c = 45 := by
  -- skipping the detailed proof steps
  sorry

end angles_arith_prog_triangle_l219_21905


namespace least_boxes_l219_21948
-- Definitions and conditions
def isPerfectCube (n : ℕ) : Prop := ∃ (k : ℕ), k^3 = n

def isFactor (a b : ℕ) : Prop := ∃ k, a * k = b

def numBoxes (N boxSize : ℕ) : ℕ := N / boxSize

-- Specific conditions for our problem
theorem least_boxes (N : ℕ) (boxSize : ℕ) 
  (h1 : N ≠ 0) 
  (h2 : isPerfectCube N)
  (h3 : isFactor boxSize N)
  (h4 : boxSize = 45): 
  numBoxes N boxSize = 75 :=
by
  sorry

end least_boxes_l219_21948


namespace inequality_holds_l219_21983

theorem inequality_holds (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h : x * y + y * z + z * x = 1) :
  (1 / (x + y)) + (1 / (y + z)) + (1 / (z + x)) ≥ 5 / 2 :=
by
  sorry

end inequality_holds_l219_21983


namespace original_amount_of_rice_l219_21902

theorem original_amount_of_rice
  (x : ℕ) -- the total amount of rice in kilograms
  (h1 : x = 10 * 500) -- statement that needs to be proven
  (h2 : 210 = x * (21 / 50)) -- remaining rice condition after given fractions are consumed
  (consume_day_one : x - (3 / 10) * x  = (7 / 10) * x) -- after the first day's consumption
  (consume_day_two : ((7 / 10) * x) - ((2 / 5) * ((7 / 10) * x)) = 210) -- after the second day's consumption
  : x = 500 :=
by
  sorry

end original_amount_of_rice_l219_21902


namespace hypotenuse_square_l219_21996

theorem hypotenuse_square (a : ℕ) : (a + 1)^2 + a^2 = 2 * a^2 + 2 * a + 1 := 
by sorry

end hypotenuse_square_l219_21996


namespace car_rental_cost_l219_21993

def day1_cost (base_rate : ℝ) (miles_driven : ℝ) (cost_per_mile : ℝ) : ℝ :=
  base_rate + miles_driven * cost_per_mile

def day2_cost (base_rate : ℝ) (miles_driven : ℝ) (cost_per_mile : ℝ) : ℝ :=
  base_rate + miles_driven * cost_per_mile

def day3_cost (base_rate : ℝ) (miles_driven : ℝ) (cost_per_mile : ℝ) : ℝ :=
  base_rate + miles_driven * cost_per_mile

def total_cost (day1 : ℝ) (day2 : ℝ) (day3 : ℝ) : ℝ :=
  day1 + day2 + day3

theorem car_rental_cost :
  let day1_base_rate := 150
  let day2_base_rate := 100
  let day3_base_rate := 75
  let day1_miles_driven := 620
  let day2_miles_driven := 744
  let day3_miles_driven := 510
  let day1_cost_per_mile := 0.50
  let day2_cost_per_mile := 0.40
  let day3_cost_per_mile := 0.30
  day1_cost day1_base_rate day1_miles_driven day1_cost_per_mile +
  day2_cost day2_base_rate day2_miles_driven day2_cost_per_mile +
  day3_cost day3_base_rate day3_miles_driven day3_cost_per_mile = 1085.60 :=
by
  let day1 := day1_cost 150 620 0.50
  let day2 := day2_cost 100 744 0.40
  let day3 := day3_cost 75 510 0.30
  let total := total_cost day1 day2 day3
  show total = 1085.60
  sorry

end car_rental_cost_l219_21993


namespace num_divisors_of_m_cubed_l219_21997

theorem num_divisors_of_m_cubed (m : ℕ) (h : ∃ p : ℕ, Nat.Prime p ∧ m = p ^ 4) :
    Nat.totient (m ^ 3) = 13 := 
sorry

end num_divisors_of_m_cubed_l219_21997


namespace minimum_value_ineq_l219_21929

theorem minimum_value_ineq (x : ℝ) (hx : 0 < x) :
  3 * Real.sqrt x + 4 / x ≥ 4 * Real.sqrt 2 :=
by
  sorry

end minimum_value_ineq_l219_21929


namespace jovana_added_shells_l219_21904

theorem jovana_added_shells (initial_amount final_amount added_amount : ℕ) 
  (h1 : initial_amount = 5) 
  (h2 : final_amount = 17) 
  (h3 : added_amount = final_amount - initial_amount) : 
  added_amount = 12 := 
by 
  -- Since the proof is not required, we add sorry here to skip the proof.
  sorry 

end jovana_added_shells_l219_21904


namespace min_project_time_l219_21939

theorem min_project_time (A B C : ℝ) (D : ℝ := 12) :
  (1 / B + 1 / C) = 1 / 2 →
  (1 / A + 1 / C) = 1 / 3 →
  (1 / A + 1 / B) = 1 / 4 →
  (1 / D) = 1 / 12 →
  ∃ x : ℝ, x = 8 / 5 ∧ 1 / x = 1 / A + 1 / B + 1 / C + 1 / (12:ℝ) :=
by
  intros h1 h2 h3 h4
  -- Combination of given hypotheses to prove the goal
  sorry

end min_project_time_l219_21939


namespace f_cos_x_l219_21982

theorem f_cos_x (f : ℝ → ℝ) (x : ℝ) (h₁ : -1 ≤ x) (h₂ : x ≤ 1) (hx : f (Real.sin x) = 2 - Real.cos (2 * x)) :
  f (Real.cos x) = 2 + (Real.cos x)^2 :=
sorry

end f_cos_x_l219_21982


namespace part1_part2_l219_21989

-- Definitions and conditions
def a : ℕ := 60
def b : ℕ := 40
def c : ℕ := 80
def d : ℕ := 20
def n : ℕ := a + b + c + d

-- Given critical value for 99% certainty
def critical_value_99 : ℝ := 6.635

-- Calculate K^2 using the given formula
noncomputable def K_squared : ℝ := (n * ((a * d - b * c) ^ 2)) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Calculation of probability of selecting 2 qualified products from 5 before renovation
def total_sampled : ℕ := 5
def qualified_before_renovation : ℕ := 3
def total_combinations (n k : ℕ) : ℕ := Nat.choose n k
def prob_selecting_2_qualified : ℚ := (total_combinations qualified_before_renovation 2 : ℚ) / 
                                      (total_combinations total_sampled 2 : ℚ)

-- Proof statements
theorem part1 : K_squared > critical_value_99 := by
  sorry

theorem part2 : prob_selecting_2_qualified = 3 / 10 := by
  sorry

end part1_part2_l219_21989


namespace intersection_point_of_given_lines_l219_21979

theorem intersection_point_of_given_lines :
  ∃ (x y : ℚ), 2 * y = -x + 3 ∧ -y = 5 * x + 1 ∧ x = -5 / 9 ∧ y = 16 / 9 :=
by
  sorry

end intersection_point_of_given_lines_l219_21979


namespace division_value_l219_21963

theorem division_value (x : ℝ) (h : 800 / x - 154 = 6) : x = 5 := by
  sorry

end division_value_l219_21963


namespace weight_differences_correct_l219_21953

-- Define the weights of Heather, Emily, Elizabeth, and Emma
def H : ℕ := 87
def E1 : ℕ := 58
def E2 : ℕ := 56
def E3 : ℕ := 64

-- Proof problem statement
theorem weight_differences_correct :
  (H - E1 = 29) ∧ (H - E2 = 31) ∧ (H - E3 = 23) :=
by
  -- Note: 'sorry' is used to skip the proof itself
  sorry

end weight_differences_correct_l219_21953


namespace january_first_is_tuesday_l219_21955

-- Define the days of the week for convenience
inductive Weekday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define the problem conditions
def daysInJanuary : Nat := 31
def weeksInJanuary : Nat := daysInJanuary / 7   -- This is 4 weeks
def extraDays : Nat := daysInJanuary % 7         -- This leaves 3 extra days

-- Define the problem as proving January 1st is a Tuesday
theorem january_first_is_tuesday (fridaysInJanuary : Nat) (mondaysInJanuary : Nat)
    (h_friday : fridaysInJanuary = 4) (h_monday: mondaysInJanuary = 4) : Weekday :=
  -- Avoid specific proof steps from the solution; assume conditions and directly prove the result
  sorry

end january_first_is_tuesday_l219_21955


namespace series_sum_equals_four_l219_21966

/-- 
  Proof of the sum of the series: 
  ∑ (n=1 to ∞) (6n² - n + 1) / (n⁵ - n⁴ + n³ - n² + n) = 4 
--/
theorem series_sum_equals_four :
  (∑' n : ℕ, (if n > 0 then (6 * n^2 - n + 1 : ℝ) / (n^5 - n^4 + n^3 - n^2 + n) else 0)) = 4 :=
by
  sorry

end series_sum_equals_four_l219_21966


namespace statues_added_in_third_year_l219_21932

/-
Definition of the turtle statues problem:

1. Initially, there are 4 statues in the first year.
2. In the second year, the number of statues quadruples.
3. In the third year, x statues are added, and then 3 statues are broken.
4. In the fourth year, 2 * 3 new statues are added.
5. In total, at the end of the fourth year, there are 31 statues.
-/

def year1_statues : ℕ := 4
def year2_statues : ℕ := 4 * year1_statues
def before_hailstorm_year3_statues (x : ℕ) : ℕ := year2_statues + x
def after_hailstorm_year3_statues (x : ℕ) : ℕ := before_hailstorm_year3_statues x - 3
def total_year4_statues (x : ℕ) : ℕ := after_hailstorm_year3_statues x + 2 * 3

theorem statues_added_in_third_year (x : ℕ) (h : total_year4_statues x = 31) : x = 12 :=
by
  sorry

end statues_added_in_third_year_l219_21932


namespace gcd_80_180_450_l219_21928

theorem gcd_80_180_450 : Int.gcd (Int.gcd 80 180) 450 = 10 := by
  sorry

end gcd_80_180_450_l219_21928


namespace eccentricity_range_of_isosceles_right_triangle_l219_21921

theorem eccentricity_range_of_isosceles_right_triangle
  (a : ℝ) (e : ℝ)
  (ellipse_eq : ∀ (x y : ℝ), (x^2)/(a^2) + y^2 = 1)
  (h_a_gt_1 : a > 1)
  (B C : ℝ × ℝ)
  (isosceles_right_triangle : ∀ (A B C : ℝ × ℝ), ∃ k : ℝ, k > 0 ∧ 
    B = (-(2*k*a^2)/(1 + a^2*k^2), 0) ∧ 
    C = ((2*k*a^2)/(a^2 + k^2), 0) ∧ 
    (B.1^2 + B.2^2 = C.1^2 + C.2^2 + 1))
  (unique_solution : ∀ (k : ℝ), ∃! k', k' = 1)
  : 0 < e ∧ e ≤ (Real.sqrt 6) / 3 :=
sorry

end eccentricity_range_of_isosceles_right_triangle_l219_21921


namespace measure_of_angle_E_l219_21954

theorem measure_of_angle_E
    (A B C D E F : ℝ)
    (h1 : A = B)
    (h2 : B = C)
    (h3 : C = D)
    (h4 : E = F)
    (h5 : A = E - 30)
    (h6 : A + B + C + D + E + F = 720) :
  E = 140 :=
by
  -- Proof goes here
  sorry

end measure_of_angle_E_l219_21954


namespace calc_expression_l219_21952

theorem calc_expression :
  (-(1 / 2))⁻¹ - 4 * Real.cos (Real.pi / 6) - (Real.pi + 2013)^0 + Real.sqrt 12 = -3 :=
by
  sorry

end calc_expression_l219_21952


namespace friendly_snakes_not_blue_l219_21923

variable (Snakes : Type)
variable (sally_snakes : Finset Snakes)
variable (blue : Snakes → Prop)
variable (friendly : Snakes → Prop)
variable (can_swim : Snakes → Prop)
variable (can_climb : Snakes → Prop)

variable [DecidablePred blue] [DecidablePred friendly] [DecidablePred can_swim] [DecidablePred can_climb]

-- The number of snakes in Sally's collection
axiom h_snakes_count : sally_snakes.card = 20
-- There are 7 blue snakes
axiom h_blue : (sally_snakes.filter blue).card = 7
-- There are 10 friendly snakes
axiom h_friendly : (sally_snakes.filter friendly).card = 10
-- All friendly snakes can swim
axiom h1 : ∀ s ∈ sally_snakes, friendly s → can_swim s
-- No blue snakes can climb
axiom h2 : ∀ s ∈ sally_snakes, blue s → ¬ can_climb s
-- Snakes that can't climb also can't swim
axiom h3 : ∀ s ∈ sally_snakes, ¬ can_climb s → ¬ can_swim s

theorem friendly_snakes_not_blue :
  ∀ s ∈ sally_snakes, friendly s → ¬ blue s :=
by
  sorry

end friendly_snakes_not_blue_l219_21923
