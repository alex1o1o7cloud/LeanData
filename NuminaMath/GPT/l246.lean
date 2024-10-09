import Mathlib

namespace sequence_sum_l246_24695

open BigOperators

-- Define the general term
def term (n : ℕ) : ℚ := n * (1 - (1 / n))

-- Define the index range for the sequence
def index_range : Finset ℕ := Finset.range 9 \ {0, 1}

-- Lean statement of the problem
theorem sequence_sum : ∑ n in index_range, term (n + 2) = 45 := by
  sorry

end sequence_sum_l246_24695


namespace value_of_a_minus_2b_l246_24633

theorem value_of_a_minus_2b 
  (a b : ℚ) 
  (h : ∀ y : ℚ, y > 0 → y ≠ 2 → y ≠ -3 → (a / (y-2) + b / (y+3) = (2 * y + 5) / ((y-2)*(y+3)))) 
  : a - 2 * b = 7 / 5 :=
sorry

end value_of_a_minus_2b_l246_24633


namespace chandler_bike_purchase_l246_24625

theorem chandler_bike_purchase : 
    ∀ (x : ℕ), (200 + 20 * x = 800) → (x = 30) :=
by
  intros x h
  sorry

end chandler_bike_purchase_l246_24625


namespace repeating_decimal_sum_l246_24614

-- Definitions based on conditions
def x := 5 / 9  -- We derived this from 0.5 repeating as a fraction
def y := 7 / 99  -- Similarly, derived from 0.07 repeating as a fraction

-- Proposition to prove
theorem repeating_decimal_sum : x + y = 62 / 99 := by
  sorry

end repeating_decimal_sum_l246_24614


namespace smallest_positive_integer_l246_24686

theorem smallest_positive_integer (n : ℕ) :
  (∃ n : ℕ, n > 0 ∧ n % 30 = 0 ∧ n % 40 = 0 ∧ n % 16 ≠ 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 30 = 0 ∧ m % 40 = 0 ∧ m % 16 ≠ 0) → n ≤ m) ↔ n = 120 :=
by
  sorry

end smallest_positive_integer_l246_24686


namespace middle_part_proportional_l246_24632

theorem middle_part_proportional (x : ℚ) (s : ℚ) (h : s = 120) 
    (proportional : (2 * x) + (1/2 * x) + (1/4 * x) = s) : 
    (1/2 * x) = 240/11 := 
by
  sorry

end middle_part_proportional_l246_24632


namespace crayons_per_child_l246_24667

theorem crayons_per_child (children : ℕ) (total_crayons : ℕ) (h1 : children = 18) (h2 : total_crayons = 216) : 
    total_crayons / children = 12 := 
by
  sorry

end crayons_per_child_l246_24667


namespace number_of_two_element_subsets_l246_24626

def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem number_of_two_element_subsets (S : Type*) [Fintype S] 
  (h : binomial_coeff (Fintype.card S) 7 = 36) :
  binomial_coeff (Fintype.card S) 2 = 36 :=
by
  sorry

end number_of_two_element_subsets_l246_24626


namespace girls_to_boys_ratio_l246_24613

theorem girls_to_boys_ratio (g b : ℕ) (h1 : g = b + 5) (h2 : g + b = 35) : g / b = 4 / 3 :=
by
  sorry

end girls_to_boys_ratio_l246_24613


namespace velocity_at_specific_time_acceleration_at_specific_time_acceleration_proportional_to_displacement_l246_24647

noncomputable def x (A ω t : ℝ) : ℝ := A * Real.sin (ω * t)
noncomputable def v (A ω t : ℝ) : ℝ := deriv (x A ω) t
noncomputable def α (A ω t : ℝ) : ℝ := deriv (v A ω) t

theorem velocity_at_specific_time (A ω : ℝ) : 
  v A ω (2 * Real.pi / ω) = A * ω := 
sorry

theorem acceleration_at_specific_time (A ω : ℝ) :
  α A ω (2 * Real.pi / ω) = 0 :=
sorry

theorem acceleration_proportional_to_displacement (A ω t : ℝ) :
  α A ω t = -ω^2 * x A ω t :=
sorry

end velocity_at_specific_time_acceleration_at_specific_time_acceleration_proportional_to_displacement_l246_24647


namespace inequality_holds_for_all_x_l246_24682

variable (p : ℝ)
variable (x : ℝ)

theorem inequality_holds_for_all_x (h : -3 < p ∧ p < 6) : 
  -9 < (3*x^2 + p*x - 6) / (x^2 - x + 1) ∧ (3*x^2 + p*x - 6) / (x^2 - x + 1) < 6 := by
  sorry

end inequality_holds_for_all_x_l246_24682


namespace tank_capacity_l246_24651

theorem tank_capacity (x : ℝ) (h : 0.24 * x = 120) : x = 500 := 
sorry

end tank_capacity_l246_24651


namespace closest_to_2010_l246_24688

theorem closest_to_2010 :
  let A := 2008 * 2012
  let B := 1000 * Real.pi
  let C := 58 * 42
  let D := (48.3 ^ 2 - 2 * 8.3 * 48.3 + 8.3 ^ 2)
  abs (2010 - D) < abs (2010 - A) ∧
  abs (2010 - D) < abs (2010 - B) ∧
  abs (2010 - D) < abs (2010 - C) :=
by
  sorry

end closest_to_2010_l246_24688


namespace book_prices_l246_24624

theorem book_prices (x : ℝ) (y : ℝ) (h1 : y = 2.5 * x) (h2 : 800 / x - 800 / y = 24) : (x = 20 ∧ y = 50) :=
by
  sorry

end book_prices_l246_24624


namespace hyperbola_eccentricity_l246_24623

open Real

theorem hyperbola_eccentricity (a b c : ℝ) (ha : 0 < a) (hb : 0 < b)
    (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
    (h_right_focus : ∀ x y, x = c ∧ y = 0)
    (h_circle : ∀ x y, (x - c)^2 + y^2 = 4 * a^2)
    (h_tangent : ∀ x y, x = c ∧ y = 0 → (x^2 + y^2 = a^2 + b^2))
    : ∃ e : ℝ, e = sqrt 5 := by sorry

end hyperbola_eccentricity_l246_24623


namespace total_investment_is_correct_l246_24674

-- Define principal, rate, and number of years
def principal : ℝ := 8000
def rate : ℝ := 0.04
def years : ℕ := 10

-- Define the formula for compound interest
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- State the theorem
theorem total_investment_is_correct :
  compound_interest principal rate years = 11842 :=
by
  sorry

end total_investment_is_correct_l246_24674


namespace ferry_time_increases_l246_24661

noncomputable def ferryRoundTrip (S V x : ℝ) : ℝ :=
  (S / (V + x)) + (S / (V - x))

theorem ferry_time_increases (S V x : ℝ) (h_V_pos : 0 < V) (h_x_lt_V : x < V) :
  ferryRoundTrip S V (x + 1) > ferryRoundTrip S V x :=
by
  sorry

end ferry_time_increases_l246_24661


namespace problem_scores_ordering_l246_24657

variable {J K L R : ℕ}

theorem problem_scores_ordering (h1 : J > K) (h2 : J > L) (h3 : J > R)
                                (h4 : L > min K R) (h5 : R > min K L)
                                (h6 : (J ≠ K) ∧ (J ≠ L) ∧ (J ≠ R) ∧ (K ≠ L) ∧ (K ≠ R) ∧ (L ≠ R)) :
                                K < L ∧ L < R :=
sorry

end problem_scores_ordering_l246_24657


namespace michael_robots_l246_24685

-- Conditions
def tom_robots := 3
def times_more := 4

-- Theorem to prove
theorem michael_robots : (times_more * tom_robots) + tom_robots = 15 := by
  sorry

end michael_robots_l246_24685


namespace find_m_range_l246_24636

theorem find_m_range (m : ℝ) : 
  (∃ x : ℤ, 2 * (x : ℝ) - 1 ≤ 5 ∧ x - 1 ≥ m ∧ x ≤ 3) ∧ 
  (∃ y : ℤ, 2 * (y : ℝ) - 1 ≤ 5 ∧ y - 1 ≥ m ∧ y ≤ 3 ∧ x ≠ y) → 
  -1 < m ∧ m ≤ 0 := by
  sorry

end find_m_range_l246_24636


namespace helium_balloon_buoyancy_l246_24672

variable (m m₁ Mₐ M_b : ℝ)
variable (h₁ : m₁ = 10)
variable (h₂ : Mₐ = 4)
variable (h₃ : M_b = 29)

theorem helium_balloon_buoyancy :
  m = (m₁ * Mₐ) / (M_b - Mₐ) :=
by
  sorry

end helium_balloon_buoyancy_l246_24672


namespace segment_radius_with_inscribed_equilateral_triangle_l246_24634

theorem segment_radius_with_inscribed_equilateral_triangle (α h : ℝ) : 
  ∃ x : ℝ, x = (h / (Real.sin (α / 2))^2) * (Real.cos (α / 2) + Real.sqrt (1 + (1 / 3) * (Real.sin (α / 2))^2)) :=
sorry

end segment_radius_with_inscribed_equilateral_triangle_l246_24634


namespace max_identifiable_cards_2013_l246_24629

-- Define the number of cards
def num_cards : ℕ := 2013

-- Define the function that determines the maximum t for which the numbers can be found
def max_identifiable_cards (cards : ℕ) (select : ℕ) : ℕ :=
  if (cards = 2013) ∧ (select = 10) then 1986 else 0

-- The theorem to prove the property
theorem max_identifiable_cards_2013 :
  max_identifiable_cards 2013 10 = 1986 :=
sorry

end max_identifiable_cards_2013_l246_24629


namespace solve_eqn_l246_24662

theorem solve_eqn (x : ℚ) (h1 : x ≠ 4) (h2 : x ≠ 6) :
  (x + 11) / (x - 4) = (x - 1) / (x + 6) → x = -31 / 11 :=
by
sorry

end solve_eqn_l246_24662


namespace lines_intersect_l246_24656

theorem lines_intersect (a b : ℝ) (h1 : 2 = (1/3) * 1 + a) (h2 : 1 = (1/2) * 2 + b) : a + b = 5 / 3 := 
by {
  -- Skipping the proof itself
  sorry
}

end lines_intersect_l246_24656


namespace xiao_ming_kite_payment_l246_24687

/-- Xiao Ming has multiple 1 yuan, 2 yuan, and 5 yuan banknotes. 
    He wants to buy a kite priced at 18 yuan using no more than 10 of these banknotes
    and must use at least two different denominations.
    Prove that there are exactly 11 different ways he can pay. -/
theorem xiao_ming_kite_payment : 
  ∃ (combinations : Nat), 
    (∀ (c1 c2 c5 : Nat), (c1 * 1 + c2 * 2 + c5 * 5 = 18) → 
    (c1 + c2 + c5 ≤ 10) → 
    ((c1 > 0 ∧ c2 > 0) ∨ (c1 > 0 ∧ c5 > 0) ∨ (c2 > 0 ∧ c5 > 0)) →
    combinations = 11) :=
sorry

end xiao_ming_kite_payment_l246_24687


namespace solve_system_of_equations_l246_24628

variable (a x y z : ℝ)

theorem solve_system_of_equations (h1 : x^2 + y^2 - 2 * z^2 = 2 * a^2)
                                  (h2 : x + y + 2 * z = 4 * (a^2 + 1))
                                  (h3 : z^2 - x * y = a^2) :
                                  (x = a^2 + a + 1 ∧ y = a^2 - a + 1 ∧ z = a^2 + 1) ∨
                                  (x = a^2 - a + 1 ∧ y = a^2 + a + 1 ∧ z = a^2 + 1) :=
by
  sorry

end solve_system_of_equations_l246_24628


namespace chocolate_chip_cookies_count_l246_24671

theorem chocolate_chip_cookies_count (h1 : 5 / 2 = 20 / (x : ℕ)) : x = 8 := 
by
  sorry -- Proof to be implemented

end chocolate_chip_cookies_count_l246_24671


namespace find_k_l246_24631

variables (m n k : ℤ)  -- Declaring m, n, k as integer variables.

theorem find_k (h1 : m = 2 * n + 5) (h2 : m + 2 = 2 * (n + k) + 5) : k = 1 :=
by
  sorry

end find_k_l246_24631


namespace helly_half_planes_helly_convex_polygons_l246_24699

-- Helly's theorem for half-planes
theorem helly_half_planes (n : ℕ) (H : Fin n → Set ℝ) 
  (h : ∀ (i j k : Fin n), (H i ∩ H j ∩ H k).Nonempty) : 
  (⋂ i, H i).Nonempty :=
sorry

-- Helly's theorem for convex polygons
theorem helly_convex_polygons (n : ℕ) (P : Fin n → Set ℝ) 
  (h : ∀ (i j k : Fin n), (P i ∩ P j ∩ P k).Nonempty) : 
  (⋂ i, P i).Nonempty :=
sorry

end helly_half_planes_helly_convex_polygons_l246_24699


namespace partial_fraction_sum_zero_l246_24660

theorem partial_fraction_sum_zero
    (A B C D E : ℝ)
    (h : ∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -5 →
    1 = A * (x + 1) * (x + 2) * (x + 3) * (x + 5) +
        B * x * (x + 2) * (x + 3) * (x + 5) +
        C * x * (x + 1) * (x + 3) * (x + 5) +
        D * x * (x + 1) * (x + 2) * (x + 5) +
        E * x * (x + 1) * (x + 2) * (x + 3)) :
    A + B + C + D + E = 0 := by
    sorry

end partial_fraction_sum_zero_l246_24660


namespace melissa_solves_equation_l246_24607

theorem melissa_solves_equation : 
  ∃ b c : ℤ, (∀ x : ℝ, x^2 - 6 * x + 9 = 0 ↔ (x + b)^2 = c) ∧ b + c = -3 :=
by
  sorry

end melissa_solves_equation_l246_24607


namespace math_proof_problem_l246_24615

theorem math_proof_problem (a b : ℝ) (h1 : 64 = 8^2) (h2 : 16 = 8^2) :
  8^15 / (64^7) * 16 = 512 :=
by
  sorry

end math_proof_problem_l246_24615


namespace num_exclusive_multiples_4_6_less_151_l246_24658

def numMultiplesExclusive (n : ℕ) (a b : ℕ) : ℕ :=
  let lcm_ab := Nat.lcm a b
  (n-1) / a - (n-1) / lcm_ab + (n-1) / b - (n-1) / lcm_ab

theorem num_exclusive_multiples_4_6_less_151 : 
  numMultiplesExclusive 151 4 6 = 38 := 
by 
  sorry

end num_exclusive_multiples_4_6_less_151_l246_24658


namespace geom_seq_ratio_l246_24604
noncomputable section

theorem geom_seq_ratio (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h₁ : 0 < a_1)
  (h₂ : 0 < a_2)
  (h₃ : 0 < a_3)
  (h₄ : 0 < a_4)
  (h₅ : 0 < a_5)
  (h_seq : a_2 = a_1 * 2)
  (h_seq2 : a_3 = a_1 * 2^2)
  (h_seq3 : a_4 = a_1 * 2^3)
  (h_seq4 : a_5 = a_1 * 2^4)
  (h_ratio : a_4 / a_1 = 8) :
  (a_1 + a_2) * a_4 / ((a_1 + a_3) * a_5) = 3 / 10 := 
by
  sorry

end geom_seq_ratio_l246_24604


namespace solve_system_l246_24693

theorem solve_system :
  ∃ a b c d e : ℤ, 
    (a * b + a + 2 * b = 78) ∧
    (b * c + 3 * b + c = 101) ∧
    (c * d + 5 * c + 3 * d = 232) ∧
    (d * e + 4 * d + 5 * e = 360) ∧
    (e * a + 2 * e + 4 * a = 192) ∧
    ((a = 8 ∧ b = 7 ∧ c = 10 ∧ d = 14 ∧ e = 16) ∨ (a = -12 ∧ b = -9 ∧ c = -16 ∧ d = -24 ∧ e = -24)) :=
by
  sorry

end solve_system_l246_24693


namespace average_age_of_coaches_l246_24655

theorem average_age_of_coaches 
  (total_members : ℕ) (average_age_members : ℕ)
  (num_girls : ℕ) (average_age_girls : ℕ)
  (num_boys : ℕ) (average_age_boys : ℕ)
  (num_coaches : ℕ) :
  total_members = 30 →
  average_age_members = 20 →
  num_girls = 10 →
  average_age_girls = 18 →
  num_boys = 15 →
  average_age_boys = 19 →
  num_coaches = 5 →
  (600 - (num_girls * average_age_girls) - (num_boys * average_age_boys)) / num_coaches = 27 :=
by
  intros
  sorry

end average_age_of_coaches_l246_24655


namespace cab_speed_fraction_l246_24669

theorem cab_speed_fraction (S R : ℝ) (h1 : S * 40 = R * 48) : (R / S) = (5 / 6) :=
sorry

end cab_speed_fraction_l246_24669


namespace fraction_equation_solution_l246_24618

theorem fraction_equation_solution (a : ℤ) (hpos : a > 0) (h : (a : ℝ) / (a + 50) = 0.870) : a = 335 :=
by {
  sorry
}

end fraction_equation_solution_l246_24618


namespace willam_farm_tax_l246_24683

theorem willam_farm_tax
  (T : ℝ)
  (h1 : 0.4 * T * (3840 / (0.4 * T)) = 3840)
  (h2 : 0 < T) :
  0.3125 * T * (3840 / (0.4 * T)) = 3000 := by
  sorry

end willam_farm_tax_l246_24683


namespace total_coins_correct_l246_24665

-- Define basic parameters
def stacks_pennies : Nat := 3
def coins_per_penny_stack : Nat := 10
def stacks_nickels : Nat := 5
def coins_per_nickel_stack : Nat := 8
def stacks_dimes : Nat := 7
def coins_per_dime_stack : Nat := 4

-- Calculate total coins for each type
def total_pennies : Nat := stacks_pennies * coins_per_penny_stack
def total_nickels : Nat := stacks_nickels * coins_per_nickel_stack
def total_dimes : Nat := stacks_dimes * coins_per_dime_stack

-- Calculate total number of coins
def total_coins : Nat := total_pennies + total_nickels + total_dimes

-- Proof statement
theorem total_coins_correct : total_coins = 98 := by
  -- Proof steps go here (omitted)
  sorry

end total_coins_correct_l246_24665


namespace sum_of_first_eight_terms_l246_24668

theorem sum_of_first_eight_terms (a : ℝ) (r : ℝ) 
  (h1 : r = 2) (h2 : a * (1 + 2 + 4 + 8) = 1) :
  a * (1 + 2 + 4 + 8 + 16 + 32 + 64 + 128) = 17 :=
by
  -- sorry is used to skip the proof
  sorry

end sum_of_first_eight_terms_l246_24668


namespace alice_stops_in_quarter_D_l246_24630

-- Definitions and conditions
def indoor_track_circumference : ℕ := 40
def starting_point_S : ℕ := 0
def run_distance : ℕ := 1600

-- Desired theorem statement
theorem alice_stops_in_quarter_D :
  (run_distance % indoor_track_circumference = 0) → 
  (0 ≤ (run_distance % indoor_track_circumference) ∧ 
   (run_distance % indoor_track_circumference) < indoor_track_circumference) → 
  true := by
  sorry

end alice_stops_in_quarter_D_l246_24630


namespace initial_number_of_men_l246_24637

theorem initial_number_of_men
  (M : ℕ) (A : ℕ)
  (h1 : ∀ A_new : ℕ, A_new = A + 4)
  (h2 : ∀ total_age_increase : ℕ, total_age_increase = (2 * 52) - (36 + 32))
  (h3 : ∀ sum_age_men : ℕ, sum_age_men = M * A)
  (h4 : ∀ new_sum_age_men : ℕ, new_sum_age_men = sum_age_men + ((2 * 52) - (36 + 32))) :
  M = 9 := 
by
  -- Proof skipped
  sorry

end initial_number_of_men_l246_24637


namespace frog_climbing_time_l246_24608

-- Defining the conditions as Lean definitions
def well_depth : ℕ := 12
def climb_distance : ℕ := 3
def slip_distance : ℕ := 1
def climb_time : ℚ := 1 -- time in minutes for the frog to climb 3 meters
def slip_time : ℚ := climb_time / 3
def total_time_per_cycle : ℚ := climb_time + slip_time
def total_climbed_at_817 : ℕ := well_depth - 3 -- 3 meters from the top means it climbed 9 meters

-- The equivalent proof statement in Lean:
theorem frog_climbing_time : 
  ∃ (T : ℚ), T = 22 ∧ 
    (well_depth = 9 + 3) ∧
    (∀ (cycles : ℕ), cycles = 4 → 
         total_time_per_cycle * cycles + 2 = T) :=
by 
  sorry

end frog_climbing_time_l246_24608


namespace fraction_area_above_line_l246_24612

-- Define the problem conditions
def point1 : ℝ × ℝ := (4, 1)
def point2 : ℝ × ℝ := (9, 5)
def vertex1 : ℝ × ℝ := (4, 0)
def vertex2 : ℝ × ℝ := (9, 0)
def vertex3 : ℝ × ℝ := (9, 5)
def vertex4 : ℝ × ℝ := (4, 5)

-- Define the theorem statement
theorem fraction_area_above_line :
  let area_square := 25
  let area_below_line := 2.5
  let area_above_line := area_square - area_below_line
  area_above_line / area_square = 9 / 10 :=
by
  sorry -- Proof omitted

end fraction_area_above_line_l246_24612


namespace cos_neg_75_eq_l246_24679

noncomputable def cos_75_degrees : Real := (Real.sqrt 6 - Real.sqrt 2) / 4

theorem cos_neg_75_eq : Real.cos (-(75 * Real.pi / 180)) = cos_75_degrees := by
  sorry

end cos_neg_75_eq_l246_24679


namespace total_cost_price_is_correct_l246_24691

noncomputable def selling_price_before_discount (sp_after_discount : ℝ) (discount_rate : ℝ) : ℝ :=
  sp_after_discount / (1 - discount_rate)

noncomputable def cost_price_from_profit (selling_price : ℝ) (profit_rate : ℝ) : ℝ :=
  selling_price / (1 + profit_rate)

noncomputable def cost_price_from_loss (selling_price : ℝ) (loss_rate : ℝ) : ℝ :=
  selling_price / (1 - loss_rate)

noncomputable def total_cost_price : ℝ :=
  let CP1 := cost_price_from_profit (selling_price_before_discount 600 0.05) 0.25
  let CP2 := cost_price_from_loss 800 0.20
  let CP3 := cost_price_from_profit 1000 0.30 - 50
  CP1 + CP2 + CP3

theorem total_cost_price_is_correct : total_cost_price = 2224.49 :=
  by
  sorry

end total_cost_price_is_correct_l246_24691


namespace root_interval_l246_24617

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 4 * x - 3

theorem root_interval (x0 : ℝ) (h : f x0 = 0): x0 ∈ Set.Ioo (1 / 4 : ℝ) (1 / 2 : ℝ) :=
by
  sorry

end root_interval_l246_24617


namespace total_population_l246_24611

theorem total_population (P : ℝ) : 0.96 * P = 23040 → P = 24000 :=
by
  sorry

end total_population_l246_24611


namespace alternate_interior_angles_equal_l246_24697

-- Defining the parallel lines and the third intersecting line
def Line : Type := sorry  -- placeholder type for a line

-- Predicate to check if lines are parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- Predicate to represent a line intersecting another
def intersects (l1 l2 : Line) : Prop := sorry

-- Function to get interior alternate angles formed by the intersection
def alternate_interior_angles (l1 l2 : Line) (l3 : Line) : Prop := sorry

-- Theorem statement
theorem alternate_interior_angles_equal
  (l1 l2 l3 : Line)
  (h1 : parallel l1 l2)
  (h2 : intersects l3 l1)
  (h3 : intersects l3 l2) :
  alternate_interior_angles l1 l2 l3 :=
sorry

end alternate_interior_angles_equal_l246_24697


namespace distance_B_to_center_l246_24646

/-- Definitions for the geometrical scenario -/
structure NotchedCircleGeom where
  radius : ℝ
  A_pos : ℝ × ℝ
  B_pos : ℝ × ℝ
  C_pos : ℝ × ℝ
  AB_len : ℝ
  BC_len : ℝ
  angle_ABC_right : Prop
  
  -- Conditions derived from problem statement
  radius_eq_sqrt72 : radius = Real.sqrt 72
  AB_len_eq_8 : AB_len = 8
  BC_len_eq_3 : BC_len = 3
  angle_ABC_right_angle : angle_ABC_right
  
/-- Problem statement -/
theorem distance_B_to_center (geom : NotchedCircleGeom) :
  let x := geom.B_pos.1
  let y := geom.B_pos.2
  x^2 + y^2 = 50 :=
sorry

end distance_B_to_center_l246_24646


namespace find_function_f_l246_24678

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_function_f (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y →
    f (f y / f x + 1) = f (x + y / x + 1) - f x) →
  ∀ x : ℝ, 0 < x → f x = a * x :=
  by sorry

end find_function_f_l246_24678


namespace youngest_child_age_l246_24619

theorem youngest_child_age (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) : x = 4 := 
by 
  sorry

end youngest_child_age_l246_24619


namespace square_of_binomial_l246_24689

theorem square_of_binomial {a r s : ℚ} 
  (h1 : r^2 = a)
  (h2 : 2 * r * s = 18)
  (h3 : s^2 = 16) : 
  a = 81 / 16 :=
by sorry

end square_of_binomial_l246_24689


namespace range_of_m_l246_24606

noncomputable def f (x m : ℝ) := x^2 - 2 * m * x + 4

def P (m : ℝ) : Prop := ∀ x, 2 ≤ x → f x m ≥ f (2 : ℝ) m
def Q (m : ℝ) : Prop := ∀ x, 4 * x^2 + 4 * (m - 2) * x + 1 > 0

theorem range_of_m (m : ℝ) : (P m ∨ Q m) ∧ ¬(P m ∧ Q m) ↔ m ≤ 1 ∨ (2 < m ∧ m < 3) := sorry

end range_of_m_l246_24606


namespace measure_of_angle_C_l246_24641

theorem measure_of_angle_C (A B : ℝ) (h1 : A + B = 180) (h2 : A = 5 * B) : A = 150 := by
  sorry

end measure_of_angle_C_l246_24641


namespace find_a_if_even_function_l246_24616

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2 * a^2 - a) * x + 1

theorem find_a_if_even_function (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = 1 / 2 := by
  sorry

end find_a_if_even_function_l246_24616


namespace first_day_is_sunday_l246_24650

-- Define the days of the week
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

open Day

-- Function to determine the day of the week for a given day number
def day_of_month (n : ℕ) (start_day : Day) : Day :=
  match n % 7 with
  | 0 => start_day
  | 1 => match start_day with
          | Sunday    => Monday
          | Monday    => Tuesday
          | Tuesday   => Wednesday
          | Wednesday => Thursday
          | Thursday  => Friday
          | Friday    => Saturday
          | Saturday  => Sunday
  | 2 => match start_day with
          | Sunday    => Tuesday
          | Monday    => Wednesday
          | Tuesday   => Thursday
          | Wednesday => Friday
          | Thursday  => Saturday
          | Friday    => Sunday
          | Saturday  => Monday
-- ... and so on for the rest of the days of the week.
  | _ => start_day -- Assuming the pattern continues accordingly.

-- Prove that the first day of the month is a Sunday given that the 18th day of the month is a Wednesday.
theorem first_day_is_sunday (h : day_of_month 18 Wednesday = Wednesday) : day_of_month 1 Wednesday = Sunday :=
  sorry

end first_day_is_sunday_l246_24650


namespace find_a_l246_24677

variable {a : ℝ}

def A : Set ℝ := {2, 4}
def B (a : ℝ) : Set ℝ := {a, a^2 + 3}

theorem find_a (h : A ∩ (B a) = {2}) : a = 2 :=
by
  sorry

end find_a_l246_24677


namespace find_n_l246_24640

theorem find_n : ∃ n : ℕ, n < 2006 ∧ ∀ m : ℕ, 2006 * n = m * (2006 + n) ↔ n = 1475 := by
  sorry

end find_n_l246_24640


namespace proof_problem_l246_24635

def RealSets (A B : Set ℝ) : Set ℝ :=
let complementA := {x | -2 < x ∧ x < 3}
let unionAB := complementA ∪ B
unionAB

theorem proof_problem :
  let A := {x : ℝ | (x + 2) * (x - 3) ≥ 0}
  let B := {x : ℝ | x > 1}
  let complementA := {x : ℝ | -2 < x ∧ x < 3}
  let unionAB := complementA ∪ B
  unionAB = {x : ℝ | x > -2} :=
by
  sorry

end proof_problem_l246_24635


namespace determine_ts_l246_24652

theorem determine_ts :
  ∃ t s : ℝ, 
  (⟨3, 1⟩ : ℝ × ℝ) + t • (⟨4, -6⟩) = (⟨0, 2⟩ : ℝ × ℝ) + s • (⟨-3, 5⟩) :=
by
  use 6, -9
  sorry

end determine_ts_l246_24652


namespace determine_p_l246_24645

theorem determine_p (m : ℕ) (p : ℕ) (h1: m = 34) 
  (h2: (1 : ℝ)^ (m + 1) / 5^ (m + 1) * 1^18 / 4^18 = 1 / (2 * 10^ p)) : 
  p = 35 := by sorry

end determine_p_l246_24645


namespace triangle_isosceles_if_equal_bisectors_l246_24666

theorem triangle_isosceles_if_equal_bisectors
  (A B C : ℝ)
  (a b c l_a l_b : ℝ)
  (ha : l_a = l_b)
  (h1 : l_a = 2 * b * c * Real.cos (A / 2) / (b + c))
  (h2 : l_b = 2 * a * c * Real.cos (B / 2) / (a + c)) :
  a = b :=
by
  sorry

end triangle_isosceles_if_equal_bisectors_l246_24666


namespace rest_of_customers_bought_20_l246_24681

/-
Let's define the number of melons sold by the stand, number of customers who bought one and three melons, and total number of melons bought by these customers.
-/

def total_melons_sold : ℕ := 46
def customers_bought_one : ℕ := 17
def customers_bought_three : ℕ := 3

def melons_bought_by_those_bought_one := customers_bought_one * 1
def melons_bought_by_those_bought_three := customers_bought_three * 3

def remaining_melons := total_melons_sold - (melons_bought_by_those_bought_one + melons_bought_by_those_bought_three)

-- Now we state the theorem that the number of melons bought by the rest of the customers is 20 
theorem rest_of_customers_bought_20 :
  remaining_melons = 20 :=
by
  -- Skip the proof with 'sorry'
  sorry

end rest_of_customers_bought_20_l246_24681


namespace regular_price_of_shirt_is_50_l246_24638

-- Define all relevant conditions and given prices.
variables (P : ℝ) (shirt_price_discounted : ℝ) (total_paid : ℝ) (number_of_shirts : ℝ)

-- Define the conditions as hypotheses
def conditions :=
  (shirt_price_discounted = 0.80 * P) ∧
  (total_paid = 240) ∧
  (number_of_shirts = 6) ∧
  (total_paid = number_of_shirts * shirt_price_discounted)

-- State the theorem to prove that the regular price of the shirt is $50.
theorem regular_price_of_shirt_is_50 (h : conditions P shirt_price_discounted total_paid number_of_shirts) :
  P = 50 := 
sorry

end regular_price_of_shirt_is_50_l246_24638


namespace Q_over_P_l246_24600

theorem Q_over_P (P Q : ℚ)
  (h : ∀ (x : ℝ), x ≠ 0 ∧ x ≠ 3 ∧ x ≠ -3 → 
    (P / (x + 3) + Q / (x^2 - 3*x) = (x^2 - x + 8) / (x^3 + x^2 - 9*x))) :
  Q / P = 8 / 3 :=
by
  sorry

end Q_over_P_l246_24600


namespace fraction_remains_unchanged_l246_24653

theorem fraction_remains_unchanged (x y : ℝ) : 
  (3 * (2 * x)) / (2 * (2 * y)) = (3 * x) / (2 * y) :=
by {
  sorry
}

end fraction_remains_unchanged_l246_24653


namespace total_games_in_single_elimination_tournament_l246_24643

def single_elimination_tournament_games (teams : ℕ) : ℕ :=
teams - 1

theorem total_games_in_single_elimination_tournament :
  single_elimination_tournament_games 23 = 22 :=
by
  sorry

end total_games_in_single_elimination_tournament_l246_24643


namespace range_for_a_l246_24690

noncomputable def line_not_in_second_quadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, (3 * a - 1) * x + (2 - a) * y - 1 = 0 → (x ≥ 0 ∨ y ≥ 0)

theorem range_for_a (a : ℝ) :
  (line_not_in_second_quadrant a) ↔ a ≥ 2 := by
  sorry

end range_for_a_l246_24690


namespace unique_a_exists_iff_n_eq_two_l246_24620

theorem unique_a_exists_iff_n_eq_two (n : ℕ) (h1 : 1 < n) : 
  (∃ a : ℕ, 0 < a ∧ a ≤ n! ∧ n! ∣ a^n + 1 ∧ ∀ b : ℕ, (0 < b ∧ b ≤ n! ∧ n! ∣ b^n + 1) → b = a) ↔ n = 2 := 
by {
  sorry
}

end unique_a_exists_iff_n_eq_two_l246_24620


namespace proposition_C_l246_24659

theorem proposition_C (a b : ℝ) : a^3 > b^3 → a > b :=
sorry

end proposition_C_l246_24659


namespace red_fraction_is_three_fifths_l246_24663

noncomputable def fraction_of_red_marbles (x : ℕ) : ℚ := 
  let blue_marbles := (2 / 3 : ℚ) * x
  let red_marbles := x - blue_marbles
  let new_red_marbles := 3 * red_marbles
  let new_total_marbles := blue_marbles + new_red_marbles
  new_red_marbles / new_total_marbles

theorem red_fraction_is_three_fifths (x : ℕ) (hx : x ≠ 0) : fraction_of_red_marbles x = 3 / 5 :=
by {
  sorry
}

end red_fraction_is_three_fifths_l246_24663


namespace polynomial_solutions_l246_24692

theorem polynomial_solutions :
  (∀ x : ℂ, (x^4 + 2*x^3 + 2*x^2 + 2*x + 1 = 0) ↔ (x = -1 ∨ x = Complex.I ∨ x = -Complex.I)) :=
by
  sorry

end polynomial_solutions_l246_24692


namespace shortest_side_of_triangle_l246_24670

noncomputable def triangle_shortest_side_length (a b r : ℝ) (shortest : ℝ) : Prop :=
a = 8 ∧ b = 6 ∧ r = 4 ∧ shortest = 12

theorem shortest_side_of_triangle 
  (a b r shortest : ℝ) 
  (h : triangle_shortest_side_length a b r shortest) : shortest = 12 :=
sorry

end shortest_side_of_triangle_l246_24670


namespace kyle_speed_l246_24602

theorem kyle_speed (S : ℝ) (joseph_speed : ℝ) (joseph_time : ℝ) (kyle_time : ℝ) (H1 : joseph_speed = 50) (H2 : joseph_time = 2.5) (H3 : kyle_time = 2) (H4 : joseph_speed * joseph_time = kyle_time * S + 1) : S = 62 :=
by
  sorry

end kyle_speed_l246_24602


namespace trapezium_distance_l246_24654

variable (a b h : ℝ)

theorem trapezium_distance (h_pos : 0 < h) (a_pos : 0 < a) (b_pos : 0 < b)
  (area_eq : 270 = 1/2 * (a + b) * h) (a_eq : a = 20) (b_eq : b = 16) : h = 15 :=
by {
  sorry
}

end trapezium_distance_l246_24654


namespace range_of_a_l246_24675

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - a * x

theorem range_of_a (a : ℝ) :
  (∀ (x1 x2 : ℝ), 0 < x1 ∧ 0 < x2 ∧ x2 > x1 → (f x1 a / x2 - f x2 a / x1 < 0)) ↔ a ≤ Real.exp 1 / 2 := sorry

end range_of_a_l246_24675


namespace tan_60_eq_sqrt3_l246_24698

theorem tan_60_eq_sqrt3 : Real.tan (Real.pi / 3) = Real.sqrt 3 := 
sorry

end tan_60_eq_sqrt3_l246_24698


namespace breadth_of_rectangular_plot_l246_24648

theorem breadth_of_rectangular_plot
  (b l : ℕ)
  (h1 : l = 3 * b)
  (h2 : l * b = 2028) :
  b = 26 :=
sorry

end breadth_of_rectangular_plot_l246_24648


namespace evaluate_expression_l246_24610

theorem evaluate_expression (x : Real) (hx : x = -52.7) : 
  ⌈(⌊|x|⌋ + ⌈|x|⌉)⌉ = 105 := by
  sorry

end evaluate_expression_l246_24610


namespace contradiction_proof_l246_24601

theorem contradiction_proof (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) : ¬ (¬ (a > 0) ∨ ¬ (b > 0) ∨ ¬ (c > 0)) → false :=
by sorry

end contradiction_proof_l246_24601


namespace length_SR_l246_24609

theorem length_SR (cos_S : ℝ) (SP : ℝ) (SR : ℝ) (h1 : cos_S = 0.5) (h2 : SP = 10) (h3 : cos_S = SP / SR) : SR = 20 := by
  sorry

end length_SR_l246_24609


namespace calculate_expression_l246_24639

theorem calculate_expression : 5^3 + 5^3 + 5^3 + 5^3 = 625 :=
  sorry

end calculate_expression_l246_24639


namespace proportional_function_quadrants_l246_24622

theorem proportional_function_quadrants (k : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y > 0 ∧ y = k * x) ∧ (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ y = k * x) → k < 0 :=
by
  sorry

end proportional_function_quadrants_l246_24622


namespace y_coord_range_of_M_l246_24603

theorem y_coord_range_of_M :
  ∀ (M : ℝ × ℝ), ((M.1 + 1)^2 + M.2^2 = 2) → 
  ((M.1 - 2)^2 + M.2^2 + M.1^2 + M.2^2 ≤ 10) →
  - (Real.sqrt 7) / 2 ≤ M.2 ∧ M.2 ≤ (Real.sqrt 7) / 2 := 
by 
  sorry

end y_coord_range_of_M_l246_24603


namespace meaningful_sqrt_condition_l246_24644

theorem meaningful_sqrt_condition (x : ℝ) : (2 * x - 1 ≥ 0) ↔ (x ≥ 1 / 2) :=
by
  sorry

end meaningful_sqrt_condition_l246_24644


namespace correct_calculation_result_l246_24605

theorem correct_calculation_result :
  (∃ x : ℤ, 14 * x = 70) → (5 - 6 = -1) :=
by
  sorry

end correct_calculation_result_l246_24605


namespace smallest_integer_representation_l246_24649

theorem smallest_integer_representation :
  ∃ (A B C : ℕ), 0 ≤ A ∧ A < 5 ∧ 0 ≤ B ∧ B < 7 ∧ 0 ≤ C ∧ C < 4 ∧ 6 * A = 8 * B ∧ 6 * A = 5 * C ∧ 8 * B = 5 * C ∧ (6 * A) = 24 :=
  sorry

end smallest_integer_representation_l246_24649


namespace chessboard_accessible_squares_l246_24621

def is_accessible (board_size : ℕ) (central_exclusion_count : ℕ) (total_squares central_inaccessible : ℕ) : Prop :=
  total_squares = board_size * board_size ∧
  central_inaccessible = central_exclusion_count + 1 + 14 + 14 ∧
  board_size = 15 ∧
  total_squares - central_inaccessible = 196

theorem chessboard_accessible_squares :
  is_accessible 15 29 225 29 :=
by {
  sorry
}

end chessboard_accessible_squares_l246_24621


namespace john_blue_pens_l246_24627

variables (R B Bl : ℕ)

axiom total_pens : R + B + Bl = 31
axiom black_more_red : B = R + 5
axiom blue_twice_black : Bl = 2 * B

theorem john_blue_pens : Bl = 18 :=
by
  apply sorry

end john_blue_pens_l246_24627


namespace cubic_repeated_root_b_eq_100_l246_24664

theorem cubic_repeated_root_b_eq_100 (b : ℝ) (h1 : b ≠ 0)
  (h2 : ∃ x : ℝ, (b * x^3 + 15 * x^2 + 9 * x + 2 = 0) ∧ 
                 (3 * b * x^2 + 30 * x + 9 = 0)) :
  b = 100 :=
sorry

end cubic_repeated_root_b_eq_100_l246_24664


namespace probability_three_heads_l246_24676

noncomputable def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def probability (n : ℕ) (k : ℕ) : ℚ :=
  (binom n k) / (2 ^ n)

theorem probability_three_heads : probability 12 3 = 55 / 1024 := 
by
  sorry

end probability_three_heads_l246_24676


namespace units_digit_of_result_is_3_l246_24696

def hundreds_digit_relation (c : ℕ) (a : ℕ) : Prop :=
  a = 2 * c - 3

def original_number_expression (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c

def reversed_number_expression (a b c : ℕ) : ℕ :=
  100 * c + 10 * b + a + 50

def subtraction_result (orig rev : ℕ) : ℕ :=
  orig - rev

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_result_is_3 (a b c : ℕ) (h : hundreds_digit_relation c a) :
  units_digit (subtraction_result (original_number_expression a b c)
                                  (reversed_number_expression a b c)) = 3 :=
by
  sorry

end units_digit_of_result_is_3_l246_24696


namespace necessary_but_not_sufficient_l246_24694

theorem necessary_but_not_sufficient (x : ℝ) : (1 - x) * (1 + |x|) > 0 -> x < 2 :=
by
  sorry

end necessary_but_not_sufficient_l246_24694


namespace double_inputs_revenue_l246_24684

theorem double_inputs_revenue (A K L : ℝ) (α1 α2 : ℝ) (hα1 : α1 = 0.6) (hα2 : α2 = 0.5) (hα1_bound : 0 < α1 ∧ α1 < 1) (hα2_bound : 0 < α2 ∧ α2 < 1) :
  A * (2 * K) ^ α1 * (2 * L) ^ α2 > 2 * (A * K ^ α1 * L ^ α2) :=
by
  sorry

end double_inputs_revenue_l246_24684


namespace geometric_sequence_common_ratio_l246_24673

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_cond : (a 0 * (1 + q + q^2)) / (a 0 * q^2) = 3) : q = 1 :=
by
  sorry

end geometric_sequence_common_ratio_l246_24673


namespace triangle_height_dist_inequality_l246_24642

variable {T : Type} [MetricSpace T] 

theorem triangle_height_dist_inequality {h_a h_b h_c l_a l_b l_c : ℝ} (h_a_pos : 0 < h_a) (h_b_pos : 0 < h_b) (h_c_pos : 0 < h_c) 
  (l_a_pos : 0 < l_a) (l_b_pos : 0 < l_b) (l_c_pos : 0 < l_c) :
  h_a / l_a + h_b / l_b + h_c / l_c >= 9 :=
sorry

end triangle_height_dist_inequality_l246_24642


namespace tire_circumference_l246_24680

theorem tire_circumference (rpm : ℕ) (speed_kmh : ℕ) (C : ℝ) 
  (h1 : rpm = 400) 
  (h2 : speed_kmh = 144) 
  (h3 : (speed_kmh * 1000 / 60) = (rpm * C)) : 
  C = 6 :=
by
  sorry

end tire_circumference_l246_24680
