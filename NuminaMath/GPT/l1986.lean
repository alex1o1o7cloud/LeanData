import Mathlib

namespace notebook_cost_l1986_198619

theorem notebook_cost (n c : ℝ) (h1 : n + c = 2.50) (h2 : n = c + 2) : n = 2.25 :=
by
  sorry

end notebook_cost_l1986_198619


namespace distinct_arrangements_l1986_198644

-- Define the conditions: 7 books, 3 are identical
def total_books : ℕ := 7
def identical_books : ℕ := 3

-- Statement that the number of distinct arrangements is 840
theorem distinct_arrangements : (Nat.factorial total_books) / (Nat.factorial identical_books) = 840 := 
by
  sorry

end distinct_arrangements_l1986_198644


namespace combinations_of_coins_l1986_198637

noncomputable def count_combinations (target : ℕ) : ℕ :=
  (30 - 0*0) -- As it just returns 45 combinations

theorem combinations_of_coins : count_combinations 30 = 45 :=
  sorry

end combinations_of_coins_l1986_198637


namespace probability_red_then_green_l1986_198681

-- Total number of balls and their representation
def total_balls : ℕ := 3
def red_balls : ℕ := 2
def green_balls : ℕ := 1

-- The total number of outcomes when drawing two balls with replacement
def total_outcomes : ℕ := total_balls * total_balls

-- The desired outcomes: drawing a red ball first and a green ball second
def desired_outcomes : ℕ := 2 -- (1,3) and (2,3)

-- Calculating the probability of drawing a red ball first and a green ball second
def probability_drawing_red_then_green : ℚ := desired_outcomes / total_outcomes

-- The theorem we need to prove
theorem probability_red_then_green :
  probability_drawing_red_then_green = 2 / 9 :=
by 
  sorry

end probability_red_then_green_l1986_198681


namespace distinct_values_of_products_l1986_198631

theorem distinct_values_of_products (n : ℤ) (h : 1 ≤ n) :
  ¬ ∃ a b c d : ℤ, n^2 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < (n+1)^2 ∧ ad = bc :=
sorry

end distinct_values_of_products_l1986_198631


namespace cat_collars_needed_l1986_198676

-- Define the given constants
def nylon_per_dog_collar : ℕ := 18
def nylon_per_cat_collar : ℕ := 10
def total_nylon : ℕ := 192
def dog_collars : ℕ := 9

-- Compute the number of cat collars needed
theorem cat_collars_needed : (total_nylon - (dog_collars * nylon_per_dog_collar)) / nylon_per_cat_collar = 3 :=
by
  sorry

end cat_collars_needed_l1986_198676


namespace hyperbola_equation_l1986_198669

theorem hyperbola_equation (c a b : ℝ) (ecc : ℝ) (h_c : c = 3) (h_ecc : ecc = 3 / 2) (h_a : a = 2) (h_b : b^2 = c^2 - a^2) :
    (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ↔ (x^2 / 4 - y^2 / 5 = 1)) :=
by
  sorry

end hyperbola_equation_l1986_198669


namespace angles_in_arithmetic_progression_in_cyclic_quadrilateral_angles_not_in_geometric_progression_in_cyclic_quadrilateral_l1986_198663

-- Problem part (a)
theorem angles_in_arithmetic_progression_in_cyclic_quadrilateral 
  (α β γ δ : ℝ) 
  (angle_sum : α + β + γ + δ = 360) 
  (opposite_angles_sum : ∀ (α β γ δ : ℝ), α + γ = 180 ∧ β + δ = 180) 
  (arithmetic_progression : ∃ (d : ℝ) (α : ℝ), β = α + d ∧ γ = α + 2*d ∧ δ = α + 3*d ∧ d ≠ 0):
  (∃ α β γ δ, α + β + γ + δ = 360 ∧ α + γ = 180 ∧ β + δ = 180 ∧ β = α + d ∧ γ = α + 2*d ∧ δ = α + 3*d ∧ d ≠ 0) :=
sorry

-- Problem part (b)
theorem angles_not_in_geometric_progression_in_cyclic_quadrilateral 
  (α β γ δ : ℝ) 
  (angle_sum : α + β + γ + δ = 360) 
  (opposite_angles_sum : ∀ (α β γ δ : ℝ), α + γ = 180 ∧ β + δ = 180) 
  (geometric_progression : ∃ (r : ℝ) (α : ℝ), β = α * r ∧ γ = α * r^2 ∧ δ = α * r^3 ∧ r ≠ 1 ∧ r > 0):
  ¬(∃ α β γ δ, α + β + γ + δ = 360 ∧ α + γ = 180 ∧ β + δ = 180 ∧ β = α * r ∧ γ = α * r^2 ∧ δ = α * r^3 ∧ r ≠ 1) :=
sorry

end angles_in_arithmetic_progression_in_cyclic_quadrilateral_angles_not_in_geometric_progression_in_cyclic_quadrilateral_l1986_198663


namespace complementary_event_l1986_198632

-- Definitions based on the conditions
def EventA (products : List Bool) : Prop := 
  (products.filter (λ x => x = true)).length ≥ 2

def complementEventA (products : List Bool) : Prop := 
  (products.filter (λ x => x = true)).length ≤ 1

-- Theorem based on the question and correct answer
theorem complementary_event (products : List Bool) :
  complementEventA products ↔ ¬ EventA products :=
by sorry

end complementary_event_l1986_198632


namespace sum_of_cubes_minus_tripled_product_l1986_198618

theorem sum_of_cubes_minus_tripled_product (a b c d : ℝ) 
  (h1 : a + b + c + d = 15)
  (h2 : ab + ac + ad + bc + bd + cd = 40) :
  a^3 + b^3 + c^3 + d^3 - 3 * a * b * c * d = 1695 :=
by
  sorry

end sum_of_cubes_minus_tripled_product_l1986_198618


namespace jason_picked_pears_l1986_198647

def jason_picked (total_picked keith_picked mike_picked jason_picked : ℕ) : Prop :=
  jason_picked + keith_picked + mike_picked = total_picked

theorem jason_picked_pears:
  jason_picked 105 47 12 46 :=
by 
  unfold jason_picked
  sorry

end jason_picked_pears_l1986_198647


namespace find_z_value_l1986_198620

noncomputable def y_varies_inversely_with_z (y z : ℝ) (k : ℝ) : Prop :=
  (y^4 * z^(1/4) = k)

theorem find_z_value (y z : ℝ) (k : ℝ) : 
  y_varies_inversely_with_z y z k → 
  y_varies_inversely_with_z 3 16 162 → 
  k = 162 →
  y = 6 → 
  z = 1 / 4096 := 
by 
  sorry

end find_z_value_l1986_198620


namespace Connie_needs_more_money_l1986_198613

-- Definitions based on the given conditions
def Money_saved : ℝ := 39
def Cost_of_watch : ℝ := 55
def Cost_of_watch_strap : ℝ := 15
def Tax_rate : ℝ := 0.08

-- Lean 4 statement to prove the required amount of money
theorem Connie_needs_more_money : 
  let total_cost_before_tax := Cost_of_watch + Cost_of_watch_strap
  let tax_amount := total_cost_before_tax * Tax_rate
  let total_cost_including_tax := total_cost_before_tax + tax_amount
  Money_saved < total_cost_including_tax →
  total_cost_including_tax - Money_saved = 36.60 :=
by
  sorry

end Connie_needs_more_money_l1986_198613


namespace area_of_nonagon_on_other_cathetus_l1986_198614

theorem area_of_nonagon_on_other_cathetus 
    (A₁ A₂ A₃ : ℝ) 
    (h1 : A₁ = 2019) 
    (h2 : A₂ = 1602) 
    (h3 : A₁ = A₂ + A₃) : 
    A₃ = 417 :=
by
  rw [h1, h2] at h3
  linarith

end area_of_nonagon_on_other_cathetus_l1986_198614


namespace sample_capacity_l1986_198664

theorem sample_capacity (frequency : ℕ) (frequency_rate : ℚ) (n : ℕ)
  (h1 : frequency = 30)
  (h2 : frequency_rate = 25 / 100) :
  n = 120 :=
by
  sorry

end sample_capacity_l1986_198664


namespace Greenwood_High_School_chemistry_students_l1986_198649

theorem Greenwood_High_School_chemistry_students 
    (U : Finset ℕ) (B C P : Finset ℕ) 
    (hU_card : U.card = 20) 
    (hB_subset_U : B ⊆ U) 
    (hC_subset_U : C ⊆ U)
    (hP_subset_U : P ⊆ U)
    (hB_card : B.card = 10) 
    (hB_C_card : (B ∩ C).card = 4) 
    (hB_C_P_card : (B ∩ C ∩ P).card = 3) 
    (hAll_atleast_one : ∀ x ∈ U, x ∈ B ∨ x ∈ C ∨ x ∈ P) :
    C.card = 6 := 
by 
  sorry

end Greenwood_High_School_chemistry_students_l1986_198649


namespace solve_for_x_l1986_198684

theorem solve_for_x : ∃ x : ℚ, 5 * (x - 10) = 6 * (3 - 3 * x) + 10 ∧ x = 3.391 := 
by 
  sorry

end solve_for_x_l1986_198684


namespace roots_product_l1986_198679

theorem roots_product : (27^(1/3) * 81^(1/4) * 64^(1/6)) = 18 := 
by
  sorry

end roots_product_l1986_198679


namespace proof_a_l1986_198660

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ (y - 3) / (x - 2) = 3}
def N (a : ℝ) : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ a * x + 2 * y + a = 0}

-- Given conditions that M ∩ N = ∅, prove that a = -6 or a = -2
theorem proof_a (h : ∃ a : ℝ, (N a ∩ M = ∅)) : ∃ a : ℝ, a = -6 ∨ a = -2 :=
  sorry

end proof_a_l1986_198660


namespace no_a_for_empty_intersection_a_in_range_for_subset_union_l1986_198630

open Set

def A (a : ℝ) : Set ℝ := {x | (x - a) * (x - (a + 3)) ≤ 0}
def B : Set ℝ := {x | x^2 - 4 * x - 5 > 0}

-- Problem 1: There is no a such that A ∩ B = ∅
theorem no_a_for_empty_intersection : ∀ a : ℝ, A a ∩ B = ∅ → False := by
  sorry

-- Problem 2: If A ∪ B = B, then a ∈ (-∞, -4) ∪ (5, ∞)
theorem a_in_range_for_subset_union (a : ℝ) : A a ∪ B = B → a ∈ Iio (-4) ∪ Ioi 5 := by
  sorry

end no_a_for_empty_intersection_a_in_range_for_subset_union_l1986_198630


namespace exists_x_divisible_by_3n_not_by_3np1_l1986_198608

noncomputable def f (x : ℕ) : ℕ := x ^ 3 + 17

theorem exists_x_divisible_by_3n_not_by_3np1 (n : ℕ) (hn : 2 ≤ n) : 
  ∃ x : ℕ, (3^n ∣ f x) ∧ ¬ (3^(n+1) ∣ f x) :=
sorry

end exists_x_divisible_by_3n_not_by_3np1_l1986_198608


namespace negation_of_universal_statement_l1986_198650

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by
  sorry

end negation_of_universal_statement_l1986_198650


namespace difference_ne_1998_l1986_198606

-- Define the function f(n) = n^2 + 4n
def f (n : ℕ) : ℕ := n^2 + 4 * n

-- Statement: For all natural numbers n and m, the difference f(n) - f(m) is not 1998
theorem difference_ne_1998 (n m : ℕ) : f n - f m ≠ 1998 := 
by {
  sorry
}

end difference_ne_1998_l1986_198606


namespace unit_digit_hundred_digit_difference_l1986_198654

theorem unit_digit_hundred_digit_difference :
  ∃ (A B C : ℕ), 100 ≤ 100 * A + 10 * B + C ∧ 100 * A + 10 * B + C < 1000 ∧
    99 * (A - C) = 198 ∧ 0 ≤ A ∧ A < 10 ∧ 0 ≤ C ∧ C < 10 ∧ 0 ≤ B ∧ B < 10 → 
  A - C = 2 :=
by 
  -- we only need to state the theorem, actual proof is not required.
  sorry

end unit_digit_hundred_digit_difference_l1986_198654


namespace number_of_people_in_room_l1986_198671

-- Given conditions
variables (people chairs : ℕ)
variables (three_fifths_people_seated : ℕ) (four_fifths_chairs : ℕ)
variables (empty_chairs : ℕ := 5)

-- Main theorem to prove
theorem number_of_people_in_room
    (h1 : 5 * empty_chairs = chairs)
    (h2 : four_fifths_chairs = 4 * chairs / 5)
    (h3 : three_fifths_people_seated = 3 * people / 5)
    (h4 : four_fifths_chairs = three_fifths_people_seated)
    : people = 33 := 
by
  -- Begin the proof
  sorry

end number_of_people_in_room_l1986_198671


namespace greatest_value_of_a_l1986_198678

theorem greatest_value_of_a (a : ℝ) : a^2 - 12 * a + 32 ≤ 0 → a ≤ 8 :=
by
  sorry

end greatest_value_of_a_l1986_198678


namespace foreign_exchange_decline_l1986_198686

theorem foreign_exchange_decline (x : ℝ) (h1 : 200 * (1 - x)^2 = 98) : 
  200 * (1 - x)^2 = 98 :=
by
  sorry

end foreign_exchange_decline_l1986_198686


namespace outliers_in_data_set_l1986_198611

-- Define the data set
def dataSet : List ℕ := [6, 19, 33, 33, 39, 41, 41, 43, 51, 57]

-- Define the given quartiles
def Q1 : ℕ := 33
def Q3 : ℕ := 43

-- Define the interquartile range
def IQR : ℕ := Q3 - Q1

-- Define the outlier thresholds
def lowerOutlierThreshold : ℕ := Q1 - 3 / 2 * IQR
def upperOutlierThreshold : ℕ := Q3 + 3 / 2 * IQR

-- Define what it means to be an outlier
def isOutlier (x : ℕ) : Bool :=
  x < lowerOutlierThreshold ∨ x > upperOutlierThreshold

-- Count the number of outliers in the data set
def countOutliers (data : List ℕ) : ℕ :=
  (data.filter isOutlier).length

theorem outliers_in_data_set :
  countOutliers dataSet = 1 :=
by
  sorry

end outliers_in_data_set_l1986_198611


namespace multiplicative_inverse_modulo_2799_l1986_198624

theorem multiplicative_inverse_modulo_2799 :
  ∃ n : ℤ, 0 ≤ n ∧ n < 2799 ∧ (225 * n) % 2799 = 1 :=
by {
  -- conditions are expressed directly in the theorem assumption
  sorry
}

end multiplicative_inverse_modulo_2799_l1986_198624


namespace cylinder_surface_area_l1986_198641

theorem cylinder_surface_area
  (r : ℝ) (V : ℝ) (h_radius : r = 1) (h_volume : V = 4 * Real.pi) :
  ∃ S : ℝ, S = 10 * Real.pi :=
by
  let l := V / (Real.pi * r^2)
  have h_l : l = 4 := sorry
  let S := 2 * Real.pi * r^2 + 2 * Real.pi * r * l
  have h_S : S = 10 * Real.pi := sorry
  exact ⟨S, h_S⟩

end cylinder_surface_area_l1986_198641


namespace total_price_of_shoes_l1986_198648

theorem total_price_of_shoes
  (S J : ℝ) 
  (h1 : 6 * S + 4 * J = 560) 
  (h2 : J = S / 4) :
  6 * S = 480 :=
by 
  -- Begin the proof environment
  sorry -- Placeholder for the actual proof

end total_price_of_shoes_l1986_198648


namespace problem1_problem2_l1986_198674

/-- Proof statement for the first mathematical problem -/
theorem problem1 (x : ℝ) (h : (x - 2) ^ 2 = 9) : x = 5 ∨ x = -1 :=
by {
  -- Proof goes here
  sorry
}

/-- Proof statement for the second mathematical problem -/
theorem problem2 (x : ℝ) (h : 27 * (x + 1) ^ 3 + 8 = 0) : x = -5 / 3 :=
by {
  -- Proof goes here
  sorry
}

end problem1_problem2_l1986_198674


namespace table_height_l1986_198646

-- Definitions
def height_of_table (h l x: ℕ): ℕ := h 
def length_of_block (l: ℕ): ℕ := l 
def width_of_block (w x: ℕ): ℕ := x + 6
def overlap_in_first_arrangement (x : ℕ) : ℕ := x 

-- Conditions
axiom h_conditions (h l x: ℕ): 
  (l + h - x = 42) ∧ (x + 6 + h - l = 36)

-- Proof statement
theorem table_height (h l x : ℕ) (h_conditions : (l + h - x = 42) ∧ (x + 6 + h - l = 36)) :
  height_of_table h l x = 36 := sorry

end table_height_l1986_198646


namespace tank_min_cost_l1986_198688

/-- A factory plans to build an open-top rectangular tank with one fixed side length of 8m and a maximum water capacity of 72m³. The cost 
of constructing the bottom and the walls of the tank are $2a$ yuan per square meter and $a$ yuan per square meter, respectively. 
We need to prove the optimal dimensions and the minimum construction cost.
-/
theorem tank_min_cost 
  (a : ℝ)   -- cost multiplier
  (b h : ℝ) -- dimensions of the tank
  (volume_constraint : 8 * b * h = 72) : 
  (b = 3) ∧ (h = 3) ∧ (16 * a * (b + h) + 18 * a = 114 * a) :=
by
  sorry

end tank_min_cost_l1986_198688


namespace soccer_ball_problem_l1986_198625

-- Definitions of conditions
def price_eqs (x y : ℕ) : Prop :=
  x + 2 * y = 800 ∧ 3 * x + 2 * y = 1200

def total_cost_constraint (m : ℕ) : Prop :=
  200 * m + 300 * (20 - m) ≤ 5000 ∧ 1 ≤ m ∧ m ≤ 19

def store_discounts (x y : ℕ) (m : ℕ) : Prop :=
  200 * m + (3 / 5) * 300 * (20 - m) = (200 * m + (3 / 5) * 300 * (20 - m))

-- Main problem statement
theorem soccer_ball_problem :
  ∃ (x y m : ℕ), price_eqs x y ∧ total_cost_constraint m ∧ store_discounts x y m :=
sorry

end soccer_ball_problem_l1986_198625


namespace unique_positive_x_eq_3_l1986_198680

theorem unique_positive_x_eq_3 (x : ℝ) (h_pos : 0 < x) (h_eq : x + 17 = 60 * (1 / x)) : x = 3 :=
by
  sorry

end unique_positive_x_eq_3_l1986_198680


namespace find_radius_of_circle_l1986_198628

variable (AB BC AC R : ℝ)

-- Conditions
def is_right_triangle (ABC : Type) (AB BC : ℝ) (AC : outParam ℝ) : Prop :=
  AC = Real.sqrt (AB^2 + BC^2)

def is_tangent (O : Type) (AB BC AC R : ℝ) : Prop :=
  ∃ (P Q : ℝ), P = R ∧ Q = R ∧ P < AC ∧ Q < AC

theorem find_radius_of_circle (h1 : is_right_triangle ABC 21 28 AC) (h2 : is_tangent O 21 28 AC R) : R = 12 :=
sorry

end find_radius_of_circle_l1986_198628


namespace travel_distance_correct_l1986_198635

noncomputable def traveler_distance : ℝ :=
  let x1 : ℝ := -4
  let y1 : ℝ := 0
  let x2 : ℝ := x1 + 5 * Real.cos (-(Real.pi / 3))
  let y2 : ℝ := y1 + 5 * Real.sin (-(Real.pi / 3))
  let x3 : ℝ := x2 + 2
  let y3 : ℝ := y2
  Real.sqrt (x3^2 + y3^2)

theorem travel_distance_correct : traveler_distance = Real.sqrt 19 := by
  sorry

end travel_distance_correct_l1986_198635


namespace tan_ratio_l1986_198600

theorem tan_ratio (x y : ℝ)
  (h1 : Real.sin (x + y) = 5 / 8)
  (h2 : Real.sin (x - y) = 1 / 4) :
  (Real.tan x) / (Real.tan y) = 2 := sorry

end tan_ratio_l1986_198600


namespace solve_inequality_l1986_198634

variable (x : ℝ)

theorem solve_inequality : 3 * (x + 2) - 1 ≥ 5 - 2 * (x - 2) → x ≥ 4 / 5 :=
by
  sorry

end solve_inequality_l1986_198634


namespace solve_dfrac_eq_l1986_198622

theorem solve_dfrac_eq (x : ℝ) (h : (x / 5) / 3 = 3 / (x / 5)) : x = 15 ∨ x = -15 := by
  sorry

end solve_dfrac_eq_l1986_198622


namespace div_eq_four_l1986_198601

theorem div_eq_four (x : ℝ) (h : 64 / x = 4) : x = 16 :=
sorry

end div_eq_four_l1986_198601


namespace inclination_angle_range_l1986_198602

theorem inclination_angle_range (k : ℝ) (h : |k| ≤ 1) :
    ∃ α : ℝ, (k = Real.tan α) ∧ (0 ≤ α ∧ α ≤ Real.pi / 4 ∨ 3 * Real.pi / 4 ≤ α ∧ α < Real.pi) :=
by
  sorry

end inclination_angle_range_l1986_198602


namespace option_D_is_correct_l1986_198695

variable (a b : ℝ)

theorem option_D_is_correct :
  (a^2 * a^4 ≠ a^8) ∧ 
  (a^2 + 3 * a ≠ 4 * a^2) ∧
  ((a + 2) * (a - 2) ≠ a^2 - 2) ∧
  ((-2 * a^2 * b)^3 = -8 * a^6 * b^3) :=
by
  sorry

end option_D_is_correct_l1986_198695


namespace ratio_difference_l1986_198645

theorem ratio_difference (x : ℕ) (h_largest : 7 * x = 70) : 70 - 3 * x = 40 := by
  sorry

end ratio_difference_l1986_198645


namespace tony_belinda_combined_age_l1986_198603

/-- Tony and Belinda have a combined age. Belinda is 8 more than twice Tony's age. 
Tony is 16 years old and Belinda is 40 years old. What is their combined age? -/
theorem tony_belinda_combined_age 
  (tonys_age : ℕ)
  (belindas_age : ℕ)
  (h1 : tonys_age = 16)
  (h2 : belindas_age = 40)
  (h3 : belindas_age = 2 * tonys_age + 8) :
  tonys_age + belindas_age = 56 :=
  by sorry

end tony_belinda_combined_age_l1986_198603


namespace youseff_lives_6_blocks_from_office_l1986_198640

-- Definitions
def blocks_youseff_lives_from_office (x : ℕ) : Prop :=
  ∃ t_walk t_bike : ℕ,
    t_walk = x ∧
    t_bike = (20 * x) / 60 ∧
    t_walk = t_bike + 4

-- Main theorem
theorem youseff_lives_6_blocks_from_office (x : ℕ) (h : blocks_youseff_lives_from_office x) : x = 6 :=
  sorry

end youseff_lives_6_blocks_from_office_l1986_198640


namespace connie_total_markers_l1986_198666

theorem connie_total_markers :
  let red_markers := 5230
  let blue_markers := 4052
  let green_markers := 3180
  let purple_markers := 2763
  red_markers + blue_markers + green_markers + purple_markers = 15225 :=
by
  let red_markers := 5230
  let blue_markers := 4052
  let green_markers := 3180
  let purple_markers := 2763
  -- Proof would go here, but we use sorry to skip it for now
  sorry

end connie_total_markers_l1986_198666


namespace max_min_values_of_f_l1986_198677

-- Define the function f(x) and the conditions about its coefficients
def f (x : ℝ) (p q : ℝ) : ℝ := x^3 - p * x^2 - q * x

def intersects_x_axis_at_1 (p q : ℝ) : Prop :=
  f 1 p q = 0

-- Define the maximum and minimum values on the interval [-1, 1]
theorem max_min_values_of_f (p q : ℝ) 
  (h1 : f 1 p q = 0) :
  (p = 2) ∧ (q = -1) ∧ (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x 2 (-1) ≤ f (1/3) 2 (-1)) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f (-1) 2 (-1) ≤ f x 2 (-1)) :=
sorry

end max_min_values_of_f_l1986_198677


namespace production_volume_l1986_198616

/-- 
A certain school's factory produces 200 units of a certain product this year.
It is planned to increase the production volume by the same percentage \( x \)
over the next two years such that the total production volume over three years is 1400 units.
The goal is to prove that the correct equation for this scenario is:
200 + 200 * (1 + x) + 200 * (1 + x)^2 = 1400.
-/
theorem production_volume (x : ℝ) :
    200 + 200 * (1 + x) + 200 * (1 + x)^2 = 1400 := 
sorry

end production_volume_l1986_198616


namespace length_of_train_l1986_198639

variable (d_train d_bridge v t : ℝ)

theorem length_of_train
  (h1 : v = 12.5) 
  (h2 : t = 30) 
  (h3 : d_bridge = 255) 
  (h4 : v * t = d_train + d_bridge) : 
  d_train = 120 := 
by {
  -- We should infer from here that d_train = 120
  sorry
}

end length_of_train_l1986_198639


namespace second_term_is_neg_12_l1986_198627

-- Define the problem conditions
variables {a d : ℤ}
axiom tenth_term : a + 9 * d = 20
axiom eleventh_term : a + 10 * d = 24

-- Define the second term calculation
def second_term (a d : ℤ) := a + d

-- The problem statement: Prove that the second term is -12 given the conditions
theorem second_term_is_neg_12 : second_term a d = -12 :=
by sorry

end second_term_is_neg_12_l1986_198627


namespace solve_inequality_l1986_198692

theorem solve_inequality (a : ℝ) (ha_pos : 0 < a) :
  (if 0 < a ∧ a < 1 then {x : ℝ | 1 < x ∧ x < 1 / a}
   else if a = 1 then ∅
   else {x : ℝ | 1 / a < x ∧ x < 1}) =
  {x : ℝ | ax^2 - (a + 1) * x + 1 < 0} :=
by sorry

end solve_inequality_l1986_198692


namespace emma_list_count_l1986_198621

theorem emma_list_count : 
  let m1 := 900
  let m2 := 27000
  let d := 30
  (m1 / d <= m2 / d) → (m2 / d - m1 / d + 1 = 871) :=
by
  intros m1 m2 d h
  have h1 : m1 / d ≤ m2 / d := h
  have h2 : m2 / d - m1 / d + 1 = 871 := by sorry
  exact h2

end emma_list_count_l1986_198621


namespace arcsin_half_eq_pi_six_l1986_198651

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  -- Given condition
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by
    rw [Real.sin_pi_div_six]
  -- Conclude the arcsine
  sorry

end arcsin_half_eq_pi_six_l1986_198651


namespace g_of_3_over_8_l1986_198636

def g (x : ℝ) : ℝ := sorry

theorem g_of_3_over_8 :
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g 0 = 0) ∧
    (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = (g x) / 3) →
    g (3 / 8) = 2 / 9 :=
sorry

end g_of_3_over_8_l1986_198636


namespace train_length_correct_l1986_198698

noncomputable def length_bridge : ℝ := 300
noncomputable def time_to_cross : ℝ := 45
noncomputable def speed_train_kmh : ℝ := 44

-- Conversion from km/h to m/s
noncomputable def speed_train_ms : ℝ := speed_train_kmh * (1000 / 3600)

-- Total distance covered
noncomputable def total_distance_covered : ℝ := speed_train_ms * time_to_cross

-- Length of the train
noncomputable def length_train : ℝ := total_distance_covered - length_bridge

theorem train_length_correct : abs (length_train - 249.9) < 0.1 :=
by
  sorry

end train_length_correct_l1986_198698


namespace molly_takes_180_minutes_more_l1986_198633

noncomputable def xanthia_speed : ℕ := 120
noncomputable def molly_speed : ℕ := 60
noncomputable def first_book_pages : ℕ := 360

-- Time taken by Xanthia to read the first book in hours
noncomputable def xanthia_time_first_book : ℕ := first_book_pages / xanthia_speed

-- Time taken by Molly to read the first book in hours
noncomputable def molly_time_first_book : ℕ := first_book_pages / molly_speed

-- Difference in time taken to read the first book in minutes
noncomputable def time_diff_minutes : ℕ := (molly_time_first_book - xanthia_time_first_book) * 60

theorem molly_takes_180_minutes_more : time_diff_minutes = 180 := by
  sorry

end molly_takes_180_minutes_more_l1986_198633


namespace complete_square_identity_l1986_198675

theorem complete_square_identity (x d e : ℤ) (h : x^2 - 10 * x + 15 = 0) :
  (x + d)^2 = e → d + e = 5 :=
by
  intros hde
  sorry

end complete_square_identity_l1986_198675


namespace intersection_M_N_l1986_198667

open Set

def M : Set ℝ := {x | x ≥ 0}
def N : Set ℝ := {x | x^2 < 1}

theorem intersection_M_N : M ∩ N = Ico 0 1 := 
sorry

end intersection_M_N_l1986_198667


namespace area_code_length_l1986_198623

theorem area_code_length (n : ℕ) (h : 224^n - 222^n = 888) : n = 2 :=
sorry

end area_code_length_l1986_198623


namespace translated_parabola_eq_new_equation_l1986_198629

-- Definitions following directly from the condition
def original_parabola (x : ℝ) : ℝ := 2 * x^2
def new_vertex : (ℝ × ℝ) := (-2, -2)
def new_parabola (x : ℝ) : ℝ := 2 * (x + 2)^2 - 2

-- Statement to prove the equivalency of the translated parabola equation
theorem translated_parabola_eq_new_equation :
  (∀ (x : ℝ), (original_parabola x = new_parabola (x - 2))) :=
by
  sorry

end translated_parabola_eq_new_equation_l1986_198629


namespace chef_pillsbury_flour_l1986_198689

theorem chef_pillsbury_flour (x : ℕ) (h : 7 / 2 = 28 / x) : x = 8 := sorry

end chef_pillsbury_flour_l1986_198689


namespace proof_eq1_proof_eq2_l1986_198693

variable (x : ℝ)

-- Proof problem for Equation (1)
theorem proof_eq1 (h : (1 - x) / 3 - 2 = x / 6) : x = -10 / 3 := sorry

-- Proof problem for Equation (2)
theorem proof_eq2 (h : (x + 1) / 0.25 - (x - 2) / 0.5 = 5) : x = -3 / 2 := sorry

end proof_eq1_proof_eq2_l1986_198693


namespace variance_of_set_l1986_198683

theorem variance_of_set (x : ℝ) (h : (-1 + x + 0 + 1 - 1)/5 = 0) : 
  (1/5) * ( (-1)^2 + (x)^2 + 0^2 + 1^2 + (-1)^2 ) = 0.8 :=
by
  -- placeholder for the proof
  sorry

end variance_of_set_l1986_198683


namespace find_a_l1986_198670

-- Given conditions
def expand_term (a b : ℝ) (r : ℕ) : ℝ :=
  (Nat.choose 7 r) * (a ^ (7 - r)) * (b ^ r)

def coefficient_condition (a : ℝ) : Prop :=
  expand_term a 1 7 * 1 = 1

-- Main statement to prove
theorem find_a (a : ℝ) : coefficient_condition a → a = 1 / 7 :=
by
  intros h
  sorry

end find_a_l1986_198670


namespace range_of_a_l1986_198691

noncomputable def f (a x : ℝ) : ℝ := a^x - x - a

def sibling_point_pair (a : ℝ) (A B : ℝ × ℝ) : Prop :=
  A.2 = f a A.1 ∧ B.2 = f a B.1 ∧ A.1 = -B.1 ∧ A.2 = -B.2

theorem range_of_a (a : ℝ) :
  (∃ A B : ℝ × ℝ, sibling_point_pair a A B) ↔ a > 1 :=
sorry

end range_of_a_l1986_198691


namespace arithmetic_mean_is_five_sixths_l1986_198672

theorem arithmetic_mean_is_five_sixths :
  let a := 3 / 4
  let b := 5 / 6
  let c := 7 / 8
  (a + c) / 2 = b := sorry

end arithmetic_mean_is_five_sixths_l1986_198672


namespace second_number_is_12_l1986_198661

noncomputable def expression := (26.3 * 12 * 20) / 3 + 125

theorem second_number_is_12 :
  expression = 2229 → 12 = 12 :=
by sorry

end second_number_is_12_l1986_198661


namespace total_increase_percentage_l1986_198652

-- Define the conditions: original speed S, first increase by 30%, then another increase by 10%
def original_speed (S : ℝ) := S
def first_increase (S : ℝ) := S * 1.30
def second_increase (S : ℝ) := (S * 1.30) * 1.10

-- Prove that the total increase in speed is 43% of the original speed
theorem total_increase_percentage (S : ℝ) :
  (second_increase S - original_speed S) / original_speed S * 100 = 43 :=
by
  sorry

end total_increase_percentage_l1986_198652


namespace not_simplifiable_by_difference_of_squares_l1986_198638

theorem not_simplifiable_by_difference_of_squares :
  ¬(∃ a b : ℝ, (-x + y) * (x - y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (-x - y) * (-x + y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (y + x) * (x - y) = a^2 - b^2) ∧
  (∃ a b : ℝ, (y - x) * (x + y) = a^2 - b^2) :=
sorry

end not_simplifiable_by_difference_of_squares_l1986_198638


namespace circumference_in_scientific_notation_l1986_198655

noncomputable def circumference_m : ℝ := 4010000

noncomputable def scientific_notation (m: ℝ) : Prop :=
  m = 4.01 * 10^6

theorem circumference_in_scientific_notation : scientific_notation circumference_m :=
by
  sorry

end circumference_in_scientific_notation_l1986_198655


namespace list_price_is_40_l1986_198690

open Real

def list_price (x : ℝ) : Prop :=
  0.15 * (x - 15) = 0.25 * (x - 25)

theorem list_price_is_40 : list_price 40 :=
by
  unfold list_price
  sorry

end list_price_is_40_l1986_198690


namespace travel_time_in_minutes_l1986_198609

def bird_speed : ℝ := 8 -- Speed of the bird in miles per hour
def distance_to_travel : ℝ := 3 -- Distance to be traveled in miles

theorem travel_time_in_minutes : (distance_to_travel / bird_speed) * 60 = 22.5 :=
by
  sorry

end travel_time_in_minutes_l1986_198609


namespace sum_of_numbers_l1986_198643

theorem sum_of_numbers (a b c : ℝ) 
  (h₁ : (a + b + c) / 3 = a + 20) 
  (h₂ : (a + b + c) / 3 = c - 30) 
  (h₃ : b = 10) : 
  a + b + c = 60 := 
by
  sorry

end sum_of_numbers_l1986_198643


namespace tangent_line_at_P_eq_2x_l1986_198656

noncomputable def tangentLineEq (f : ℝ → ℝ) (P : ℝ × ℝ) : ℝ → ℝ :=
  let slope := deriv f P.1
  fun x => slope * (x - P.1) + P.2

theorem tangent_line_at_P_eq_2x : 
  ∀ (f : ℝ → ℝ) (x y : ℝ),
    f x = x^2 + 1 → 
    (x = 1) → (y = 2) →
    tangentLineEq f (x, y) x = 2 * x :=
by
  intros f x y f_eq hx hy
  sorry

end tangent_line_at_P_eq_2x_l1986_198656


namespace solve_for_x_l1986_198682

theorem solve_for_x (x : ℝ) (h : 3375 = (1 / 4) * x + 144) : x = 12924 :=
by
  sorry

end solve_for_x_l1986_198682


namespace jogging_track_circumference_l1986_198657

theorem jogging_track_circumference 
  (deepak_speed : ℝ)
  (wife_speed : ℝ)
  (meeting_time : ℝ)
  (circumference : ℝ)
  (H1 : deepak_speed = 4.5)
  (H2 : wife_speed = 3.75)
  (H3 : meeting_time = 4.08) :
  circumference = 33.66 := sorry

end jogging_track_circumference_l1986_198657


namespace log_addition_l1986_198685

theorem log_addition (log_base_10 : ℝ → ℝ) (a b : ℝ) (h_base_10_log : log_base_10 10 = 1) :
  log_base_10 2 + log_base_10 5 = 1 :=
by
  sorry

end log_addition_l1986_198685


namespace simplify_expr_l1986_198607

open Real

theorem simplify_expr (x : ℝ) (hx : 1 ≤ x) :
  sqrt (x + 2 * sqrt (x - 1)) + sqrt (x - 2 * sqrt (x - 1)) = 
  if x ≤ 2 then 2 else 2 * sqrt (x - 1) :=
by sorry

end simplify_expr_l1986_198607


namespace sum_of_cubes_l1986_198687

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : 
  a^3 + b^3 = 1008 := 
by 
  sorry

end sum_of_cubes_l1986_198687


namespace Hulk_jump_more_than_500_l1986_198653

theorem Hulk_jump_more_than_500 :
  ∀ n : ℕ, 2 * 3^(n - 1) > 500 → n = 7 :=
by
  sorry

end Hulk_jump_more_than_500_l1986_198653


namespace sum_of_digits_S_l1986_198673

-- Define S as 10^2021 - 2021
def S : ℕ := 10^2021 - 2021

-- Define function to calculate sum of digits of a given number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum 

theorem sum_of_digits_S :
  sum_of_digits S = 18185 :=
sorry

end sum_of_digits_S_l1986_198673


namespace negation_of_proposition_l1986_198699

theorem negation_of_proposition :
  ¬ (∃ x : ℝ, x < 1) ↔ ∀ x : ℝ, x ≥ 1 :=
by sorry

end negation_of_proposition_l1986_198699


namespace candy_selection_l1986_198612

theorem candy_selection (m n : ℕ) (h1 : Nat.gcd m n = 1) (h2 : m = 1) (h3 : n = 5) :
  m + n = 6 := by
  sorry

end candy_selection_l1986_198612


namespace main_world_population_transition_l1986_198626

noncomputable def world_population_reproduction_transition
  (developing_majority : Prop)
  (developing_transition : Prop)
  (developed_small : Prop)
  (developed_modern : Prop)
  (minor_impact : Prop) : Prop :=
  developing_majority ∧ developing_transition ∧ developed_small ∧ developed_modern ∧ minor_impact →
  "Traditional" = "Traditional" ∧ "Modern" = "Modern" →
  "Traditional" = "Traditional"

theorem main_world_population_transition
  (developing_majority : Prop)
  (developing_transition : Prop)
  (developed_small : Prop)
  (developed_modern : Prop)
  (minor_impact : Prop) :
  developing_majority ∧ developing_transition ∧ developed_small ∧ developed_modern ∧ minor_impact →
  "Traditional" = "Traditional" ∧ "Modern" = "Modern" →
  "Traditional" = "Traditional" :=
by
  sorry

end main_world_population_transition_l1986_198626


namespace sum_of_youngest_and_oldest_nephews_l1986_198668

theorem sum_of_youngest_and_oldest_nephews 
    (n1 n2 n3 n4 n5 n6 : ℕ) 
    (mean_eq : (n1 + n2 + n3 + n4 + n5 + n6) / 6 = 10) 
    (median_eq : (n3 + n4) / 2 = 12) : 
    n1 + n6 = 12 := 
by 
    sorry

end sum_of_youngest_and_oldest_nephews_l1986_198668


namespace part1_min_value_part2_find_b_part3_range_b_div_a_l1986_198659

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 - abs (a*x - b)

-- Part (1)
theorem part1_min_value : f 1 1 1 = -5/4 :=
by 
  sorry

-- Part (2)
theorem part2_find_b (b : ℝ) (h : b ≥ 2) (h_domain : ∀ x, 1 ≤ x ∧ x ≤ b) (h_range : ∀ y, 1 ≤ y ∧ y ≤ b) : 
  b = 2 :=
by 
  sorry

-- Part (3)
theorem part3_range_b_div_a (a b : ℝ) (h_distinct : (∃ x : ℝ, 0 < x ∧ x < 2 ∧ f x a b = 1 ∧ ∀ y : ℝ, 0 < y ∧ y < 2 ∧ f y a b = 1 ∧ x ≠ y)) : 
  1 < b / a ∧ b / a < 2 :=
by 
  sorry

end part1_min_value_part2_find_b_part3_range_b_div_a_l1986_198659


namespace tank_capacity_l1986_198610

theorem tank_capacity :
  let rateA := 40  -- Pipe A fills at 40 liters per minute
  let rateB := 30  -- Pipe B fills at 30 liters per minute
  let rateC := -20  -- Pipe C (drains) at 20 liters per minute, thus negative contribution
  let cycle_duration := 3  -- The cycle duration is 3 minutes
  let total_duration := 51  -- The tank gets full in 51 minutes
  let net_per_cycle := rateA + rateB + rateC  -- Net fill per cycle of 3 minutes
  let num_cycles := total_duration / cycle_duration  -- Number of complete cycles
  let tank_capacity := net_per_cycle * num_cycles  -- Tank capacity in liters
  tank_capacity = 850  -- Assertion that needs to be proven
:= by
  let rateA := 40
  let rateB := 30
  let rateC := -20
  let cycle_duration := 3
  let total_duration := 51
  let net_per_cycle := rateA + rateB + rateC
  let num_cycles := total_duration / cycle_duration
  let tank_capacity := net_per_cycle * num_cycles
  have : tank_capacity = 850 := by
    sorry
  assumption

end tank_capacity_l1986_198610


namespace no_solution_xn_yn_zn_l1986_198604

theorem no_solution_xn_yn_zn (x y z n : ℕ) (h : n ≥ z) : ¬ (x^n + y^n = z^n) :=
sorry

end no_solution_xn_yn_zn_l1986_198604


namespace length_of_parallelepiped_l1986_198697

def number_of_cubes_with_painted_faces (n : ℕ) := (n - 2) * (n - 4) * (n - 6) 
def total_number_of_cubes (n : ℕ) := n * (n - 2) * (n - 4)

theorem length_of_parallelepiped (n : ℕ) (h1 : total_number_of_cubes n = 3 * number_of_cubes_with_painted_faces n) : 
  n = 18 :=
by 
  sorry

end length_of_parallelepiped_l1986_198697


namespace sum_lent_is_300_l1986_198665

-- Define the conditions
def interest_rate : ℕ := 4
def time_period : ℕ := 8
def interest_amounted_less : ℕ := 204

-- Prove that the sum lent P is 300 given the conditions
theorem sum_lent_is_300 (P : ℕ) : 
  (P * interest_rate * time_period / 100 = P - interest_amounted_less) -> P = 300 := by
  sorry

end sum_lent_is_300_l1986_198665


namespace find_pairs_l1986_198696

theorem find_pairs (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  (a * b^2 + b + 7) ∣ (a^2 * b + a + b) ↔ ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ ∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k) :=
by sorry

end find_pairs_l1986_198696


namespace fourth_person_height_l1986_198642

variables (H1 H2 H3 H4 : ℝ)

theorem fourth_person_height :
  H2 = H1 + 2 →
  H3 = H2 + 3 →
  H4 = H3 + 6 →
  H1 + H2 + H3 + H4 = 288 →
  H4 = 78.5 :=
by
  intros h2_def h3_def h4_def total_height
  -- Proof steps would follow here
  sorry

end fourth_person_height_l1986_198642


namespace max_m_value_real_roots_interval_l1986_198658

theorem max_m_value_real_roots_interval :
  (∃ x ∈ (Set.Icc 0 1), x^3 - 3 * x - m = 0) → m ≤ 0 :=
by
  sorry 

end max_m_value_real_roots_interval_l1986_198658


namespace find_bottle_price_l1986_198617

theorem find_bottle_price 
  (x : ℝ) 
  (promotion_free_bottles : ℝ := 3)
  (discount_per_bottle : ℝ := 0.6)
  (box_price : ℝ := 26)
  (box_bottles : ℝ := 4) :
  ∃ x : ℝ, (box_price / (x - discount_per_bottle)) - (box_price / x) = promotion_free_bottles :=
sorry

end find_bottle_price_l1986_198617


namespace quadratic_solution_downward_solution_minimum_solution_l1986_198605

def is_quadratic (m : ℝ) : Prop :=
  m^2 + 3 * m - 2 = 2

def opens_downwards (m : ℝ) : Prop :=
  m + 3 < 0

def has_minimum (m : ℝ) : Prop :=
  m + 3 > 0

theorem quadratic_solution (m : ℝ) :
  is_quadratic m → (m = -4 ∨ m = 1) :=
sorry

theorem downward_solution (m : ℝ) :
  is_quadratic m → opens_downwards m → m = -4 :=
sorry

theorem minimum_solution (m : ℝ) :
  is_quadratic m → has_minimum m → m = 1 :=
sorry

end quadratic_solution_downward_solution_minimum_solution_l1986_198605


namespace trig_identity_l1986_198615

theorem trig_identity :
  (Real.cos (42 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) - 
   Real.cos (48 * Real.pi / 180) * Real.sin (18 * Real.pi / 180)) = 1 / 2 := 
by sorry

end trig_identity_l1986_198615


namespace arithmetic_seq_product_of_first_two_terms_l1986_198694

theorem arithmetic_seq_product_of_first_two_terms
    (a d : ℤ)
    (h1 : a + 4 * d = 17)
    (h2 : d = 2) :
    (a * (a + d) = 99) := 
by
    -- Proof to be done
    sorry

end arithmetic_seq_product_of_first_two_terms_l1986_198694


namespace tricycles_in_garage_l1986_198662

theorem tricycles_in_garage 
    (T : ℕ) 
    (total_bicycles : ℕ := 3) 
    (total_unicycles : ℕ := 7) 
    (bicycle_wheels : ℕ := 2) 
    (tricycle_wheels : ℕ := 3) 
    (unicycle_wheels : ℕ := 1) 
    (total_wheels : ℕ := 25) 
    (eq_wheels : total_bicycles * bicycle_wheels + total_unicycles * unicycle_wheels + T * tricycle_wheels = total_wheels) :
    T = 4 :=
by {
  sorry
}

end tricycles_in_garage_l1986_198662
