import Mathlib

namespace three_layers_rug_area_l164_16417

theorem three_layers_rug_area :
  ∀ (A B C D E : ℝ),
    A + B + C = 212 →
    (A + B + C) - D - 2 * E = 140 →
    D = 24 →
    E = 24 :=
by
  intros A B C D E h1 h2 h3
  sorry

end three_layers_rug_area_l164_16417


namespace adam_chocolate_boxes_l164_16403

theorem adam_chocolate_boxes 
  (c : ℕ) -- number of chocolate boxes Adam bought
  (h1 : 4 * c + 4 * 5 = 28) : 
  c = 2 := 
by
  sorry

end adam_chocolate_boxes_l164_16403


namespace powderman_distance_when_blast_heard_l164_16463

-- Define constants
def fuse_time : ℝ := 30  -- seconds
def run_rate : ℝ := 8    -- yards per second
def sound_rate : ℝ := 1080  -- feet per second
def yards_to_feet : ℝ := 3  -- conversion factor

-- Define the time at which the blast was heard
noncomputable def blast_heard_time : ℝ := 675 / 22

-- Define distance functions
def p (t : ℝ) : ℝ := run_rate * yards_to_feet * t  -- distance run by powderman in feet
def q (t : ℝ) : ℝ := sound_rate * (t - fuse_time)  -- distance sound has traveled in feet

-- Proof statement: given the conditions, the distance run by the powderman equals 245 yards
theorem powderman_distance_when_blast_heard :
  p (blast_heard_time) / yards_to_feet = 245 := by
  sorry

end powderman_distance_when_blast_heard_l164_16463


namespace equation_of_line_l_l164_16498

theorem equation_of_line_l :
  (∃ l : ℝ → ℝ → Prop, 
     (∀ x y, l x y ↔ (x - y + 3) = 0)
     ∧ (∀ x y, l x y → x^2 + (y - 3)^2 = 4)
     ∧ (∀ x y, l x y → x + y + 1 = 0)) :=
sorry

end equation_of_line_l_l164_16498


namespace gcd_sequence_inequality_l164_16442

-- Add your Lean 4 statement here
theorem gcd_sequence_inequality {n : ℕ} 
  (h : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 35 → Nat.gcd n k < Nat.gcd n (k+1)) : 
  Nat.gcd n 35 < Nat.gcd n 36 := 
sorry

end gcd_sequence_inequality_l164_16442


namespace total_cost_is_346_l164_16421

-- Definitions of the given conditions
def total_people : ℕ := 35 + 5 + 1
def total_lunches : ℕ := total_people + 3
def vegetarian_lunches : ℕ := 10
def gluten_free_lunches : ℕ := 5
def nut_free_lunches : ℕ := 3
def halal_lunches : ℕ := 4
def veg_and_gluten_free_lunches : ℕ := 2
def regular_cost : ℕ := 7
def special_cost : ℕ := 8
def veg_and_gluten_free_cost : ℕ := 9

-- Calculate regular lunches considering dietary overlaps
def regular_lunches : ℕ := 
  total_lunches - vegetarian_lunches - gluten_free_lunches - nut_free_lunches - halal_lunches + veg_and_gluten_free_lunches

-- Calculate costs per category of lunches
def total_regular_cost : ℕ := regular_lunches * regular_cost
def total_vegetarian_cost : ℕ := (vegetarian_lunches - veg_and_gluten_free_lunches) * special_cost
def total_gluten_free_cost : ℕ := gluten_free_lunches * special_cost
def total_nut_free_cost : ℕ := nut_free_lunches * special_cost
def total_halal_cost : ℕ := halal_lunches * special_cost
def total_veg_and_gluten_free_cost : ℕ := veg_and_gluten_free_lunches * veg_and_gluten_free_cost

-- Calculate total cost
def total_cost : ℕ :=
  total_regular_cost + total_vegetarian_cost + total_gluten_free_cost + total_nut_free_cost + total_halal_cost + total_veg_and_gluten_free_cost

-- Theorem stating the main question
theorem total_cost_is_346 : total_cost = 346 :=
  by
    -- This is where the proof would go
    sorry

end total_cost_is_346_l164_16421


namespace cost_of_adult_ticket_is_8_l164_16428

variables (A : ℕ) (num_people : ℕ := 22) (total_money : ℕ := 50) (num_children : ℕ := 18) (child_ticket_cost : ℕ := 1)

-- Definitions based on the given conditions
def child_tickets_cost := num_children * child_ticket_cost
def num_adults := num_people - num_children
def adult_tickets_cost := total_money - child_tickets_cost
def cost_per_adult_ticket := adult_tickets_cost / num_adults

-- The theorem stating that the cost of an adult ticket is 8 dollars
theorem cost_of_adult_ticket_is_8 : cost_per_adult_ticket = 8 :=
by sorry

end cost_of_adult_ticket_is_8_l164_16428


namespace not_lt_neg_version_l164_16423

theorem not_lt_neg_version (a b : ℝ) (h : a < b) : ¬ (-3 * a < -3 * b) :=
by 
  -- This is where the proof would go
  sorry

end not_lt_neg_version_l164_16423


namespace probability_selecting_cooking_l164_16451

theorem probability_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let favorable_outcomes := 1
  let total_outcomes := courses.length
  let probability := favorable_outcomes / total_outcomes
  probability = 1 / 4 :=
by
  sorry

end probability_selecting_cooking_l164_16451


namespace geom_seq_product_a2_a3_l164_16473

theorem geom_seq_product_a2_a3 :
  ∃ (a_n : ℕ → ℝ), (a_n 1 * a_n 4 = -3) ∧ (∀ n, a_n n = a_n 1 * (a_n 2 / a_n 1) ^ (n - 1)) → a_n 2 * a_n 3 = -3 :=
by
  sorry

end geom_seq_product_a2_a3_l164_16473


namespace division_multiplication_calculation_l164_16457

theorem division_multiplication_calculation :
  (30 / (7 + 2 - 3)) * 4 = 20 :=
by
  sorry

end division_multiplication_calculation_l164_16457


namespace net_percentage_gain_approx_l164_16461

noncomputable def netPercentageGain : ℝ :=
  let costGlassBowls := 250 * 18
  let costCeramicPlates := 150 * 25
  let totalCostBeforeDiscount := costGlassBowls + costCeramicPlates
  let discount := 0.05 * totalCostBeforeDiscount
  let totalCostAfterDiscount := totalCostBeforeDiscount - discount
  let revenueGlassBowls := 200 * 25
  let revenueCeramicPlates := 120 * 32
  let totalRevenue := revenueGlassBowls + revenueCeramicPlates
  let costBrokenGlassBowls := 30 * 18
  let costBrokenCeramicPlates := 10 * 25
  let totalCostBrokenItems := costBrokenGlassBowls + costBrokenCeramicPlates
  let netGain := totalRevenue - (totalCostAfterDiscount + totalCostBrokenItems)
  let netPercentageGain := (netGain / totalCostAfterDiscount) * 100
  netPercentageGain

theorem net_percentage_gain_approx :
  abs (netPercentageGain - 2.71) < 0.01 := sorry

end net_percentage_gain_approx_l164_16461


namespace time_for_B_alone_l164_16434

theorem time_for_B_alone (W_A W_B : ℝ) (h1 : W_A = 2 * W_B) (h2 : W_A + W_B = 1/6) : 1 / W_B = 18 := by
  sorry

end time_for_B_alone_l164_16434


namespace smallest_positive_x_l164_16452

theorem smallest_positive_x (x : ℝ) (h : x > 0) (h_eq : x / 4 + 3 / (4 * x) = 1) : x = 1 :=
by
  sorry

end smallest_positive_x_l164_16452


namespace exp_neg_eq_l164_16448

theorem exp_neg_eq (θ φ : ℝ) (h : Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (1 / 2 : ℂ) + (1 / 3 : ℂ) * Complex.I) :
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (1 / 2 : ℂ) - (1 / 3 : ℂ) * Complex.I :=
by sorry

end exp_neg_eq_l164_16448


namespace odd_function_five_value_l164_16464

variable (f : ℝ → ℝ)

theorem odd_function_five_value (h_odd : ∀ x : ℝ, f (-x) = -f x)
                               (h_f1 : f 1 = 1 / 2)
                               (h_f_recurrence : ∀ x : ℝ, f (x + 2) = f x + f 2) :
  f 5 = 5 / 2 :=
sorry

end odd_function_five_value_l164_16464


namespace area_of_square_l164_16487

theorem area_of_square (side_length : ℝ) (h : side_length = 17) : side_length * side_length = 289 :=
by
  sorry

end area_of_square_l164_16487


namespace find_x_l164_16493

-- Defining the sum of integers from 30 to 40 inclusive
def sum_30_to_40 : ℕ := (30 + 31 + 32 + 33 + 34 + 35 + 36 + 37 + 38 + 39 + 40)

-- Defining the number of even integers from 30 to 40 inclusive
def count_even_30_to_40 : ℕ := 6

-- Given that x + y = 391, and y = count_even_30_to_40
-- Prove that x is equal to 385
theorem find_x (h : sum_30_to_40 + count_even_30_to_40 = 391) : sum_30_to_40 = 385 :=
by
  simp [sum_30_to_40, count_even_30_to_40] at h
  sorry

end find_x_l164_16493


namespace sqrt_sum_l164_16416

theorem sqrt_sum : (Real.sqrt 50) + (Real.sqrt 32) = 9 * (Real.sqrt 2) :=
by
  sorry

end sqrt_sum_l164_16416


namespace max_x_minus_y_isosceles_l164_16439

theorem max_x_minus_y_isosceles (x y : ℝ) (hx : x ≠ 50) (hy : y ≠ 50) 
  (h_iso1 : x = y ∨ 50 = y) (h_iso2 : x = y ∨ 50 = x)
  (h_triangle : 50 + x + y = 180) : 
  max (x - y) (y - x) = 30 :=
sorry

end max_x_minus_y_isosceles_l164_16439


namespace ratio_of_areas_l164_16409

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ :=
1 / 2 * a * b

theorem ratio_of_areas (a b c x y z : ℝ)
  (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) 
  (h4 : x = 9) (h5 : y = 12) (h6 : z = 15)
  (h7 : a^2 + b^2 = c^2) (h8 : x^2 + y^2 = z^2) :
  (area_of_right_triangle a b) / (area_of_right_triangle x y) = 4 / 9 :=
sorry

end ratio_of_areas_l164_16409


namespace sum_of_squares_of_roots_l164_16484

theorem sum_of_squares_of_roots (x_1 x_2 : ℚ) (h1 : 6 * x_1^2 - 13 * x_1 + 5 = 0)
                                (h2 : 6 * x_2^2 - 13 * x_2 + 5 = 0) 
                                (h3 : x_1 ≠ x_2) :
  x_1^2 + x_2^2 = 109 / 36 :=
sorry

end sum_of_squares_of_roots_l164_16484


namespace systemOfEquationsUniqueSolution_l164_16497

def largeBarrelHolds (x : ℝ) (y : ℝ) : Prop :=
  5 * x + y = 3

def smallBarrelHolds (x : ℝ) (y : ℝ) : Prop :=
  x + 5 * y = 2

theorem systemOfEquationsUniqueSolution (x y : ℝ) :
  (largeBarrelHolds x y) ∧ (smallBarrelHolds x y) ↔ 
  (5 * x + y = 3 ∧ x + 5 * y = 2) :=
by
  sorry

end systemOfEquationsUniqueSolution_l164_16497


namespace DVDs_sold_is_168_l164_16445

variables (C D : ℕ)
variables (h1 : D = (16 * C) / 10)
variables (h2 : D + C = 273)

theorem DVDs_sold_is_168 : D = 168 := by
  sorry

end DVDs_sold_is_168_l164_16445


namespace degree_of_divisor_l164_16481

theorem degree_of_divisor (f d q r : Polynomial ℝ) 
  (hf : f.degree = 15) 
  (hq : q.degree = 9) 
  (hr : r.degree = 4) 
  (hr_poly : r = (Polynomial.C 5) * (Polynomial.X^4) + (Polynomial.C 6) * (Polynomial.X^3) - (Polynomial.C 2) * (Polynomial.X) + (Polynomial.C 7)) 
  (hdiv : f = d * q + r) : 
  d.degree = 6 := 
sorry

end degree_of_divisor_l164_16481


namespace find_lambda_l164_16455

noncomputable def vec_length (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def dot_product (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

theorem find_lambda {a b : ℝ × ℝ} (lambda : ℝ) 
  (ha : vec_length a = 1) (hb : vec_length b = 2)
  (hab_angle : dot_product a b = -1) 
  (h_perp : dot_product (lambda • a + b) (a - 2 • b) = 0) : 
  lambda = 3 := 
sorry

end find_lambda_l164_16455


namespace mean_of_remaining_students_l164_16402

theorem mean_of_remaining_students
  (n : ℕ) (h : n > 20)
  (mean_score_first_15 : ℝ)
  (mean_score_next_5 : ℝ)
  (overall_mean_score : ℝ) :
  mean_score_first_15 = 10 →
  mean_score_next_5 = 16 →
  overall_mean_score = 11 →
  ∀ a, a = (11 * n - 230) / (n - 20) := by
sorry

end mean_of_remaining_students_l164_16402


namespace range_of_m_l164_16412

noncomputable def f (x m : ℝ) : ℝ :=
if x < 0 then 1 / (Real.exp x) + m * x^2
else Real.exp x + m * x^2

theorem range_of_m {m : ℝ} : (∀ m, ∃ x y, f x m = 0 ∧ f y m = 0 ∧ x ≠ y) ↔ m < -Real.exp 2 / 4 := by
  sorry

end range_of_m_l164_16412


namespace value_at_1971_l164_16494

def sequence_x (x : ℕ → ℝ) :=
  ∀ n > 1, 3 * x n - x (n - 1) = n

theorem value_at_1971 (x : ℕ → ℝ) (hx : sequence_x x) (h_initial : abs (x 1) < 1971) :
  abs (x 1971 - 985.25) < 0.000001 :=
by sorry

end value_at_1971_l164_16494


namespace angela_spent_78_l164_16495

-- Definitions
def angela_initial_money : ℕ := 90
def angela_left_money : ℕ := 12
def angela_spent_money : ℕ := angela_initial_money - angela_left_money

-- Theorem statement
theorem angela_spent_78 : angela_spent_money = 78 := by
  -- Proof would go here, but it is not required.
  sorry

end angela_spent_78_l164_16495


namespace simplify_expression_l164_16437

theorem simplify_expression (p q x : ℝ) (h₀ : p ≠ 0) (h₁ : q ≠ 0) (h₂ : x > 0) (h₃ : x ≠ 1) :
  (x^(3 / p) - x^(3 / q)) / ((x^(1 / p) + x^(1 / q))^2 - 2 * x^(1 / q) * (x^(1 / q) + x^(1 / p)))
  + x^(1 / p) / (x^((q - p) / (p * q)) + 1) = x^(1 / p) + x^(1 / q) := 
sorry

end simplify_expression_l164_16437


namespace students_neither_cool_l164_16460

variable (total_students : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ)

def only_cool_dads := cool_dads - both_cool
def only_cool_moms := cool_moms - both_cool
def only_cool := only_cool_dads + only_cool_moms + both_cool
def neither_cool := total_students - only_cool

theorem students_neither_cool (h1 : total_students = 40) (h2 : cool_dads = 18) (h3 : cool_moms = 22) (h4 : both_cool = 10) 
: neither_cool total_students cool_dads cool_moms both_cool = 10 :=
by 
  sorry

end students_neither_cool_l164_16460


namespace combined_spots_l164_16438

-- Definitions of the conditions
def Rover_spots : ℕ := 46
def Cisco_spots : ℕ := Rover_spots / 2 - 5
def Granger_spots : ℕ := 5 * Cisco_spots

-- The proof statement
theorem combined_spots :
  Granger_spots + Cisco_spots = 108 := by
  sorry

end combined_spots_l164_16438


namespace find_LCM_of_three_numbers_l164_16424

noncomputable def LCM_of_three_numbers (a b c : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm a b) c

theorem find_LCM_of_three_numbers
  (a b c : ℕ)
  (h_prod : a * b * c = 1354808)
  (h_gcd : Nat.gcd (Nat.gcd a b) c = 11) :
  LCM_of_three_numbers a b c = 123164 := by
  sorry

end find_LCM_of_three_numbers_l164_16424


namespace div_by_5_l164_16489

theorem div_by_5 (a b : ℕ) (h: 5 ∣ (a * b)) : (5 ∣ a) ∨ (5 ∣ b) :=
by
  -- Proof by contradiction
  -- Assume the negation of the conclusion
  have h_nand : ¬ (5 ∣ a) ∧ ¬ (5 ∣ b) := sorry

  -- Derive a contradiction based on the assumptions
  sorry

end div_by_5_l164_16489


namespace trains_total_distance_l164_16471

theorem trains_total_distance (speed_A speed_B : ℝ) (time_A time_B : ℝ) (dist_A dist_B : ℝ):
  speed_A = 90 ∧ 
  speed_B = 120 ∧ 
  time_A = 1 ∧ 
  time_B = 5/6 ∧ 
  dist_A = speed_A * time_A ∧ 
  dist_B = speed_B * time_B ->
  (dist_A + dist_B) = 190 :=
by 
  intros h
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := h
  sorry

end trains_total_distance_l164_16471


namespace iron_column_lifted_by_9_6_cm_l164_16480

namespace VolumeLift

def base_area_container : ℝ := 200
def base_area_column : ℝ := 40
def height_water : ℝ := 16
def distance_water_surface : ℝ := 4

theorem iron_column_lifted_by_9_6_cm :
  ∃ (h_lift : ℝ),
    h_lift = 9.6 ∧ height_water - distance_water_surface = 16 - h_lift :=
by
sorry

end VolumeLift

end iron_column_lifted_by_9_6_cm_l164_16480


namespace part1_part2_l164_16478

def f (x a : ℝ) := abs (x - a)

theorem part1 (a : ℝ) :
  (∀ x : ℝ, (f x a) ≤ 2 ↔ 1 ≤ x ∧ x ≤ 5) → a = 3 :=
by
  intros h
  sorry

theorem part2 (m : ℝ) : 
  (∀ x : ℝ, f (2 * x) 3 + f (x + 2) 3 ≥ m) → m ≤ 1 / 2 :=
by
  intros h
  sorry

end part1_part2_l164_16478


namespace circumference_ratio_l164_16408

theorem circumference_ratio (C D : ℝ) (hC : C = 94.2) (hD : D = 30) : C / D = 3.14 :=
by {
  sorry
}

end circumference_ratio_l164_16408


namespace line_PQ_passes_through_fixed_point_l164_16420

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 2 = 1

-- Define the conditions for points P and Q on the hyperbola
def on_hyperbola (P Q : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2 ∧ hyperbola Q.1 Q.2

-- Define the condition for perpendicular lines, given points A, P, and Q
def perpendicular (A P Q : ℝ × ℝ) : Prop :=
  ((P.2 - A.2) / (P.1 - A.1)) * ((Q.2 - A.2) / (Q.1 - A.1)) = -1

-- Define the main theorem to prove
theorem line_PQ_passes_through_fixed_point :
  ∀ (P Q : ℝ × ℝ), on_hyperbola P Q → perpendicular ⟨-1, 0⟩ P Q →
    ∃ (b : ℝ), ∀ (y : ℝ), (P.1 = y * P.2 + b ∨ Q.1 = y * Q.2 + b) → (b = 3) :=
by
  sorry

end line_PQ_passes_through_fixed_point_l164_16420


namespace solve_for_x_l164_16477

theorem solve_for_x (x : ℚ) (h : 3 / 4 - 1 / x = 1 / 2) : x = 4 :=
sorry

end solve_for_x_l164_16477


namespace difference_of_interchanged_digits_l164_16472

theorem difference_of_interchanged_digits (X Y : ℕ) (h : X - Y = 5) : (10 * X + Y) - (10 * Y + X) = 45 :=
by
  sorry

end difference_of_interchanged_digits_l164_16472


namespace ernaldo_friends_count_l164_16427

-- Define the members of the group
inductive Member
| Arnaldo
| Bernaldo
| Cernaldo
| Dernaldo
| Ernaldo

open Member

-- Define the number of friends for each member
def number_of_friends : Member → ℕ
| Arnaldo  => 1
| Bernaldo => 2
| Cernaldo => 3
| Dernaldo => 4
| Ernaldo  => 0  -- This will be our unknown to solve

-- The main theorem we need to prove
theorem ernaldo_friends_count : number_of_friends Ernaldo = 2 :=
sorry

end ernaldo_friends_count_l164_16427


namespace original_profit_percentage_l164_16453

theorem original_profit_percentage (C : ℝ) (C' : ℝ) (S' : ℝ) (H1 : C = 40) (H2 : C' = 32) (H3 : S' = 41.60) 
  (H4 : S' = (1.30 * C')) : (S' + 8.40 - C) / C * 100 = 25 := 
by 
  sorry

end original_profit_percentage_l164_16453


namespace total_distance_travelled_l164_16422

def speed_one_sail : ℕ := 25 -- knots
def speed_two_sails : ℕ := 50 -- knots
def conversion_factor : ℕ := 115 -- 1.15, in hundredths

def distance_in_nautical_miles : ℕ :=
  (2 * speed_one_sail) +      -- Two hours, one sail
  (3 * speed_two_sails) +     -- Three hours, two sails
  (1 * speed_one_sail) +      -- One hour, one sail, navigating around obstacles
  (2 * (speed_one_sail - speed_one_sail * 30 / 100)) -- Two hours, strong winds, 30% reduction in speed

def distance_in_land_miles : ℕ :=
  distance_in_nautical_miles * conversion_factor / 100 -- Convert to land miles

theorem total_distance_travelled : distance_in_land_miles = 299 := by
  sorry

end total_distance_travelled_l164_16422


namespace total_amount_spent_l164_16426

theorem total_amount_spent : 
  let value_half_dollar := 0.50
  let wednesday_spending := 4 * value_half_dollar
  let next_day_spending := 14 * value_half_dollar
  wednesday_spending + next_day_spending = 9.00 :=
by
  let value_half_dollar := 0.50
  let wednesday_spending := 4 * value_half_dollar
  let next_day_spending := 14 * value_half_dollar
  show _ 
  sorry

end total_amount_spent_l164_16426


namespace triangle_inscribed_relation_l164_16415

noncomputable def herons_area (p a b c : ℝ) : ℝ := (p * (p - a) * (p - b) * (p - c)).sqrt

theorem triangle_inscribed_relation
  (S S' p p' : ℝ)
  (a b c a' b' c' r : ℝ)
  (h1 : r = S / p)
  (h2 : r = S' / p')
  (h3 : S = herons_area p a b c)
  (h4 : S' = herons_area p' a' b' c') :
  (p - a) * (p - b) * (p - c) / p = (p' - a') * (p' - b') * (p' - c') / p' :=
by sorry

end triangle_inscribed_relation_l164_16415


namespace golden_state_total_points_l164_16479

theorem golden_state_total_points :
  ∀ (Draymond Curry Kelly Durant Klay : ℕ),
  Draymond = 12 →
  Curry = 2 * Draymond →
  Kelly = 9 →
  Durant = 2 * Kelly →
  Klay = Draymond / 2 →
  Draymond + Curry + Kelly + Durant + Klay = 69 :=
by
  intros Draymond Curry Kelly Durant Klay
  intros hD hC hK hD2 hK2
  rw [hD, hC, hK, hD2, hK2]
  sorry

end golden_state_total_points_l164_16479


namespace area_of_triangle_ABC_l164_16490

variable (A B C : ℝ × ℝ)
variable (x1 y1 x2 y2 x3 y3 : ℝ)

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let x1 := A.1
  let y1 := A.2
  let x2 := B.1
  let y2 := B.2
  let x3 := C.1
  let y3 := C.2
  0.5 * (abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem area_of_triangle_ABC :
  let A := (1, 2)
  let B := (-2, 5)
  let C := (4, -2)
  area_of_triangle A B C = 1.5 :=
by
  sorry

end area_of_triangle_ABC_l164_16490


namespace total_pairs_purchased_l164_16466

-- Define the conditions as hypotheses
def foxPrice : ℝ := 15
def ponyPrice : ℝ := 18
def totalSaved : ℝ := 8.91
def foxPairs : ℕ := 3
def ponyPairs : ℕ := 2
def sumDiscountRates : ℝ := 0.22
def ponyDiscountRate : ℝ := 0.10999999999999996

-- Prove that the total number of pairs of jeans purchased is 5
theorem total_pairs_purchased : foxPairs + ponyPairs = 5 := by
  sorry

end total_pairs_purchased_l164_16466


namespace ratio_of_speeds_l164_16470

variable (b r : ℝ) (h1 : 1 / (b - r) = 2 * (1 / (b + r)))
variable (f1 f2 : ℝ) (h2 : b * (1/4) + b * (3/4) = b)

theorem ratio_of_speeds (b r : ℝ) (h1 : 1 / (b - r) = 2 * (1 / (b + r))) : b = 3 * r :=
by sorry

end ratio_of_speeds_l164_16470


namespace gcd_lcm_product_24_36_proof_l164_16444

def gcd_lcm_product_24_36 : Prop :=
  let a := 24
  let b := 36
  let gcd_ab := Int.gcd a b
  let lcm_ab := Int.lcm a b
  gcd_ab * lcm_ab = 864

theorem gcd_lcm_product_24_36_proof : gcd_lcm_product_24_36 :=
by
  sorry

end gcd_lcm_product_24_36_proof_l164_16444


namespace remainder_8437_by_9_l164_16456

theorem remainder_8437_by_9 : 8437 % 9 = 4 :=
by
  -- proof goes here
  sorry

end remainder_8437_by_9_l164_16456


namespace probability_green_or_yellow_l164_16499

def green_faces : ℕ := 3
def yellow_faces : ℕ := 2
def blue_faces : ℕ := 1
def total_faces : ℕ := 6

theorem probability_green_or_yellow : 
  (green_faces + yellow_faces) / total_faces = 5 / 6 :=
by
  sorry

end probability_green_or_yellow_l164_16499


namespace james_total_distance_l164_16418

-- Define the conditions
def speed_part1 : ℝ := 30  -- mph
def time_part1 : ℝ := 0.5  -- hours
def speed_part2 : ℝ := 2 * speed_part1  -- 2 * 30 mph
def time_part2 : ℝ := 2 * time_part1  -- 2 * 0.5 hours

-- Compute distances
def distance_part1 : ℝ := speed_part1 * time_part1
def distance_part2 : ℝ := speed_part2 * time_part2

-- Total distance
def total_distance : ℝ := distance_part1 + distance_part2

-- The theorem to prove
theorem james_total_distance :
  total_distance = 75 := 
sorry

end james_total_distance_l164_16418


namespace range_independent_variable_l164_16488

noncomputable def range_of_independent_variable (x : ℝ) : Prop :=
  x ≠ 3

theorem range_independent_variable (x : ℝ) :
  (∃ y : ℝ, y = 1 / (x - 3)) → x ≠ 3 :=
by
  intro h
  sorry

end range_independent_variable_l164_16488


namespace triangle_inequality_l164_16468

variables (a b c S : ℝ) (S_def : S = (a + b + c) / 2)

theorem triangle_inequality 
  (h_triangle: a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  2 * S * (Real.sqrt (S - a) + Real.sqrt (S - b) + Real.sqrt (S - c)) 
  ≤ 3 * (Real.sqrt (b * c * (S - a)) + Real.sqrt (c * a * (S - b)) + Real.sqrt (a * b * (S - c))) :=
sorry

end triangle_inequality_l164_16468


namespace brownie_pieces_count_l164_16482

def pan_width : ℕ := 24
def pan_height : ℕ := 15
def brownie_width : ℕ := 3
def brownie_height : ℕ := 2

theorem brownie_pieces_count : (pan_width * pan_height) / (brownie_width * brownie_height) = 60 := by
  sorry

end brownie_pieces_count_l164_16482


namespace problem1_problem2_l164_16491

-- First Problem Statement:
theorem problem1 :  12 - (-18) + (-7) - 20 = 3 := 
by 
  sorry

-- Second Problem Statement:
theorem problem2 : -4 / (1 / 2) * 8 = -64 := 
by 
  sorry

end problem1_problem2_l164_16491


namespace article_initial_cost_l164_16433

theorem article_initial_cost (x : ℝ) (h : 0.44 * x = 4400) : x = 10000 :=
by
  sorry

end article_initial_cost_l164_16433


namespace translate_parabola_l164_16459

theorem translate_parabola (x : ℝ) :
  (∃ (h k : ℝ), h = 1 ∧ k = 3 ∧ ∀ x: ℝ, y = 2*x^2 → y = 2*(x - h)^2 + k) := 
by
  use 1, 3
  sorry

end translate_parabola_l164_16459


namespace percentage_hindus_l164_16486

-- Conditions 
def total_boys : ℕ := 850
def percentage_muslims : ℝ := 0.44
def percentage_sikhs : ℝ := 0.10
def boys_other_communities : ℕ := 272

-- Question and proof statement
theorem percentage_hindus (total_boys : ℕ) (percentage_muslims percentage_sikhs : ℝ) (boys_other_communities : ℕ) : 
  (total_boys = 850) →
  (percentage_muslims = 0.44) →
  (percentage_sikhs = 0.10) →
  (boys_other_communities = 272) →
  ((850 - (374 + 85 + 272)) / 850) * 100 = 14 := 
by
  intros
  sorry

end percentage_hindus_l164_16486


namespace find_a_l164_16407

theorem find_a (a : ℝ) (h : ∫ x in -a..a, (2 * x - 1) = -8) : a = 4 :=
sorry

end find_a_l164_16407


namespace find_a_b_find_extreme_values_l164_16431

-- Definitions based on the conditions in the problem
def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + 2 * b

-- The function f attains a maximum value of 2 at x = -1
def f_max_at_neg_1 (a b : ℝ) : Prop :=
  (∃ x : ℝ, x = -1 ∧ 
  (∀ y : ℝ, f x a b ≤ f y a b)) ∧ f (-1) a b = 2

-- Statement (1): Finding the values of a and b
theorem find_a_b : ∃ a b : ℝ, f_max_at_neg_1 a b ∧ a = 2 ∧ b = 1 :=
sorry

-- The function f with a=2 and b=1
def f_specific (x : ℝ) : ℝ := f x 2 1

-- Statement (2): Finding the extreme values of f(x) on the interval [-1, 1]
def extreme_values_on_interval : Prop :=
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f_specific x ≤ 6 ∧ f_specific x ≥ 50/27) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f_specific x = 6) ∧ 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f_specific x = 50/27)

theorem find_extreme_values : extreme_values_on_interval :=
sorry

end find_a_b_find_extreme_values_l164_16431


namespace find_quotient_l164_16469

theorem find_quotient (D d R Q : ℤ) (hD : D = 729) (hd : d = 38) (hR : R = 7)
  (h : D = d * Q + R) : Q = 19 := by
  sorry

end find_quotient_l164_16469


namespace sum_of_first_100_digits_of_1_div_2222_l164_16425

theorem sum_of_first_100_digits_of_1_div_2222 : 
  (let repeating_block := [0, 0, 0, 4, 5];
  let sum_of_digits (lst : List ℕ) := lst.sum;
  let block_sum := sum_of_digits repeating_block;
  let num_blocks := 100 / 5;
  num_blocks * block_sum = 180) :=
by 
  let repeating_block := [0, 0, 0, 4, 5]
  let sum_of_digits (lst : List ℕ) := lst.sum
  let block_sum := sum_of_digits repeating_block
  let num_blocks := 100 / 5
  have h : num_blocks * block_sum = 180 := sorry
  exact h

end sum_of_first_100_digits_of_1_div_2222_l164_16425


namespace min_moves_to_visit_all_non_forbidden_squares_l164_16474

def min_diagonal_moves (n : ℕ) : ℕ :=
  2 * (n / 2) - 1

theorem min_moves_to_visit_all_non_forbidden_squares (n : ℕ) :
  min_diagonal_moves n = 2 * (n / 2) - 1 := by
  sorry

end min_moves_to_visit_all_non_forbidden_squares_l164_16474


namespace subset_iff_a_values_l164_16441

theorem subset_iff_a_values (a : ℝ) :
  let P := { x : ℝ | x^2 = 1 }
  let Q := { x : ℝ | a * x = 1 }
  Q ⊆ P ↔ a = 0 ∨ a = 1 ∨ a = -1 :=
by sorry

end subset_iff_a_values_l164_16441


namespace number_of_solutions_eq_two_l164_16446

theorem number_of_solutions_eq_two : 
  (∃ (x y : ℝ), x^2 - 3*x - 4 = 0 ∧ y^2 - 6*y + 9 = 0) ∧
  (∀ (x y : ℝ), (x^2 - 3*x - 4 = 0 ∧ y^2 - 6*y + 9 = 0) → ((x = 4 ∨ x = -1) ∧ y = 3)) :=
by
  sorry

end number_of_solutions_eq_two_l164_16446


namespace junior_titles_in_sample_l164_16458

noncomputable def numberOfJuniorTitlesInSample (totalEmployees: ℕ) (juniorEmployees: ℕ) (sampleSize: ℕ) : ℕ :=
  (juniorEmployees * sampleSize) / totalEmployees

theorem junior_titles_in_sample (totalEmployees juniorEmployees intermediateEmployees seniorEmployees sampleSize : ℕ) 
  (h_total : totalEmployees = 150) 
  (h_junior : juniorEmployees = 90) 
  (h_intermediate : intermediateEmployees = 45) 
  (h_senior : seniorEmployees = 15) 
  (h_sampleSize : sampleSize = 30) : 
  numberOfJuniorTitlesInSample totalEmployees juniorEmployees sampleSize = 18 := by
  sorry

end junior_titles_in_sample_l164_16458


namespace infinite_either_interval_exists_rational_infinite_elements_l164_16410

variable {ε : ℝ} (x : ℕ → ℝ) (hε : ε > 0) (hεlt : ε < 1/2)

-- Problem 1
theorem infinite_either_interval (x : ℕ → ℝ) (hx : ∀ n, 0 ≤ x n ∧ x n < 1) :
  (∃ N : ℕ, ∀ n ≥ N, x n < 1/2) ∨ (∃ N : ℕ, ∀ n ≥ N, x n ≥ 1/2) :=
sorry

-- Problem 2
theorem exists_rational_infinite_elements (x : ℕ → ℝ) (hx : ∀ n, 0 ≤ x n ∧ x n < 1) (hε : ε > 0) (hεlt : ε < 1/2) :
  ∃ (α : ℚ), 0 ≤ α ∧ α ≤ 1 ∧ ∃ N : ℕ, ∀ n ≥ N, x n ∈ [α - ε, α + ε] :=
sorry

end infinite_either_interval_exists_rational_infinite_elements_l164_16410


namespace part_1_part_2_l164_16447

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x + a) + 2 * a

theorem part_1 (h : ∀ x : ℝ, f x a = f (3 - x) a) : a = -3 :=
by
  sorry

theorem part_2 (h : ∃ x : ℝ, f x a ≤ -abs (2 * x - 1) + a) : a ≤ -1 / 2 :=
by
  sorry

end part_1_part_2_l164_16447


namespace ara_current_height_l164_16440

variable (h : ℝ)  -- Original height of both Shea and Ara
variable (sheas_growth_rate : ℝ := 0.20)  -- Shea's growth rate (20%)
variable (sheas_current_height : ℝ := 60)  -- Shea's current height
variable (aras_growth_rate : ℝ := 0.5)  -- Ara's growth rate in terms of Shea's growth

theorem ara_current_height : 
  h * (1 + sheas_growth_rate) = sheas_current_height →
  (h + (sheas_current_height - h) * aras_growth_rate) = 55 :=
  by
    sorry

end ara_current_height_l164_16440


namespace range_of_a_l164_16476

theorem range_of_a (a : ℝ) (h : (2 - a)^3 > (a - 1)^3) : a < 3/2 :=
sorry

end range_of_a_l164_16476


namespace factorize_cubic_l164_16475

theorem factorize_cubic (a : ℝ) : a^3 - 16 * a = a * (a + 4) * (a - 4) :=
sorry

end factorize_cubic_l164_16475


namespace tim_original_vocab_l164_16404

theorem tim_original_vocab (days_in_year : ℕ) (years : ℕ) (learned_per_day : ℕ) (vocab_increase : ℝ) :
  let days := days_in_year * years
  let learned_words := learned_per_day * days
  let original_vocab := learned_words / vocab_increase
  original_vocab = 14600 :=
by
  let days := days_in_year * years
  let learned_words := learned_per_day * days
  let original_vocab := learned_words / vocab_increase
  show original_vocab = 14600
  sorry

end tim_original_vocab_l164_16404


namespace green_square_area_percentage_l164_16450

variable (s a : ℝ)
variable (h : a^2 + 4 * a * (s - 2 * a) = 0.49 * s^2)

theorem green_square_area_percentage :
  (a^2 / s^2) = 0.1225 :=
sorry

end green_square_area_percentage_l164_16450


namespace is_linear_equation_l164_16432

def quadratic_equation (x y : ℝ) : Prop := x * y + 2 * x = 7
def fractional_equation (x y : ℝ) : Prop := (1 / x) + y = 5
def quadratic_equation_2 (x y : ℝ) : Prop := x^2 + y = 2

def linear_equation (x y : ℝ) : Prop := 2 * x - y = 2

theorem is_linear_equation (x y : ℝ) (h1 : quadratic_equation x y) (h2 : fractional_equation x y) (h3 : quadratic_equation_2 x y) : linear_equation x y :=
  sorry

end is_linear_equation_l164_16432


namespace find_a_l164_16419

def line1 (a : ℝ) (P : ℝ × ℝ) : Prop := 2 * P.1 - a * P.2 - 1 = 0

def line2 (P : ℝ × ℝ) : Prop := P.1 + 2 * P.2 = 0

theorem find_a (a : ℝ) :
  (∀ P : ℝ × ℝ, line1 a P ∧ line2 P) → a = 1 := by
  sorry

end find_a_l164_16419


namespace division_of_power_l164_16454

theorem division_of_power (m : ℕ) 
  (h : m = 16^2018) : m / 8 = 2^8069 := by
  sorry

end division_of_power_l164_16454


namespace fraction_work_AC_l164_16435

theorem fraction_work_AC (total_payment Rs B_payment : ℝ)
  (payment_AC : ℝ)
  (h1 : total_payment = 529)
  (h2 : B_payment = 12)
  (h3 : payment_AC = total_payment - B_payment) : 
  payment_AC / total_payment = 517 / 529 :=
by
  rw [h1, h2] at h3
  rw [h3]
  norm_num
  sorry

end fraction_work_AC_l164_16435


namespace thirty_percent_less_than_ninety_eq_one_fourth_more_than_n_l164_16467

theorem thirty_percent_less_than_ninety_eq_one_fourth_more_than_n (n : ℝ) :
  0.7 * 90 = (5 / 4) * n → n = 50.4 :=
by sorry

end thirty_percent_less_than_ninety_eq_one_fourth_more_than_n_l164_16467


namespace overall_average_score_l164_16429

structure Club where
  members : Nat
  average_score : Nat

def ClubA : Club := { members := 40, average_score := 90 }
def ClubB : Club := { members := 50, average_score := 81 }

theorem overall_average_score : 
  (ClubA.members * ClubA.average_score + ClubB.members * ClubB.average_score) / 
  (ClubA.members + ClubB.members) = 85 :=
by
  sorry

end overall_average_score_l164_16429


namespace smallest_natural_with_50_perfect_squares_l164_16449

theorem smallest_natural_with_50_perfect_squares (a : ℕ) (h : a = 4486) :
  (∃ n, n^2 ≤ a ∧ (n+50)^2 < 3 * a ∧ (∀ b, n^2 ≤ b^2 ∧ b^2 < 3 * a → n ≤ b-1 ∧ b-1 ≤ n+49)) :=
by {
  sorry
}

end smallest_natural_with_50_perfect_squares_l164_16449


namespace values_of_a_plus_b_l164_16462

theorem values_of_a_plus_b (a b : ℝ) (h1 : abs (-a) = abs (-1)) (h2 : b^2 = 9) (h3 : abs (a - b) = b - a) : a + b = 2 ∨ a + b = 4 := 
by 
  sorry

end values_of_a_plus_b_l164_16462


namespace fraction_capacity_noah_ali_l164_16430

def capacity_Ali_closet : ℕ := 200
def total_capacity_Noah_closet : ℕ := 100
def each_capacity_Noah_closet : ℕ := total_capacity_Noah_closet / 2

theorem fraction_capacity_noah_ali : (each_capacity_Noah_closet : ℚ) / capacity_Ali_closet = 1 / 4 :=
by sorry

end fraction_capacity_noah_ali_l164_16430


namespace hyperbola_foci_condition_l164_16483

theorem hyperbola_foci_condition (m n : ℝ) (h : m * n > 0) :
    (m > 0 ∧ n > 0) ↔ ((∃ (x y : ℝ), m * x^2 - n * y^2 = 1) ∧ (∃ (x y : ℝ), m * x^2 - n * y^2 = 1)) :=
sorry

end hyperbola_foci_condition_l164_16483


namespace average_rainfall_february_1964_l164_16436

theorem average_rainfall_february_1964 :
  let total_rainfall := 280
  let days_february := 29
  let hours_per_day := 24
  (total_rainfall / (days_february * hours_per_day)) = (280 / (29 * 24)) :=
by
  sorry

end average_rainfall_february_1964_l164_16436


namespace quadratic_equation_l164_16413

theorem quadratic_equation (p q : ℝ) 
  (h1 : p^2 + 9 * q^2 + 3 * p - p * q = 30)
  (h2 : p - 5 * q - 8 = 0) : 
  p^2 - p - 6 = 0 :=
by sorry

end quadratic_equation_l164_16413


namespace probability_all_and_at_least_one_pass_l164_16443

-- Define conditions
def pA : ℝ := 0.8
def pB : ℝ := 0.6
def pC : ℝ := 0.5

-- Define the main theorem we aim to prove
theorem probability_all_and_at_least_one_pass :
  (pA * pB * pC = 0.24) ∧ ((1 - (1 - pA) * (1 - pB) * (1 - pC)) = 0.96) := by
  sorry

end probability_all_and_at_least_one_pass_l164_16443


namespace cost_difference_zero_l164_16485

theorem cost_difference_zero
  (A O X : ℝ)
  (h1 : 3 * A + 7 * O = 4.56)
  (h2 : A + O = 0.26)
  (h3 : O = A + X) :
  X = 0 := 
sorry

end cost_difference_zero_l164_16485


namespace orange_balls_count_l164_16400

theorem orange_balls_count (P_black : ℚ) (O : ℕ) (total_balls : ℕ) 
  (condition1 : total_balls = O + 7 + 6) 
  (condition2 : P_black = 7 / total_balls) 
  (condition3 : P_black = 0.38095238095238093) :
  O = 5 := 
by
  sorry

end orange_balls_count_l164_16400


namespace third_row_number_of_trees_l164_16401

theorem third_row_number_of_trees (n : ℕ) 
  (divisible_by_7 : 84 % 7 = 0) 
  (divisible_by_6 : 84 % 6 = 0) 
  (divisible_by_n : 84 % n = 0) 
  (least_trees : 84 = 84): 
  n = 4 := 
sorry

end third_row_number_of_trees_l164_16401


namespace matrix_addition_l164_16496

def M1 : Matrix (Fin 3) (Fin 3) ℤ :=
![![4, 1, -3],
  ![0, -2, 5],
  ![7, 0, 1]]

def M2 : Matrix (Fin 3) (Fin 3) ℤ :=
![![ -6,  9, 2],
  ![  3, -4, -8],
  ![  0,  5, -3]]

def M3 : Matrix (Fin 3) (Fin 3) ℤ :=
![![ -2, 10, -1],
  ![  3, -6, -3],
  ![  7,  5, -2]]

theorem matrix_addition : M1 + M2 = M3 := by
  sorry

end matrix_addition_l164_16496


namespace value_of_bill_used_to_pay_l164_16414

-- Definitions of the conditions
def num_games : ℕ := 6
def cost_per_game : ℕ := 15
def num_change_bills : ℕ := 2
def change_per_bill : ℕ := 5
def total_cost : ℕ := num_games * cost_per_game
def total_change : ℕ := num_change_bills * change_per_bill

-- Proof statement: What was the value of the bill Jed used to pay
theorem value_of_bill_used_to_pay : 
  total_value = (total_cost + total_change) :=
by
  sorry

end value_of_bill_used_to_pay_l164_16414


namespace crates_needed_l164_16406

def ceil_div (a b : ℕ) : ℕ := (a + b - 1) / b

theorem crates_needed :
  ceil_div 145 12 + ceil_div 271 8 + ceil_div 419 10 + ceil_div 209 14 = 104 :=
by
  sorry

end crates_needed_l164_16406


namespace geometric_sequence_third_term_l164_16411

theorem geometric_sequence_third_term (r : ℕ) (h_r : 5 * r ^ 4 = 1620) : 5 * r ^ 2 = 180 := by
  sorry

end geometric_sequence_third_term_l164_16411


namespace opposite_number_in_circle_l164_16465

theorem opposite_number_in_circle (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 200) (h3 : ∀ k, 1 ≤ k ∧ k ≤ 200 → ∃ m, (m = (k + 100) % 200) ∧ (m ≠ k) ∧ (n + 100) % 200 < k):
  ∃ m : ℕ, m = 114 ∧ (113 + 100) % 200 = m :=
by
  sorry

end opposite_number_in_circle_l164_16465


namespace bike_price_l164_16492

theorem bike_price (x : ℝ) (h : 0.20 * x = 240) : x = 1200 :=
by
  sorry

end bike_price_l164_16492


namespace new_number_is_100t_plus_10u_plus_3_l164_16405

theorem new_number_is_100t_plus_10u_plus_3 (t u : ℕ) (ht : t < 10) (hu : u < 10) :
  let original_number := 10 * t + u
  let new_number := original_number * 10 + 3
  new_number = 100 * t + 10 * u + 3 :=
by
  let original_number := 10 * t + u
  let new_number := original_number * 10 + 3
  show new_number = 100 * t + 10 * u + 3
  sorry

end new_number_is_100t_plus_10u_plus_3_l164_16405
