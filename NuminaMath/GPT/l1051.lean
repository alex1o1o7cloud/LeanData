import Mathlib

namespace max_regions_with_five_lines_l1051_105146

def max_regions (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * (n + 1) / 2 + 1

theorem max_regions_with_five_lines (n : ℕ) (h : n = 5) : max_regions n = 16 :=
by {
  rw [h, max_regions];
  norm_num;
  done
}

end max_regions_with_five_lines_l1051_105146


namespace mean_equality_l1051_105183

theorem mean_equality (y : ℝ) :
  ((3 + 7 + 11 + 15) / 4 = (10 + 14 + y) / 3) → y = 3 :=
by
  sorry

end mean_equality_l1051_105183


namespace num_of_solutions_eq_28_l1051_105111

def num_solutions : Nat :=
  sorry

theorem num_of_solutions_eq_28 : num_solutions = 28 :=
  sorry

end num_of_solutions_eq_28_l1051_105111


namespace jerry_current_average_l1051_105140

theorem jerry_current_average (A : ℚ) (h1 : 3 * A + 89 = 4 * (A + 2)) : A = 81 := 
by
  sorry

end jerry_current_average_l1051_105140


namespace acute_triangle_exists_l1051_105167

theorem acute_triangle_exists {a1 a2 a3 a4 a5 : ℝ} 
  (h1 : a1 + a2 > a3) (h2 : a1 + a3 > a2) (h3 : a2 + a3 > a1)
  (h4 : a2 + a3 > a4) (h5 : a3 + a4 > a2) (h6 : a2 + a4 > a3)
  (h7 : a3 + a4 > a5) (h8 : a4 + a5 > a3) (h9 : a3 + a5 > a4) : 
  ∃ (t1 t2 t3 : ℝ), (t1 + t2 > t3) ∧ (t1 + t3 > t2) ∧ (t2 + t3 > t1) ∧ (t3 ^ 2 < t1 ^ 2 + t2 ^ 2) :=
sorry

end acute_triangle_exists_l1051_105167


namespace contradiction_method_example_l1051_105144

variables {a b c : ℝ}
variables (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
variables (h4 : a + b + c > 0) (h5 : ab + bc + ca > 0)
variables (h6 : (a < 0 ∧ b < 0) ∨ (a < 0 ∧ c < 0) ∨ (b < 0 ∧ c < 0))

theorem contradiction_method_example : false :=
by {
  sorry
}

end contradiction_method_example_l1051_105144


namespace marissa_tied_boxes_l1051_105198

theorem marissa_tied_boxes 
  (r_total : ℝ) (r_per_box : ℝ) (r_left : ℝ) (h_total : r_total = 4.5)
  (h_per_box : r_per_box = 0.7) (h_left : r_left = 1) :
  (r_total - r_left) / r_per_box = 5 :=
by
  sorry

end marissa_tied_boxes_l1051_105198


namespace sequence_S15_is_211_l1051_105137

theorem sequence_S15_is_211 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 1) 
  (h2 : a 2 = 2)
  (h3 : ∀ n > 1, S (n + 1) + S (n - 1) = 2 * (S n + S 1)) :
  S 15 = 211 := 
sorry

end sequence_S15_is_211_l1051_105137


namespace large_monkey_doll_cost_l1051_105158

theorem large_monkey_doll_cost (S L E : ℝ) 
  (h1 : S = L - 2) 
  (h2 : E = L + 1) 
  (h3 : 300 / S = 300 / L + 25) 
  (h4 : 300 / E = 300 / L - 15) : 
  L = 6 := 
sorry

end large_monkey_doll_cost_l1051_105158


namespace least_zorgs_to_drop_more_points_than_eating_l1051_105162

theorem least_zorgs_to_drop_more_points_than_eating :
  ∃ (n : ℕ), (∀ m < n, m * (m + 1) / 2 ≤ 20 * m) ∧ n * (n + 1) / 2 > 20 * n :=
sorry

end least_zorgs_to_drop_more_points_than_eating_l1051_105162


namespace chad_ice_cost_l1051_105191

theorem chad_ice_cost
  (n : ℕ) -- Number of people
  (p : ℕ) -- Pounds of ice per person
  (c : ℝ) -- Cost per 10 pound bag of ice
  (h1 : n = 20) 
  (h2 : p = 3)
  (h3 : c = 4.5) :
  (3 * 20 / 10) * 4.5 = 27 :=
by
  sorry

end chad_ice_cost_l1051_105191


namespace vertex_angle_isosceles_triangle_l1051_105195

theorem vertex_angle_isosceles_triangle (B V : ℝ) (h1 : 2 * B + V = 180) (h2 : B = 40) : V = 100 :=
by
  sorry

end vertex_angle_isosceles_triangle_l1051_105195


namespace sequence_term_l1051_105107

theorem sequence_term (S : ℕ → ℕ) (a : ℕ → ℕ) (n : ℕ) (hn : n > 0)
  (hSn : ∀ n, S n = n^2)
  (hrec : ∀ n, n > 1 → a n = S n - S (n-1)) :
  a n = 2 * n - 1 := by
  -- Base case
  cases n with
  | zero => contradiction  -- n > 0 implies n ≠ 0
  | succ n' =>
    cases n' with
    | zero => sorry  -- When n = 0 + 1 = 1, we need to show a 1 = 2 * 1 - 1 = 1 based on given conditions
    | succ k => sorry -- When n = k + 1, we use the provided recursive relation to prove the statement

end sequence_term_l1051_105107


namespace max_possible_median_l1051_105197

/-- 
Given:
1. The Beverage Barn sold 300 cans of soda to 120 customers.
2. Every customer bought at least 1 can of soda but no more than 5 cans.
Prove that the maximum possible median number of cans of soda bought per customer is 5.
-/
theorem max_possible_median (total_cans : ℕ) (customers : ℕ) (min_can_per_customer : ℕ) (max_can_per_customer : ℕ) :
  total_cans = 300 ∧ customers = 120 ∧ min_can_per_customer = 1 ∧ max_can_per_customer = 5 →
  (∃ median : ℕ, median = 5) :=
by
  sorry

end max_possible_median_l1051_105197


namespace unique_solution_l1051_105150

theorem unique_solution (k : ℝ) (h : k + 1 ≠ 0) : 
  (∀ x y : ℝ, ((x + 3) / (k * x + x - 3) = x) → ((y + 3) / (k * y + y - 3) = y) → x = y) ↔ k = -7/3 :=
by sorry

end unique_solution_l1051_105150


namespace required_extra_money_l1051_105110

theorem required_extra_money 
(Patricia_money Lisa_money Charlotte_money : ℕ) 
(hP : Patricia_money = 6) 
(hL : Lisa_money = 5 * Patricia_money) 
(hC : Lisa_money = 2 * Charlotte_money) 
(cost : ℕ) 
(hCost : cost = 100) : 
  cost - (Patricia_money + Lisa_money + Charlotte_money) = 49 := 
by 
  sorry

end required_extra_money_l1051_105110


namespace horse_revolutions_l1051_105119

theorem horse_revolutions (r1 r2 r3 : ℝ) (rev1 : ℕ) 
  (h1 : r1 = 30) (h2 : r2 = 15) (h3 : r3 = 10) (h4 : rev1 = 40) :
  (r2 / r1 = 1 / 2 ∧ 2 * rev1 = 80) ∧ (r3 / r1 = 1 / 3 ∧ 3 * rev1 = 120) :=
by
  sorry

end horse_revolutions_l1051_105119


namespace fraction_complex_eq_l1051_105176

theorem fraction_complex_eq (z : ℂ) (h : z = 2 + I) : 2 * I / (z - 1) = 1 + I := by
  sorry

end fraction_complex_eq_l1051_105176


namespace distance_probability_l1051_105152

theorem distance_probability :
  let speed := 5
  let num_roads := 8
  let total_outcomes := num_roads * (num_roads - 1)
  let favorable_outcomes := num_roads * 3
  let probability := (favorable_outcomes : ℝ) / total_outcomes
  probability = 0.375 :=
by
  sorry

end distance_probability_l1051_105152


namespace total_ticket_revenue_l1051_105174

theorem total_ticket_revenue (total_seats : Nat) (price_adult_ticket : Nat) (price_child_ticket : Nat)
  (theatre_full : Bool) (child_tickets : Nat) (adult_tickets := total_seats - child_tickets)
  (rev_adult := adult_tickets * price_adult_ticket) (rev_child := child_tickets * price_child_ticket) :
  total_seats = 250 →
  price_adult_ticket = 6 →
  price_child_ticket = 4 →
  theatre_full = true →
  child_tickets = 188 →
  rev_adult + rev_child = 1124 := 
by
  intros h_total_seats h_price_adult h_price_child h_theatre_full h_child_tickets
  sorry

end total_ticket_revenue_l1051_105174


namespace computation_is_correct_l1051_105131

def large_multiplication : ℤ := 23457689 * 84736521

def denominator_subtraction : ℤ := 7589236 - 3145897

def computed_m : ℚ := large_multiplication / denominator_subtraction

theorem computation_is_correct : computed_m = 447214.999 :=
by 
  -- exact calculation to be provided
  sorry

end computation_is_correct_l1051_105131


namespace positional_relationship_l1051_105139

-- Definitions of the lines l1 and l2
def l1 (m x y : ℝ) : Prop := (m + 3) * x + 5 * y = 5 - 3 * m
def l2 (m x y : ℝ) : Prop := 2 * x + (m + 6) * y = 8

theorem positional_relationship (m : ℝ) :
  (∃ x y : ℝ, l1 m x y ∧ l2 m x y) ∨ (∀ x y : ℝ, l1 m x y ↔ l2 m x y) ∨
  ¬(∃ x y : ℝ, l1 m x y ∨ l2 m x y) :=
sorry

end positional_relationship_l1051_105139


namespace num_sets_N_l1051_105129

open Set

-- Define the set M and the set U
def M : Set ℕ := {1, 2}
def U : Set ℕ := {1, 2, 3, 4}

-- The statement to prove
theorem num_sets_N : 
  ∃ count : ℕ, count = 4 ∧ 
  (∀ N : Set ℕ, M ∪ N = U → N = {3, 4} ∨ N = {1, 3, 4} ∨ N = {2, 3, 4} ∨ N = {1, 2, 3, 4}) :=
by
  sorry

end num_sets_N_l1051_105129


namespace cube_volume_l1051_105122

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l1051_105122


namespace intersection_volume_l1051_105168

noncomputable def volume_of_intersection (k : ℝ) : ℝ :=
  ∫ x in -k..k, 4 * (k^2 - x^2)

theorem intersection_volume (k : ℝ) : volume_of_intersection k = 16 * k^3 / 3 :=
  by
  sorry

end intersection_volume_l1051_105168


namespace slope_divides_polygon_area_l1051_105101

structure Point where
  x : ℝ
  y : ℝ

noncomputable def polygon_vertices : List Point :=
  [⟨0, 0⟩, ⟨0, 4⟩, ⟨4, 4⟩, ⟨4, 2⟩, ⟨7, 2⟩, ⟨7, 0⟩]

-- Define the area calculation and conditions needed 
noncomputable def area_of_polygon (vertices : List Point) : ℝ :=
  -- Assuming here that a function exists to calculate the area given the vertices
  sorry

def line_through_origin (slope : ℝ) (x : ℝ) : Point :=
  ⟨x, slope * x⟩

theorem slope_divides_polygon_area :
  let line := line_through_origin (2 / 7)
  ∀ x : ℝ, ∃ (G : Point), 
  polygon_vertices = [⟨0, 0⟩, ⟨0, 4⟩, ⟨4, 4⟩, ⟨4, 2⟩, ⟨7, 2⟩, ⟨7, 0⟩] →
  area_of_polygon polygon_vertices / 2 = 
  area_of_polygon [⟨0, 0⟩, line x, G] :=
sorry

end slope_divides_polygon_area_l1051_105101


namespace threshold_mu_l1051_105145

/-- 
Find threshold values μ₁₀₀ and μ₁₀₀₀₀₀ such that 
F = m * n * sin (π / m) * sqrt (1 / n² + sin⁴ (π / m)) 
is definitely greater than 100 and 1,000,000 respectively for all m greater than μ₁₀₀ and μ₁₀₀₀₀₀, 
assuming n = m³. -/
theorem threshold_mu : 
  (∃ (μ₁₀₀ μ₁₀₀₀₀₀ : ℝ), ∀ (m : ℝ), m > μ₁₀₀ → 
    m * (m ^ 3) * (Real.sin (Real.pi / m)) * 
      (Real.sqrt ((1 : ℝ) / (m ^ 6) + (Real.sin (Real.pi / m)) ^ 4)) > 100) ∧ 
  (∃ (μ₁₀₀₀₀₀ μ₁₀₀₀₀₀ : ℝ), ∀ (m : ℝ), m > μ₁₀₀₀₀₀ → 
    m * (m ^ 3) * (Real.sin (Real.pi / m)) * 
      (Real.sqrt ((1 : ℝ) / (m ^ 6) + (Real.sin (Real.pi / m)) ^ 4)) > 1000000) :=
sorry

end threshold_mu_l1051_105145


namespace solution_set_of_inequality_l1051_105181

theorem solution_set_of_inequality :
  {x : ℝ | 4 * x ^ 2 - 4 * x + 1 ≤ 0} = {1 / 2} :=
by
  sorry

end solution_set_of_inequality_l1051_105181


namespace general_term_arithmetic_sequence_l1051_105125

theorem general_term_arithmetic_sequence (a : ℕ → ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = a n + 2) : 
  ∀ n, a n = 2 * n - 1 := 
by 
  sorry

end general_term_arithmetic_sequence_l1051_105125


namespace digit_Phi_l1051_105157

theorem digit_Phi (Phi : ℕ) (h1 : 220 / Phi = 40 + 3 * Phi) : Phi = 4 :=
by
  sorry

end digit_Phi_l1051_105157


namespace opposite_of_neg_3_is_3_l1051_105100

theorem opposite_of_neg_3_is_3 : -(-3) = 3 := by
  sorry

end opposite_of_neg_3_is_3_l1051_105100


namespace ravi_first_has_more_than_500_paperclips_on_wednesday_l1051_105179

noncomputable def paperclips (k : Nat) : Nat :=
  5 * 4^k

theorem ravi_first_has_more_than_500_paperclips_on_wednesday :
  ∃ k : Nat, paperclips k > 500 ∧ k = 3 :=
by
  sorry

end ravi_first_has_more_than_500_paperclips_on_wednesday_l1051_105179


namespace symmetric_point_coordinates_l1051_105102

theorem symmetric_point_coordinates (Q : ℝ × ℝ × ℝ) 
  (P : ℝ × ℝ × ℝ := (-6, 7, -9)) 
  (A : ℝ × ℝ × ℝ := (1, 3, -1)) 
  (B : ℝ × ℝ × ℝ := (6, 5, -2)) 
  (C : ℝ × ℝ × ℝ := (0, -3, -5)) : Q = (2, -5, 7) :=
sorry

end symmetric_point_coordinates_l1051_105102


namespace arithmetic_sequence_general_term_and_k_l1051_105136

theorem arithmetic_sequence_general_term_and_k (a : ℕ → ℚ) (d : ℚ)
  (h1 : a 4 + a 7 + a 10 = 17)
  (h2 : a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12 + a 13 + a 14 = 77) :
  (∀ n : ℕ, a n = (2 * n + 3) / 3) ∧ (∃ k : ℕ, a k = 13 ∧ k = 18) := 
by
  sorry

end arithmetic_sequence_general_term_and_k_l1051_105136


namespace total_wings_l1051_105171

-- Conditions
def money_per_grandparent : ℕ := 50
def number_of_grandparents : ℕ := 4
def bird_cost : ℕ := 20
def wings_per_bird : ℕ := 2

-- Calculate the total amount of money John received:
def total_money_received : ℕ := number_of_grandparents * money_per_grandparent

-- Determine the number of birds John can buy:
def number_of_birds : ℕ := total_money_received / bird_cost

-- Prove that the total number of wings all the birds have is 20:
theorem total_wings : number_of_birds * wings_per_bird = 20 :=
by
  sorry

end total_wings_l1051_105171


namespace ratio_of_x_y_l1051_105192

theorem ratio_of_x_y (x y : ℝ) (h : x + y = 3 * (x - y)) : x / y = 2 :=
by
  sorry

end ratio_of_x_y_l1051_105192


namespace trig_identity_l1051_105166

theorem trig_identity 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < (Real.pi / 2)) 
  (h3 : Real.tan θ = 1 / 3) :
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 :=
by
  sorry

end trig_identity_l1051_105166


namespace samuel_distance_from_hotel_l1051_105188

/-- Samuel's driving problem conditions. -/
structure DrivingConditions where
  total_distance : ℕ -- in miles
  first_speed : ℕ -- in miles per hour
  first_time : ℕ -- in hours
  second_speed : ℕ -- in miles per hour
  second_time : ℕ -- in hours

def distance_remaining (c : DrivingConditions) : ℕ :=
  let distance_covered := (c.first_speed * c.first_time) + (c.second_speed * c.second_time)
  c.total_distance - distance_covered

/-- Prove that Samuel is 130 miles from the hotel. -/
theorem samuel_distance_from_hotel : 
  ∀ (c : DrivingConditions), 
    c.total_distance = 600 ∧
    c.first_speed = 50 ∧
    c.first_time = 3 ∧
    c.second_speed = 80 ∧
    c.second_time = 4 → distance_remaining c = 130 := by
  intros c h
  cases h
  sorry

end samuel_distance_from_hotel_l1051_105188


namespace lyle_percentage_l1051_105141

theorem lyle_percentage (chips : ℕ) (ian_ratio lyle_ratio : ℕ) (h_ratio_sum : ian_ratio + lyle_ratio = 10) (h_chips : chips = 100) :
  (lyle_ratio / (ian_ratio + lyle_ratio) : ℚ) * 100 = 60 := 
by
  sorry

end lyle_percentage_l1051_105141


namespace reciprocal_of_neg2_l1051_105149

-- Define the number
def num : ℤ := -2

-- Define the reciprocal function
def reciprocal (x : ℤ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_neg2 : reciprocal num = -1 / 2 :=
by
  -- Proof is skipped
  sorry

end reciprocal_of_neg2_l1051_105149


namespace total_sand_donated_l1051_105182

theorem total_sand_donated (A B C D: ℚ) (hA: A = 33 / 2) (hB: B = 26) (hC: C = 49 / 2) (hD: D = 28) : 
  A + B + C + D = 95 := by
  sorry

end total_sand_donated_l1051_105182


namespace completing_square_result_l1051_105187

theorem completing_square_result:
  ∀ x : ℝ, (x^2 + 4 * x - 1 = 0) → (x + 2)^2 = 5 :=
by
  sorry

end completing_square_result_l1051_105187


namespace arithmetic_sequence_common_difference_l1051_105185

-- Arithmetic sequence with condition and proof of common difference
theorem arithmetic_sequence_common_difference (a : ℕ → ℚ) (d : ℚ) :
  (a 2015 = a 2013 + 6) → ((a 2015 - a 2013) = 2 * d) → (d = 3) :=
by
  intro h1 h2
  sorry

end arithmetic_sequence_common_difference_l1051_105185


namespace number_of_new_students_l1051_105147

theorem number_of_new_students (initial_students end_students students_left : ℕ) 
  (h_initial: initial_students = 33) 
  (h_left: students_left = 18) 
  (h_end: end_students = 29) : 
  initial_students - students_left + (end_students - (initial_students - students_left)) = 14 :=
by
  sorry

end number_of_new_students_l1051_105147


namespace Gandalf_reachability_l1051_105173

theorem Gandalf_reachability (n : ℕ) (h : n ≥ 1) :
  ∃ (m : ℕ), m = 1 :=
sorry

end Gandalf_reachability_l1051_105173


namespace find_roots_l1051_105161

def polynomial (x: ℝ) := x^3 - 2*x^2 - x + 2

theorem find_roots : { x : ℝ // polynomial x = 0 } = ({1, -1, 2} : Set ℝ) :=
by
  sorry

end find_roots_l1051_105161


namespace g_one_minus_g_four_l1051_105133

theorem g_one_minus_g_four (g : ℝ → ℝ)
  (h_linear : ∀ x y : ℝ, g (x + y) = g x + g y)
  (h_diff : ∀ x : ℝ, g (x + 1) - g x = 5) :
  g 1 - g 4 = -15 :=
sorry

end g_one_minus_g_four_l1051_105133


namespace right_triangle_area_l1051_105127

def roots (a b : ℝ) : Prop :=
  a * b = 12 ∧ a + b = 7

def area (A : ℝ) : Prop :=
  A = 6 ∨ A = 3 * Real.sqrt 7 / 2

theorem right_triangle_area (a b A : ℝ) (h : roots a b) : area A := 
by 
  -- The proof steps would go here
  sorry

end right_triangle_area_l1051_105127


namespace number_of_ordered_triples_l1051_105123

theorem number_of_ordered_triples (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
    (h_eq : a * b * c - b * c - a * c - a * b + a + b + c = 2013) :
    ∃ n, n = 39 :=
by
  sorry

end number_of_ordered_triples_l1051_105123


namespace sum_of_discounts_l1051_105112

theorem sum_of_discounts
  (price_fox : ℝ)
  (price_pony : ℝ)
  (savings : ℝ)
  (discount_pony : ℝ) :
  (3 * price_fox * (F / 100) + 2 * price_pony * (discount_pony / 100) = savings) →
  (F + discount_pony = 22) :=
sorry


end sum_of_discounts_l1051_105112


namespace smallest_resolvable_debt_l1051_105134

def pig_value : ℤ := 450
def goat_value : ℤ := 330
def gcd_pig_goat : ℤ := Int.gcd pig_value goat_value

theorem smallest_resolvable_debt :
  ∃ p g : ℤ, gcd_pig_goat * 4 = pig_value * p + goat_value * g := 
by
  sorry

end smallest_resolvable_debt_l1051_105134


namespace cattle_area_correct_l1051_105194

-- Definitions based on the problem conditions
def length_km := 3.6
def width_km := 2.5 * length_km
def total_area_km2 := length_km * width_km
def cattle_area_km2 := total_area_km2 / 2

-- Theorem statement
theorem cattle_area_correct : cattle_area_km2 = 16.2 := by
  sorry

end cattle_area_correct_l1051_105194


namespace ruby_height_is_192_l1051_105199

def height_janet := 62
def height_charlene := 2 * height_janet
def height_pablo := height_charlene + 70
def height_ruby := height_pablo - 2

theorem ruby_height_is_192 : height_ruby = 192 := by
  sorry

end ruby_height_is_192_l1051_105199


namespace nth_term_pattern_l1051_105164

theorem nth_term_pattern (a : ℕ → ℕ) (h : ∀ n, a n = n * (n - 1)) : 
  (a 0 = 0) ∧ (a 1 = 2) ∧ (a 2 = 6) ∧ (a 3 = 12) ∧ (a 4 = 20) ∧ 
  (a 5 = 30) ∧ (a 6 = 42) ∧ (a 7 = 56) ∧ (a 8 = 72) ∧ (a 9 = 90) := sorry

end nth_term_pattern_l1051_105164


namespace andrew_age_l1051_105138

theorem andrew_age (a g : ℝ) (h1 : g = 9 * a) (h2 : g - a = 63) : a = 7.875 :=
by
  sorry

end andrew_age_l1051_105138


namespace smallest_range_l1051_105126

theorem smallest_range {x1 x2 x3 x4 x5 : ℝ} 
  (h1 : (x1 + x2 + x3 + x4 + x5) = 100)
  (h2 : x3 = 18)
  (h3 : 2 * x1 + 2 * x5 + 18 = 100): 
  x5 - x1 = 19 :=
by {
  sorry
}

end smallest_range_l1051_105126


namespace find_value_of_function_l1051_105155

theorem find_value_of_function (f : ℝ → ℝ) (h : ∀ x : ℝ, 2 * f x + f (-x) = 3 * x + 2) : 
  f 2 = 20 / 3 :=
sorry

end find_value_of_function_l1051_105155


namespace calc_expr_l1051_105118

theorem calc_expr : 4⁻¹ - Real.sqrt (1 / 16) + (3 - Real.sqrt 2)^0 = 1 := by
  sorry

end calc_expr_l1051_105118


namespace linear_function_quadrants_l1051_105103

theorem linear_function_quadrants (k b : ℝ) 
  (h1 : k < 0)
  (h2 : b < 0) 
  : k * b > 0 := 
sorry

end linear_function_quadrants_l1051_105103


namespace valid_functions_l1051_105124

theorem valid_functions (g : ℝ → ℝ) (h : ∀ x y : ℝ, g (x + y) * g (x - y) = (g x + g y)^2 - 4 * x^2 * g y + 2 * y^2 * g x) :
  (∀ x, g x = 0) ∨ (∀ x, g x = x^2) :=
by sorry

end valid_functions_l1051_105124


namespace games_within_division_l1051_105116

theorem games_within_division (N M : ℕ) (h1 : N > 3 * M) (h2 : M > 5) (h3 : 2 * N + 6 * M = 76) : 2 * N = 40 :=
by {
  sorry
}

end games_within_division_l1051_105116


namespace exam_percentage_l1051_105163

theorem exam_percentage (x : ℝ) (h_cond : 100 - x >= 0 ∧ x >= 0 ∧ 60 * x + 90 * (100 - x) = 69 * 100) : x = 70 := by
  sorry

end exam_percentage_l1051_105163


namespace calculate_molecular_weight_CaBr2_l1051_105159

def atomic_weight_Ca : ℝ := 40.08                 -- The atomic weight of calcium (Ca)
def atomic_weight_Br : ℝ := 79.904                -- The atomic weight of bromine (Br)
def molecular_weight_CaBr2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_Br  -- Definition of molecular weight of CaBr₂

theorem calculate_molecular_weight_CaBr2 : molecular_weight_CaBr2 = 199.888 := by
  sorry

end calculate_molecular_weight_CaBr2_l1051_105159


namespace volume_range_of_rectangular_solid_l1051_105128

theorem volume_range_of_rectangular_solid
  (a b c : ℝ)
  (h1 : 2 * (a * b + b * c + c * a) = 48)
  (h2 : 4 * (a + b + c) = 36) :
  (16 : ℝ) ≤ a * b * c ∧ a * b * c ≤ 20 :=
by sorry

end volume_range_of_rectangular_solid_l1051_105128


namespace utensils_in_each_pack_l1051_105184

/-- Prove that given John needs to buy 5 packs to get 50 spoons
    and each pack contains an equal number of knives, forks, and spoons,
    the total number of utensils in each pack is 30. -/
theorem utensils_in_each_pack
  (packs : ℕ)
  (total_spoons : ℕ)
  (equal_parts : ∀ p : ℕ, p = total_spoons / packs)
  (knives forks spoons : ℕ)
  (equal_utensils : ∀ u : ℕ, u = spoons)
  (knives_forks : knives = forks)
  (knives_spoons : knives = spoons)
  (packs_needed : packs = 5)
  (total_utensils_needed : total_spoons = 50) :
  knives + forks + spoons = 30 := by
  sorry

end utensils_in_each_pack_l1051_105184


namespace correct_quadratic_equation_l1051_105106

-- The main statement to prove.
theorem correct_quadratic_equation :
  (∀ (x y a : ℝ), (3 * x + 2 * y - 1 ≠ 0) ∧ (5 * x^2 - 6 * y - 3 ≠ 0) ∧ (a * x^2 - x + 2 ≠ 0) ∧ (x^2 - 1 = 0) → (x^2 - 1 = 0)) :=
by
  sorry

end correct_quadratic_equation_l1051_105106


namespace sufficient_but_not_necessary_condition_l1051_105180

theorem sufficient_but_not_necessary_condition :
  (∀ (x : ℝ), x = 1 → x^2 - 3 * x + 2 = 0) ∧ ¬(∀ (x : ℝ), x^2 - 3 * x + 2 = 0 → x = 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1051_105180


namespace inequality_proof_l1051_105190

open Real

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / (x^2 + y * z) + 1 / (y^2 + z * x) + 1 / (z^2 + x * y)) ≤ 
  (1 / 2) * (1 / (x * y) + 1 / (y * z) + 1 / (z * x)) :=
by sorry

end inequality_proof_l1051_105190


namespace solution_problem_l1051_105196

noncomputable def problem :=
  ∀ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 1 →
  2 ≤ (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ∧
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≤ (1 + a) * (1 + b) * (1 + c)

theorem solution_problem : problem :=
  sorry

end solution_problem_l1051_105196


namespace necessary_but_not_sufficient_condition_l1051_105165

theorem necessary_but_not_sufficient_condition {a b : ℝ} (ha : a > 0) (hb : b > 0) : 
  ((a + b > 1) ↔ (ab > 1)) → false :=
by
  sorry

end necessary_but_not_sufficient_condition_l1051_105165


namespace num_valid_n_l1051_105154

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (Nat.succ n') => Nat.succ n' * factorial n'

def divisible (a b : ℕ) : Prop := b ∣ a

theorem num_valid_n (N : ℕ) :
  N ≤ 30 → 
  ¬ (∃ k, k + 1 ≤ 31 ∧ k + 1 > 1 ∧ (Prime (k + 1)) ∧ ¬ divisible (2 * factorial (k - 1)) (k + 1)) →
  ∃ m : ℕ, m = 20 :=
by
  sorry

end num_valid_n_l1051_105154


namespace distance_traveled_l1051_105169

-- Define the conditions
def rate : Real := 60  -- rate of 60 miles per hour
def total_break_time : Real := 1  -- total break time of 1 hour
def total_trip_time : Real := 9  -- total trip time of 9 hours

-- The theorem to prove the distance traveled
theorem distance_traveled : rate * (total_trip_time - total_break_time) = 480 := 
by
  sorry

end distance_traveled_l1051_105169


namespace minimum_value_of_a_l1051_105105

theorem minimum_value_of_a :
  (∀ x : ℝ, x > 0 → (a : ℝ) * x * Real.exp x - x - Real.log x ≥ 0) → a ≥ 1 / Real.exp 1 :=
by
  sorry

end minimum_value_of_a_l1051_105105


namespace total_airflow_in_one_week_l1051_105109

-- Define the conditions
def airflow_rate : ℕ := 10 -- liters per second
def working_time_per_day : ℕ := 10 -- minutes per day
def days_per_week : ℕ := 7

-- Define the conversion factors
def minutes_to_seconds : ℕ := 60

-- Define the total working time in seconds
def total_working_time_per_week : ℕ := working_time_per_day * days_per_week * minutes_to_seconds

-- Define the expected total airflow in one week
def expected_total_airflow : ℕ := airflow_rate * total_working_time_per_week

-- Prove that the expected total airflow is 42000 liters
theorem total_airflow_in_one_week : expected_total_airflow = 42000 := 
by
  -- assertion is correct given the conditions above 
  -- skip the proof
  sorry

end total_airflow_in_one_week_l1051_105109


namespace ratio_of_arithmetic_seqs_l1051_105121

noncomputable def arithmetic_seq_sum (a_1 a_n : ℕ) (n : ℕ) : ℝ := (n * (a_1 + a_n)) / 2

theorem ratio_of_arithmetic_seqs (a_1 a_6 a_11 b_1 b_6 b_11 : ℕ) :
  (∀ n : ℕ, (arithmetic_seq_sum a_1 a_n n) / (arithmetic_seq_sum b_1 b_n n) = n / (2 * n + 1))
  → (a_1 + a_6) / (b_1 + b_6) = 6 / 13
  → (a_1 + a_11) / (b_1 + b_11) = 11 / 23
  → (a_6 : ℝ) / (b_6 : ℝ) = 11 / 23 :=
  by
    intros h₁₁ h₆ h₁₁b
    sorry

end ratio_of_arithmetic_seqs_l1051_105121


namespace marie_gift_boxes_l1051_105143

theorem marie_gift_boxes
  (total_eggs : ℕ)
  (weight_per_egg : ℕ)
  (remaining_weight : ℕ)
  (melted_eggs_weight : ℕ)
  (eggs_per_box : ℕ)
  (total_boxes : ℕ)
  (H1 : total_eggs = 12)
  (H2 : weight_per_egg = 10)
  (H3 : remaining_weight = 90)
  (H4 : melted_eggs_weight = total_eggs * weight_per_egg - remaining_weight)
  (H5 : melted_eggs_weight / weight_per_egg = eggs_per_box)
  (H6 : total_eggs / eggs_per_box = total_boxes) :
  total_boxes = 4 := 
sorry

end marie_gift_boxes_l1051_105143


namespace window_total_width_l1051_105156

theorem window_total_width 
  (panes : Nat := 6)
  (ratio_height_width : ℤ := 3)
  (border_width : ℤ := 1)
  (rows : Nat := 2)
  (columns : Nat := 3)
  (pane_width : ℤ := 12) :
  3 * pane_width + 2 * border_width + 2 * border_width = 40 := 
by
  sorry

end window_total_width_l1051_105156


namespace sum_of_digits_of_n_l1051_105177

theorem sum_of_digits_of_n :
  ∃ n : ℕ,
    n > 2000 ∧
    n + 135 % 75 = 15 ∧
    n + 75 % 135 = 45 ∧
    (n = 2025 ∧ (2 + 0 + 2 + 5 = 9)) :=
by
  sorry

end sum_of_digits_of_n_l1051_105177


namespace company_match_percentage_l1051_105172

theorem company_match_percentage (total_contribution : ℝ) (holly_contribution_per_paycheck : ℝ) (total_paychecks : ℕ) (total_contribution_one_year : ℝ) : 
  let holly_contribution := holly_contribution_per_paycheck * total_paychecks
  let company_contribution := total_contribution_one_year - holly_contribution
  (company_contribution / holly_contribution) * 100 = 6 :=
by
  let holly_contribution := holly_contribution_per_paycheck * total_paychecks
  let company_contribution := total_contribution_one_year - holly_contribution
  have h : holly_contribution = 2600 := by sorry
  have c : company_contribution = 156 := by sorry
  exact sorry

end company_match_percentage_l1051_105172


namespace max_bishops_1000x1000_l1051_105117

def bishop_max_non_attacking (n : ℕ) : ℕ :=
  2 * (n - 1)

theorem max_bishops_1000x1000 : bishop_max_non_attacking 1000 = 1998 :=
by sorry

end max_bishops_1000x1000_l1051_105117


namespace inequality_solution_set_l1051_105104

theorem inequality_solution_set :
  {x : ℝ | (x + 1) / (x - 3) ≥ 0} = {x : ℝ | x ≤ -1 ∨ x > 3} := 
by 
  sorry

end inequality_solution_set_l1051_105104


namespace ratio_a7_b7_l1051_105170

variables (a b : ℕ → ℤ) (Sa Tb : ℕ → ℤ)
variables (h1 : ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0))
variables (h2 : ∀ n : ℕ, b n = b 0 + n * (b 1 - b 0))
variables (h3 : ∀ n : ℕ, Sa n = n * (a 0 + a n) / 2)
variables (h4 : ∀ n : ℕ, Tb n = n * (b 0 + b n) / 2)
variables (h5 : ∀ n : ℕ, n > 0 → Sa n / Tb n = (7 * n + 1) / (4 * n + 27))

theorem ratio_a7_b7 : ∀ n : ℕ, n = 7 → a 7 / b 7 = 92 / 79 :=
by
  intros n hn_eq
  sorry

end ratio_a7_b7_l1051_105170


namespace overlap_area_rhombus_l1051_105142

noncomputable def area_of_overlap (α : ℝ) (hα : 0 < α ∧ α < π / 2) : ℝ :=
  1 / (Real.sin (α / 2))

theorem overlap_area_rhombus (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  area_of_overlap α hα = 1 / (Real.sin (α / 2)) :=
sorry

end overlap_area_rhombus_l1051_105142


namespace positive_solution_y_l1051_105114

theorem positive_solution_y (x y z : ℝ) 
  (h1 : x * y = 8 - 3 * x - 2 * y) 
  (h2 : y * z = 15 - 5 * y - 3 * z) 
  (h3 : x * z = 40 - 5 * x - 4 * z) : 
  y = 4 := 
sorry

end positive_solution_y_l1051_105114


namespace first_student_can_ensure_one_real_root_l1051_105178

noncomputable def can_first_student_ensure_one_real_root : Prop :=
  ∀ (b c : ℝ), ∃ a : ℝ, ∃ d : ℝ, ∀ (e : ℝ), 
    (d = 0 ∧ (e = b ∨ e = c)) → 
    (∀ x : ℝ, x^3 + d * x^2 + e * x + (if e = b then c else b) = 0)

theorem first_student_can_ensure_one_real_root :
  can_first_student_ensure_one_real_root := sorry

end first_student_can_ensure_one_real_root_l1051_105178


namespace right_triangle_hypotenuse_l1051_105132

theorem right_triangle_hypotenuse (a b c : ℕ) (h : a = 6) (k : b = 8) (pt : a^2 + b^2 = c^2) : c = 10 := by
  sorry

end right_triangle_hypotenuse_l1051_105132


namespace intersection_points_count_l1051_105113

theorem intersection_points_count : 
  ∃ n : ℕ, n = 2 ∧
  (∀ x ∈ (Set.Icc 0 (2 * Real.pi)), (1 + Real.sin x = 3 / 2) → n = 2) :=
sorry

end intersection_points_count_l1051_105113


namespace max_s_value_l1051_105193

variables (X Y Z P X' Y' Z' : Type)
variables (p q r XX' YY' ZZ' s : ℝ)

-- Defining the conditions
def triangle_XYZ (p q r : ℝ) : Prop :=
p ≤ r ∧ r ≤ q ∧ p + q > r ∧ p + r > q ∧ q + r > p

def point_P_inside (X Y Z P : Type) : Prop :=
true -- Simplified assumption since point P is given to be inside

def segments_XX'_YY'_ZZ' (XX' YY' ZZ' : ℝ) : ℝ :=
XX' + YY' + ZZ'

def given_ratio (p q r : ℝ) : Prop :=
(p / (q + r)) = (r / (p + q))

-- The maximum value of s being 3p
def max_value_s_eq_3p (s p : ℝ) : Prop :=
s = 3 * p

-- The final theorem statement
theorem max_s_value 
  (p q r XX' YY' ZZ' s : ℝ)
  (h_triangle : triangle_XYZ p q r)
  (h_ratio : given_ratio p q r)
  (h_segments : s = segments_XX'_YY'_ZZ' XX' YY' ZZ') : 
  max_value_s_eq_3p s p :=
by
  sorry

end max_s_value_l1051_105193


namespace problem_statement_l1051_105153

-- Defining the real numbers and the hypothesis
variables {a b c x y z : ℝ}
variables (h1 : 17 * x + b * y + c * z = 0)
variables (h2 : a * x + 31 * y + c * z = 0)
variables (h3 : a * x + b * y + 53 * z = 0)
variables (ha : a ≠ 17)
variables (hx : x ≠ 0)

-- State the theorem
theorem problem_statement : 
  (a / (a - 17) + b / (b - 31) + c / (c - 53) = 1) :=
by
  sorry

end problem_statement_l1051_105153


namespace movies_shown_eq_twenty_four_l1051_105120

-- Define conditions
variables (screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ)

-- Define the total number of movies calculation
noncomputable def total_movies_shown (screens : ℕ) (open_hours : ℕ) (movie_duration : ℕ) : ℕ :=
  screens * (open_hours / movie_duration)

-- Theorem to prove the total number of movies shown is 24
theorem movies_shown_eq_twenty_four : 
  total_movies_shown 6 8 2 = 24 :=
by
  sorry

end movies_shown_eq_twenty_four_l1051_105120


namespace gain_percent_l1051_105135

variable (MP CP SP : ℝ)

def costPrice (CP : ℝ) (MP : ℝ) := CP = 0.64 * MP

def sellingPrice (SP : ℝ) (MP : ℝ) := SP = MP * 0.88

theorem gain_percent (h1 : costPrice CP MP) (h2 : sellingPrice SP MP) : 
  ((SP - CP) / CP) * 100 = 37.5 :=
by
  sorry

end gain_percent_l1051_105135


namespace example_function_indeterminate_unbounded_l1051_105151

theorem example_function_indeterminate_unbounded:
  (∀ x, ∃ f : ℝ → ℝ, (f x = (x^2 + x - 2) / (x^3 + 2 * x + 1)) ∧ 
                      (f 1 = (0 / (1^3 + 2 * 1 + 1))) ∧
                      (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 1) < δ → abs (f x) > ε)) :=
by
  sorry

end example_function_indeterminate_unbounded_l1051_105151


namespace bucket_holds_120_ounces_l1051_105175

theorem bucket_holds_120_ounces :
  ∀ (fill_buckets remove_buckets baths_per_day ounces_per_week : ℕ),
    fill_buckets = 14 →
    remove_buckets = 3 →
    baths_per_day = 7 →
    ounces_per_week = 9240 →
    baths_per_day * (fill_buckets - remove_buckets) * (ounces_per_week / (baths_per_day * (fill_buckets - remove_buckets))) = ounces_per_week →
    (ounces_per_week / (baths_per_day * (fill_buckets - remove_buckets))) = 120 :=
by
  intros fill_buckets remove_buckets baths_per_day ounces_per_week Hfill Hremove Hbaths Hounces Hcalc
  sorry

end bucket_holds_120_ounces_l1051_105175


namespace luna_badges_correct_l1051_105186

-- conditions
def total_badges : ℕ := 83
def hermione_badges : ℕ := 14
def celestia_badges : ℕ := 52

-- question and answer
theorem luna_badges_correct : total_badges - (hermione_badges + celestia_badges) = 17 :=
by
  sorry

end luna_badges_correct_l1051_105186


namespace evaporation_days_l1051_105189

theorem evaporation_days
    (initial_water : ℝ)
    (evap_rate : ℝ)
    (percent_evaporated : ℝ)
    (evaporated_water : ℝ)
    (days : ℝ)
    (h1 : initial_water = 10)
    (h2 : evap_rate = 0.012)
    (h3 : percent_evaporated = 0.06)
    (h4 : evaporated_water = initial_water * percent_evaporated)
    (h5 : days = evaporated_water / evap_rate) :
  days = 50 :=
by
  sorry

end evaporation_days_l1051_105189


namespace monotonicity_of_f_inequality_f_l1051_105108

section
variables {f : ℝ → ℝ}
variables (h_dom : ∀ x, x > 0 → f x > 0)
variables (h_f2 : f 2 = 1)
variables (h_fxy : ∀ x y, f (x * y) = f x + f y)
variables (h_pos : ∀ x, 1 < x → f x > 0)

-- Monotonicity of f(x)
theorem monotonicity_of_f :
  ∀ x1 x2, 0 < x1 → x1 < x2 → f x1 < f x2 :=
sorry

-- Inequality f(x) + f(x-2) ≤ 3 
theorem inequality_f (x : ℝ) :
  2 < x ∧ x ≤ 4 → f x + f (x - 2) ≤ 3 :=
sorry

end

end monotonicity_of_f_inequality_f_l1051_105108


namespace inverse_of_composite_l1051_105130

-- Define the function g
def g (x : ℕ) : ℕ :=
  if x = 1 then 4 else
  if x = 2 then 3 else
  if x = 3 then 1 else
  if x = 4 then 5 else
  if x = 5 then 2 else
  0  -- g is not defined for values other than 1 to 5

-- Define the inverse g_inv
def g_inv (x : ℕ) : ℕ :=
  if x = 4 then 1 else
  if x = 3 then 2 else
  if x = 1 then 3 else
  if x = 5 then 4 else
  if x = 2 then 5 else
  0  -- g_inv is not defined for values other than 1 to 5

theorem inverse_of_composite :
  g_inv (g_inv (g_inv 3)) = 4 :=
by
  sorry

end inverse_of_composite_l1051_105130


namespace total_trees_planted_l1051_105148

theorem total_trees_planted (apple_trees orange_trees : ℕ) (h₁ : apple_trees = 47) (h₂ : orange_trees = 27) : apple_trees + orange_trees = 74 := 
by
  -- We skip the proof step
  sorry

end total_trees_planted_l1051_105148


namespace find_p4_q4_l1051_105160

-- Definitions
def p (x : ℝ) : ℝ := 3 * (x - 6) * (x - 2)
def q (x : ℝ) : ℝ := (x - 6) * (x + 3)

-- Statement to prove
theorem find_p4_q4 : (p 4) / (q 4) = 6 / 7 :=
by
  sorry

end find_p4_q4_l1051_105160


namespace solve_eq_l1051_105115

theorem solve_eq (x y : ℕ) (h : x^2 - 2 * x * y + y^2 + 5 * x + 5 * y = 1500) :
  (x = 150 ∧ y = 150) ∨ (x = 150 ∧ y = 145) ∨ (x = 145 ∧ y = 135) ∨
  (x = 135 ∧ y = 120) ∨ (x = 120 ∧ y = 100) ∨ (x = 100 ∧ y = 75) ∨
  (x = 75 ∧ y = 45) ∨ (x = 45 ∧ y = 10) ∨ (x = 145 ∧ y = 150) ∨
  (x = 135 ∧ y = 145) ∨ (x = 120 ∧ y = 135) ∨ (x = 100 ∧ y = 120) ∨
  (x = 75 ∧ y = 100) ∨ (x = 45 ∧ y = 75) ∨ (x = 10 ∧ y = 45) :=
sorry

end solve_eq_l1051_105115
