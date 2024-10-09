import Mathlib

namespace sin_segment_ratio_is_rel_prime_l154_15414

noncomputable def sin_segment_ratio : ℕ × ℕ :=
  let p := 1
  let q := 8
  (p, q)
  
theorem sin_segment_ratio_is_rel_prime :
  1 < 8 ∧ gcd 1 8 = 1 ∧ sin_segment_ratio = (1, 8) :=
by
  -- gcd 1 8 = 1
  have h1 : gcd 1 8 = 1 := by exact gcd_one_right 8
  -- 1 < 8
  have h2 : 1 < 8 := by decide
  -- final tuple
  have h3 : sin_segment_ratio = (1, 8) := by rfl
  exact ⟨h2, h1, h3⟩

end sin_segment_ratio_is_rel_prime_l154_15414


namespace correctly_calculated_value_l154_15493

theorem correctly_calculated_value : 
  ∃ x : ℝ, (x + 4 = 40) ∧ (x / 4 = 9) :=
sorry

end correctly_calculated_value_l154_15493


namespace part_a_l154_15417

theorem part_a (n : ℕ) (h_condition : n < 135) : ∃ r, r = 239 % n ∧ r ≤ 119 := 
sorry

end part_a_l154_15417


namespace maximize_sales_volume_l154_15479

open Real

def profit (x : ℝ) : ℝ := (x - 20) * (400 - 20 * (x - 30))

theorem maximize_sales_volume : 
  ∃ x : ℝ, (∀ x' : ℝ, profit x' ≤ profit x) ∧ x = 35 := 
by
  sorry

end maximize_sales_volume_l154_15479


namespace sqrt_six_lt_a_lt_cubic_two_l154_15426

theorem sqrt_six_lt_a_lt_cubic_two (a : ℝ) (h : a^5 - a^3 + a = 2) : (Real.sqrt 3)^6 < a ∧ a < 2^(1/3) :=
sorry

end sqrt_six_lt_a_lt_cubic_two_l154_15426


namespace john_paid_correct_amount_l154_15438

theorem john_paid_correct_amount : 
  let upfront_fee := 1000
  let hourly_rate := 100
  let court_hours := 50
  let prep_hours := 2 * court_hours
  let total_hours_fee := (court_hours + prep_hours) * hourly_rate
  let paperwork_fee := 500
  let transportation_costs := 300
  let total_fee := total_hours_fee + upfront_fee + paperwork_fee + transportation_costs
  let john_share := total_fee / 2
  john_share = 8400 :=
by
  let upfront_fee := 1000
  let hourly_rate := 100
  let court_hours := 50
  let prep_hours := 2 * court_hours
  let total_hours_fee := (court_hours + prep_hours) * hourly_rate
  let paperwork_fee := 500
  let transportation_costs := 300
  let total_fee := total_hours_fee + upfront_fee + paperwork_fee + transportation_costs
  let john_share := total_fee / 2
  show john_share = 8400
  sorry

end john_paid_correct_amount_l154_15438


namespace fourth_number_ninth_row_eq_206_l154_15460

-- Define the first number in a given row
def first_number_in_row (i : Nat) : Nat :=
  2 + 4 * 6 * (i - 1)

-- Define the number in the j-th position in the i-th row
def number_in_row (i j : Nat) : Nat :=
  first_number_in_row i + 4 * (j - 1)

-- Define the 9th row and fourth number in it
def fourth_number_ninth_row : Nat :=
  number_in_row 9 4

-- The theorem to prove the fourth number in the 9th row is 206
theorem fourth_number_ninth_row_eq_206 : fourth_number_ninth_row = 206 := by
  sorry

end fourth_number_ninth_row_eq_206_l154_15460


namespace cost_prices_three_watches_l154_15412

theorem cost_prices_three_watches :
  ∃ (C1 C2 C3 : ℝ), 
    (0.9 * C1 + 210 = 1.04 * C1) ∧ 
    (0.85 * C2 + 180 = 1.03 * C2) ∧ 
    (0.95 * C3 + 250 = 1.06 * C3) ∧ 
    C1 = 1500 ∧ 
    C2 = 1000 ∧ 
    C3 = (25000 / 11) :=
by 
  sorry

end cost_prices_three_watches_l154_15412


namespace sum_leq_two_l154_15404

open Classical

theorem sum_leq_two (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^3 + b^3 = 2) : a + b ≤ 2 :=
by
  sorry

end sum_leq_two_l154_15404


namespace overlapping_rectangles_perimeter_l154_15423

namespace RectangleOverlappingPerimeter

def length := 7
def width := 3

/-- Prove that the perimeter of the shape formed by overlapping two rectangles,
    each measuring 7 cm by 3 cm, is 28 cm. -/
theorem overlapping_rectangles_perimeter : 
  let total_perimeter := 2 * (length + (2 * width))
  total_perimeter = 28 :=
by
  sorry

end RectangleOverlappingPerimeter

end overlapping_rectangles_perimeter_l154_15423


namespace lauren_total_earnings_l154_15463

-- Define earnings conditions
def mondayCommercialEarnings (views : ℕ) : ℝ := views * 0.40
def mondaySubscriptionEarnings (subs : ℕ) : ℝ := subs * 0.80

def tuesdayCommercialEarnings (views : ℕ) : ℝ := views * 0.50
def tuesdaySubscriptionEarnings (subs : ℕ) : ℝ := subs * 1.00

def weekendMerchandiseEarnings (sales : ℝ) : ℝ := 0.10 * sales

-- Specific conditions for each day
def mondayTotalEarnings : ℝ := mondayCommercialEarnings 80 + mondaySubscriptionEarnings 20
def tuesdayTotalEarnings : ℝ := tuesdayCommercialEarnings 100 + tuesdaySubscriptionEarnings 27
def weekendTotalEarnings : ℝ := weekendMerchandiseEarnings 150

-- Total earnings for the period
def totalEarnings : ℝ := mondayTotalEarnings + tuesdayTotalEarnings + weekendTotalEarnings

-- Examining the final value
theorem lauren_total_earnings : totalEarnings = 140.00 := by
  sorry

end lauren_total_earnings_l154_15463


namespace smallest_three_digit_multiple_of_17_l154_15451

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n ∧ ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → n ≤ m := by
  sorry

end smallest_three_digit_multiple_of_17_l154_15451


namespace snow_at_Brecknock_l154_15462

theorem snow_at_Brecknock (hilt_snow brecknock_snow : ℕ) (h1 : hilt_snow = 29) (h2 : hilt_snow = brecknock_snow + 12) : brecknock_snow = 17 :=
by
  sorry

end snow_at_Brecknock_l154_15462


namespace ship_length_correct_l154_15450

noncomputable def ship_length : ℝ :=
  let speed_kmh := 24
  let speed_mps := speed_kmh * 1000 / 3600
  let time := 202.48
  let bridge_length := 900
  let total_distance := speed_mps * time
  total_distance - bridge_length

theorem ship_length_correct : ship_length = 450.55 :=
by
  -- This is where the proof would be written, but we're skipping the proof as per instructions
  sorry

end ship_length_correct_l154_15450


namespace quadratic_equation_m_l154_15418

theorem quadratic_equation_m (m : ℝ) (h1 : |m| + 1 = 2) (h2 : m + 1 ≠ 0) : m = 1 :=
sorry

end quadratic_equation_m_l154_15418


namespace cost_of_bag_l154_15480

variable (cost_per_bag : ℝ)
variable (chips_per_bag : ℕ := 24)
variable (calories_per_chip : ℕ := 10)
variable (total_calories : ℕ := 480)
variable (total_cost : ℝ := 4)

theorem cost_of_bag :
  (chips_per_bag * (total_calories / calories_per_chip / chips_per_bag) = (total_calories / calories_per_chip)) →
  (total_cost / (total_calories / (calories_per_chip * chips_per_bag))) = 2 :=
by
  sorry

end cost_of_bag_l154_15480


namespace car_speed_without_red_light_l154_15469

theorem car_speed_without_red_light (v : ℝ) :
  (∃ k : ℕ+, v = 10 / k) ↔ 
  ∀ (dist : ℝ) (green_duration red_duration total_cycle : ℝ),
    dist = 1500 ∧ green_duration = 90 ∧ red_duration = 60 ∧ total_cycle = 150 →
    v * total_cycle = dist / (green_duration + red_duration) := 
by
  sorry

end car_speed_without_red_light_l154_15469


namespace car_division_ways_l154_15456

/-- 
Prove that the number of ways to divide 6 people 
into two different cars, with each car holding 
a maximum of 4 people, is equal to 50. 
-/
theorem car_division_ways : 
  (∃ s1 s2 : Finset ℕ, s1.card = 2 ∧ s2.card = 4) ∨ 
  (∃ s1 s2 : Finset ℕ, s1.card = 3 ∧ s2.card = 3) ∨ 
  (∃ s1 s2 : Finset ℕ, s1.card = 4 ∧ s2.card = 2) →
  (15 + 20 + 15 = 50) := 
by 
  sorry

end car_division_ways_l154_15456


namespace simplify_expression_and_evaluate_at_zero_l154_15498

theorem simplify_expression_and_evaluate_at_zero :
  ((2 * (0 : ℝ) - 1) / (0 + 1) - 0 + 1) / ((0 - 2) / ((0 ^ 2) + 2 * 0 + 1)) = 0 :=
by
  -- proof omitted
  sorry

end simplify_expression_and_evaluate_at_zero_l154_15498


namespace dodecahedron_interior_diagonals_count_l154_15457

-- Define a dodecahedron structure
structure Dodecahedron :=
  (vertices : ℕ)
  (edges_per_vertex : ℕ)
  (faces_per_vertex : ℕ)

-- Define the property of a dodecahedron
def dodecahedron_property : Dodecahedron :=
{
  vertices := 20,
  edges_per_vertex := 3,
  faces_per_vertex := 3
}

-- The theorem statement
theorem dodecahedron_interior_diagonals_count (d : Dodecahedron)
  (h1 : d.vertices = 20)
  (h2 : d.edges_per_vertex = 3)
  (h3 : d.faces_per_vertex = 3) : 
  (d.vertices * (d.vertices - d.edges_per_vertex)) / 2 = 160 :=
by
  sorry

end dodecahedron_interior_diagonals_count_l154_15457


namespace angus_tokens_l154_15454

theorem angus_tokens (x : ℕ) (h1 : x = 60 - (25 / 100) * 60) : x = 45 :=
by
  sorry

end angus_tokens_l154_15454


namespace partA_partB_partC_l154_15461
noncomputable section

def n : ℕ := 100
def p : ℝ := 0.8
def q : ℝ := 1 - p

def binomial_prob (k1 k2 : ℕ) : ℝ := sorry

theorem partA : binomial_prob 70 85 = 0.8882 := sorry
theorem partB : binomial_prob 70 100 = 0.9938 := sorry
theorem partC : binomial_prob 0 69 = 0.0062 := sorry

end partA_partB_partC_l154_15461


namespace second_discarded_number_l154_15434

theorem second_discarded_number (S : ℝ) (X : ℝ) :
  (S = 50 * 44) →
  ((S - 45 - X) / 48 = 43.75) →
  X = 55 :=
by
  intros h1 h2
  -- The proof steps would go here, but we leave it unproved
  sorry

end second_discarded_number_l154_15434


namespace inheritance_amount_l154_15452

theorem inheritance_amount (x : ℝ) 
  (h1 : x * 0.25 + (x - x * 0.25) * 0.12 = 13600) : x = 40000 :=
by
  -- This is where the proof would go
  sorry

end inheritance_amount_l154_15452


namespace f_minus_ten_l154_15464

noncomputable def f : ℝ → ℝ := sorry

theorem f_minus_ten :
  (∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y) →
  (f 1 = 2) →
  f (-10) = 90 :=
by
  intros h1 h2
  sorry

end f_minus_ten_l154_15464


namespace solve_quadratic_eqn_l154_15477

theorem solve_quadratic_eqn :
  ∀ x : ℝ, (x - 2) * (x + 3) = 0 ↔ (x = 2 ∨ x = -3) :=
by
  intros
  simp
  sorry

end solve_quadratic_eqn_l154_15477


namespace no_real_roots_of_quadratic_l154_15492

theorem no_real_roots_of_quadratic :
  ∀ (a b c : ℝ), a = 1 → b = -Real.sqrt 5 → c = Real.sqrt 2 →
  (b^2 - 4 * a * c < 0) → ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
by
  intros a b c ha hb hc hD
  rw [ha, hb, hc] at hD
  sorry

end no_real_roots_of_quadratic_l154_15492


namespace probability_of_specific_combination_l154_15473

theorem probability_of_specific_combination :
  let shirts := 6
  let shorts := 8
  let socks := 7
  let total_clothes := shirts + shorts + socks
  let ways_total := Nat.choose total_clothes 4
  let ways_shirts := Nat.choose shirts 2
  let ways_shorts := Nat.choose shorts 1
  let ways_socks := Nat.choose socks 1
  let ways_favorable := ways_shirts * ways_shorts * ways_socks
  let probability := (ways_favorable: ℚ) / ways_total
  probability = 56 / 399 :=
by
  simp
  sorry

end probability_of_specific_combination_l154_15473


namespace symmetric_graph_inverse_l154_15458

def f (x : ℝ) : ℝ := sorry -- We assume f is defined accordingly somewhere, as the inverse of ln.

theorem symmetric_graph_inverse (h : ∀ x, f (f x) = x) : f 2 = Real.exp 2 := by
  sorry

end symmetric_graph_inverse_l154_15458


namespace sum_mod_17_l154_15424

theorem sum_mod_17 :
  (78 + 79 + 80 + 81 + 82 + 83 + 84 + 85) % 17 = 6 :=
by
  sorry

end sum_mod_17_l154_15424


namespace least_tiles_required_l154_15497

def room_length : ℕ := 7550
def room_breadth : ℕ := 2085
def tile_size : ℕ := 5
def total_area : ℕ := room_length * room_breadth
def tile_area : ℕ := tile_size * tile_size
def number_of_tiles : ℕ := total_area / tile_area

theorem least_tiles_required : number_of_tiles = 630270 := by
  sorry

end least_tiles_required_l154_15497


namespace perfect_square_expression_l154_15449

theorem perfect_square_expression (x y z : ℤ) :
    9 * (x^2 + y^2 + z^2)^2 - 8 * (x + y + z) * (x^3 + y^3 + z^3 - 3 * x * y * z) =
      ((x + y + z)^2 - 6 * (x * y + y * z + z * x))^2 := 
by 
  sorry

end perfect_square_expression_l154_15449


namespace fourth_vertex_of_square_l154_15415

def A : ℂ := 2 - 3 * Complex.I
def B : ℂ := 3 + 2 * Complex.I
def C : ℂ := -3 + 2 * Complex.I

theorem fourth_vertex_of_square : ∃ D : ℂ, 
  (D - B) = (B - A) * Complex.I ∧ 
  (D - C) = (C - A) * Complex.I ∧ 
  (D = -3 + 8 * Complex.I) :=
sorry

end fourth_vertex_of_square_l154_15415


namespace sum_first_12_terms_l154_15453

variable (S : ℕ → ℝ)

def sum_of_first_n_terms (n : ℕ) : ℝ :=
  S n

theorem sum_first_12_terms (h₁ : sum_of_first_n_terms 4 = 30) (h₂ : sum_of_first_n_terms 8 = 100) :
  sum_of_first_n_terms 12 = 210 := 
sorry

end sum_first_12_terms_l154_15453


namespace gcd_lcm_45_150_l154_15405

theorem gcd_lcm_45_150 : Nat.gcd 45 150 = 15 ∧ Nat.lcm 45 150 = 450 :=
by
  sorry

end gcd_lcm_45_150_l154_15405


namespace sum_zero_implies_inequality_l154_15470

variable {a b c d : ℝ}

theorem sum_zero_implies_inequality
  (h : a + b + c + d = 0) :
  5 * (a * b + b * c + c * d) + 8 * (a * c + a * d + b * d) ≤ 0 := 
sorry

end sum_zero_implies_inequality_l154_15470


namespace problem_statement_l154_15420

noncomputable def f (x : ℝ) : ℝ := 2 / (2019^x + 1) + Real.sin x

noncomputable def f' (x : ℝ) := (deriv f) x

theorem problem_statement :
  f 2018 + f (-2018) + f' 2019 - f' (-2019) = 2 :=
by {
  sorry
}

end problem_statement_l154_15420


namespace least_clock_equivalent_to_square_greater_than_4_l154_15474

theorem least_clock_equivalent_to_square_greater_than_4 : 
  ∃ (x : ℕ), x > 4 ∧ (x^2 - x) % 12 = 0 ∧ ∀ (y : ℕ), y > 4 → (y^2 - y) % 12 = 0 → x ≤ y :=
by
  -- The proof will go here
  sorry

end least_clock_equivalent_to_square_greater_than_4_l154_15474


namespace triangle_area_of_tangent_line_l154_15476

theorem triangle_area_of_tangent_line (a : ℝ) 
  (h : a > 0) 
  (ha : (1/2) * 3 * a * (3 / (2 * a ^ (1/2))) = 18)
  : a = 64 := 
sorry

end triangle_area_of_tangent_line_l154_15476


namespace decreasing_interval_of_even_function_l154_15468

-- Defining the function f(x)
def f (x : ℝ) (k : ℝ) : ℝ := (k-2) * x^2 + (k-1) * x + 3

-- Defining the condition that f is an even function
def isEvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

theorem decreasing_interval_of_even_function (k : ℝ) :
  isEvenFunction (f · k) → k = 1 ∧ ∀ x ≥ 0, f x k ≤ f 0 k :=
by
  sorry

end decreasing_interval_of_even_function_l154_15468


namespace disjoint_sets_l154_15446

def P : ℕ → ℝ → ℝ
| 0, x => x
| 1, x => 4 * x^3 + 3 * x
| (n + 1), x => (4 * x^2 + 2) * P n x - P (n - 1) x

def A (m : ℝ) : Set ℝ := {x | ∃ n : ℕ, P n m = x }

theorem disjoint_sets (m : ℝ) : Disjoint (A m) (A (m + 4)) :=
by
  -- Proof goes here
  sorry

end disjoint_sets_l154_15446


namespace two_point_questions_l154_15444

theorem two_point_questions (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : x = 30 :=
by
  sorry

end two_point_questions_l154_15444


namespace total_cost_two_rackets_l154_15422

theorem total_cost_two_rackets (full_price : ℕ) (discount : ℕ) (total_cost : ℕ) :
  (full_price = 60) →
  (discount = full_price / 2) →
  (total_cost = full_price + (full_price - discount)) →
  total_cost = 90 :=
by
  intros h_full_price h_discount h_total_cost
  rw [h_full_price, h_discount] at h_total_cost
  sorry

end total_cost_two_rackets_l154_15422


namespace conversion_base8_to_base10_l154_15499

theorem conversion_base8_to_base10 : 5 * 8^3 + 2 * 8^2 + 1 * 8^1 + 4 * 8^0 = 2700 :=
by 
  sorry

end conversion_base8_to_base10_l154_15499


namespace gumballs_per_package_correct_l154_15443

-- Define the conditions
def total_gumballs_eaten : ℕ := 20
def number_of_boxes_finished : ℕ := 4

-- Define the target number of gumballs in each package
def gumballs_in_each_package := 5

theorem gumballs_per_package_correct :
  total_gumballs_eaten / number_of_boxes_finished = gumballs_in_each_package :=
by
  sorry

end gumballs_per_package_correct_l154_15443


namespace hyperbola_eccentricity_l154_15439

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (c : ℝ) (h3 : a^2 + b^2 = c^2) 
  (h4 : ∃ M : ℝ × ℝ, (M.fst^2 / a^2 - M.snd^2 / b^2 = 1) ∧ (M.snd^2 = 8 * M.fst)
    ∧ (|M.fst - 2| + |M.snd| = 5)) : 
  (c / a = 2) :=
by
  sorry

end hyperbola_eccentricity_l154_15439


namespace sample_variance_l154_15416

theorem sample_variance (a : ℝ) (h_avg : (a + 0 + 1 + 2 + 3) / 5 = 1) :
  (1 / 5) * ((a - 1)^2 + (0 - 1)^2 + (1 - 1)^2 + (2 - 1)^2 + (3 - 1)^2) = 2 :=
by
  sorry

end sample_variance_l154_15416


namespace equation_solution_l154_15401

theorem equation_solution (x : ℝ) (h : x ≠ 1 ∧ x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5) :
  (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
   1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 4) ↔ 
  (x = 3 + 2 * Real.sqrt 5 ∨ x = 3 - 2 * Real.sqrt 5) := 
sorry

end equation_solution_l154_15401


namespace relationship_between_y1_y2_y3_l154_15403

-- Define the parabola equation and points
def parabola (x c : ℝ) : ℝ := 2 * (x - 1)^2 + c

-- Define the points
def point1 := -2
def point2 := 0
def point3 := 5 / 3

-- Define the y values at these points
def y1 (c : ℝ) := parabola point1 c
def y2 (c : ℝ) := parabola point2 c
def y3 (c : ℝ) := parabola point3 c

-- Proof statement
theorem relationship_between_y1_y2_y3 (c : ℝ) : 
  y1 c > y2 c ∧ y2 c > y3 c :=
sorry

end relationship_between_y1_y2_y3_l154_15403


namespace sector_area_l154_15471

theorem sector_area (r α S : ℝ) (h1 : α = 2) (h2 : 2 * r + α * r = 8) : S = 4 :=
sorry

end sector_area_l154_15471


namespace trains_at_start_2016_l154_15402

def traversal_time_red := 7
def traversal_time_blue := 8
def traversal_time_green := 9

def return_period_red := 2 * traversal_time_red
def return_period_blue := 2 * traversal_time_blue
def return_period_green := 2 * traversal_time_green

def train_start_pos_time := 2016
noncomputable def lcm_period := Nat.lcm return_period_red (Nat.lcm return_period_blue return_period_green)

theorem trains_at_start_2016 :
  train_start_pos_time % lcm_period = 0 :=
by
  have return_period_red := 2 * traversal_time_red
  have return_period_blue := 2 * traversal_time_blue
  have return_period_green := 2 * traversal_time_green
  have lcm_period := Nat.lcm return_period_red (Nat.lcm return_period_blue return_period_green)
  have train_start_pos_time := 2016
  exact sorry

end trains_at_start_2016_l154_15402


namespace wise_men_correct_guesses_l154_15421

noncomputable def max_correct_guesses (n k : ℕ) : ℕ :=
  if n > k + 1 then n - k - 1 else 0

theorem wise_men_correct_guesses (n k : ℕ) :
  ∃ (m : ℕ), m = max_correct_guesses n k ∧ m ≤ n - k - 1 :=
by {
  sorry
}

end wise_men_correct_guesses_l154_15421


namespace vector_expression_l154_15494

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (i j k a b : V)
variables (h_i_j_k_non_coplanar : ∃ (l m n : ℝ), l • i + m • j + n • k = 0 → l = 0 ∧ m = 0 ∧ n = 0)
variables (h_a : a = (1 / 2 : ℝ) • i - j + k)
variables (h_b : b = 5 • i - 2 • j - k)

theorem vector_expression :
  4 • a - 3 • b = -13 • i + 2 • j + 7 • k :=
by
  sorry

end vector_expression_l154_15494


namespace molecular_weight_N2O5_l154_15408

variable {x : ℕ}

theorem molecular_weight_N2O5 (hx : 10 * 108 = 1080) : (108 * x = 1080 * x / 10) :=
by
  sorry

end molecular_weight_N2O5_l154_15408


namespace max_value_of_PQ_l154_15459

noncomputable def f (x : ℝ) := Real.sin (2 * x - Real.pi / 12)
noncomputable def g (x : ℝ) := Real.sqrt 3 * Real.cos (2 * x - Real.pi / 12)

theorem max_value_of_PQ (t : ℝ) : abs (f t - g t) ≤ 2 :=
by sorry

end max_value_of_PQ_l154_15459


namespace percentage_blue_and_red_l154_15448

theorem percentage_blue_and_red (F : ℕ) (h_even: F % 2 = 0)
  (h1: ∃ C, 50 / 100 * C = F / 2)
  (h2: ∃ C, 60 / 100 * C = F / 2)
  (h3: ∃ C, 40 / 100 * C = F / 2) :
  ∃ C, (50 / 100 * C + 60 / 100 * C - 100 / 100 * C) = 10 / 100 * C :=
sorry

end percentage_blue_and_red_l154_15448


namespace find_p_l154_15433

theorem find_p (x y : ℝ) (h : y = 1.15 * x * (1 - p / 100)) : p = 15 :=
sorry

end find_p_l154_15433


namespace digits_subtraction_eq_zero_l154_15475

theorem digits_subtraction_eq_zero (d A B : ℕ) (h1 : d > 8)
  (h2 : A < d) (h3 : B < d)
  (h4 : A * d + B + A * d + A = 2 * d + 3 * d + 4) :
  A - B = 0 :=
by sorry

end digits_subtraction_eq_zero_l154_15475


namespace tyrone_gave_marbles_l154_15488

theorem tyrone_gave_marbles :
  ∃ x : ℝ, (120 - x = 3 * (30 + x)) ∧ x = 7.5 :=
by
  sorry

end tyrone_gave_marbles_l154_15488


namespace relationship_x2_ax_bx_l154_15437

variable {x a b : ℝ}

theorem relationship_x2_ax_bx (h1 : x < a) (h2 : a < 0) (h3 : b > 0) : x^2 > ax ∧ ax > bx :=
by
  sorry

end relationship_x2_ax_bx_l154_15437


namespace total_fruits_proof_l154_15410

-- Definitions of the quantities involved in the problem.
def apples_basket1_to_3 := 9
def oranges_basket1_to_3 := 15
def bananas_basket1_to_3 := 14
def apples_basket4 := apples_basket1_to_3 - 2
def oranges_basket4 := oranges_basket1_to_3 - 2
def bananas_basket4 := bananas_basket1_to_3 - 2

-- Total fruits in first three baskets
def total_fruits_baskets1_to_3 := 3 * (apples_basket1_to_3 + oranges_basket1_to_3 + bananas_basket1_to_3)

-- Total fruits in fourth basket
def total_fruits_basket4 := apples_basket4 + oranges_basket4 + bananas_basket4

-- Total fruits in all four baskets
def total_fruits_all_baskets := total_fruits_baskets1_to_3 + total_fruits_basket4

-- Theorem statement
theorem total_fruits_proof : total_fruits_all_baskets = 146 :=
by
  -- Placeholder for proof
  sorry

end total_fruits_proof_l154_15410


namespace jared_annual_earnings_l154_15484

open Nat

noncomputable def diploma_monthly_pay : ℕ := 4000
noncomputable def months_in_year : ℕ := 12
noncomputable def multiplier : ℕ := 3

theorem jared_annual_earnings :
  let jared_monthly_earnings := diploma_monthly_pay * multiplier
  let jared_annual_earnings := jared_monthly_earnings * months_in_year
  jared_annual_earnings = 144000 :=
by
  let jared_monthly_earnings := diploma_monthly_pay * multiplier
  let jared_annual_earnings := jared_monthly_earnings * months_in_year
  exact sorry

end jared_annual_earnings_l154_15484


namespace first_recipe_cups_l154_15486

-- Definitions based on the given conditions
def ounces_per_bottle : ℕ := 16
def ounces_per_cup : ℕ := 8
def cups_second_recipe : ℕ := 1
def cups_third_recipe : ℕ := 3
def total_bottles : ℕ := 3
def total_ounces : ℕ := total_bottles * ounces_per_bottle
def total_cups_needed : ℕ := total_ounces / ounces_per_cup

-- Proving the amount of cups of soy sauce needed for the first recipe
theorem first_recipe_cups : 
  total_cups_needed - (cups_second_recipe + cups_third_recipe) = 2 
:= by 
-- Proof omitted
  sorry

end first_recipe_cups_l154_15486


namespace equilibrium_temperature_l154_15430

theorem equilibrium_temperature 
  (c_B : ℝ) (c_m : ℝ)
  (m_B : ℝ) (m_m : ℝ)
  (T₁ : ℝ) (T_eq₁ : ℝ) (T_metal : ℝ) 
  (T_eq₂ : ℝ)
  (h₁ : T₁ = 80)
  (h₂ : T_eq₁ = 60)
  (h₃ : T_metal = 20)
  (h₄ : T₂ = 50)
  (h_ratio : c_B * m_B = 2 * c_m * m_m) :
  T_eq₂ = 50 :=
by
  sorry

end equilibrium_temperature_l154_15430


namespace cost_of_one_package_of_berries_l154_15472

noncomputable def martin_daily_consumption : ℚ := 1 / 2

noncomputable def package_content : ℚ := 1

noncomputable def total_period_days : ℚ := 30

noncomputable def total_spent : ℚ := 30

theorem cost_of_one_package_of_berries :
  (total_spent / (total_period_days * martin_daily_consumption / package_content)) = 2 :=
sorry

end cost_of_one_package_of_berries_l154_15472


namespace pagoda_top_story_lanterns_l154_15485

/--
Given a 7-story pagoda where each story has twice as many lanterns as the one above it, 
and a total of 381 lanterns across all stories, prove the number of lanterns on the top (7th) story is 3.
-/
theorem pagoda_top_story_lanterns (a : ℕ) (n : ℕ) (r : ℚ) (sum_lanterns : ℕ) :
  n = 7 → r = 1 / 2 → sum_lanterns = 381 →
  (a * (1 - r^n) / (1 - r) = sum_lanterns) → (a * r^(n - 1) = 3) :=
by
  intros h_n h_r h_sum h_geo_sum
  let a_val := 192 -- from the solution steps
  rw [h_n, h_r, h_sum] at h_geo_sum
  have h_a : a = a_val := by sorry
  rw [h_a, h_n, h_r]
  exact sorry

end pagoda_top_story_lanterns_l154_15485


namespace condition_on_p_l154_15466

theorem condition_on_p (p q r M : ℝ) (hq : 0 < q ∧ q < 100) (hr : 0 < r ∧ r < 100) (hM : 0 < M) :
  p > (100 * (q + r)) / (100 - q - r) → 
  M * (1 + p / 100) * (1 - q / 100) * (1 - r / 100) > M :=
by
  intro h
  -- The proof will go here
  sorry

end condition_on_p_l154_15466


namespace opposite_of_num_l154_15409

-- Define the number whose opposite we are calculating
def num := -1 / 2

-- Theorem statement that the opposite of num is 1/2
theorem opposite_of_num : -num = 1 / 2 := by
  -- The proof would go here
  sorry

end opposite_of_num_l154_15409


namespace c_work_rate_l154_15483

/--
A can do a piece of work in 4 days.
B can do it in 8 days.
With the assistance of C, A and B completed the work in 2 days.
Prove that C alone can do the work in 8 days.
-/
theorem c_work_rate :
  (1 / 4 + 1 / 8 + 1 / c = 1 / 2) → c = 8 :=
by
  intro h
  sorry

end c_work_rate_l154_15483


namespace range_of_m_l154_15432

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (m+1)*x^2 + (m+1)*x + (m+2) ≥ 0) ↔ m ≥ -1 := by
  sorry

end range_of_m_l154_15432


namespace g_of_neg_2_l154_15487

def f (x : ℚ) : ℚ := 4 * x - 9

def g (y : ℚ) : ℚ :=
  3 * ((y + 9) / 4)^2 - 4 * ((y + 9) / 4) + 2

theorem g_of_neg_2 : g (-2) = 67 / 16 :=
by
  sorry

end g_of_neg_2_l154_15487


namespace number_of_females_l154_15407

-- Definitions
variable (F : ℕ) -- ℕ = Natural numbers, ensuring F is a non-negative integer
variable (h_male : ℕ := 2 * F)
variable (h_total : F + 2 * F = 18)
variable (h_female_pos : F > 0)

-- Theorem
theorem number_of_females (F : ℕ) (h_male : ℕ := 2 * F) (h_total : F + 2 * F = 18) (h_female_pos : F > 0) : F = 6 := 
by 
  sorry

end number_of_females_l154_15407


namespace problem_statement_l154_15455

open Complex

theorem problem_statement :
  (3 - I) / (2 + I) = 1 - I :=
by
  sorry

end problem_statement_l154_15455


namespace supplemental_tank_time_l154_15419

-- Define the given conditions as assumptions
def primary_tank_time : Nat := 2
def total_time_needed : Nat := 8
def supplemental_tanks : Nat := 6
def additional_time_needed : Nat := total_time_needed - primary_tank_time

-- Define the theorem to prove
theorem supplemental_tank_time :
  additional_time_needed / supplemental_tanks = 1 :=
by
  -- Here we would provide the proof, but it is omitted with "sorry"
  sorry

end supplemental_tank_time_l154_15419


namespace must_true_l154_15491

axiom p : Prop
axiom q : Prop
axiom h1 : ¬ (p ∧ q)
axiom h2 : p ∨ q

theorem must_true : (¬ p) ∨ (¬ q) := by
  sorry

end must_true_l154_15491


namespace bob_friends_l154_15465

-- Define the total price and the amount paid by each person
def total_price : ℕ := 40
def amount_per_person : ℕ := 8

-- Define the total number of people who paid
def total_people : ℕ := total_price / amount_per_person

-- Define Bob's presence and require proving the number of his friends
theorem bob_friends (total_price amount_per_person total_people : ℕ) (h1 : total_price = 40)
  (h2 : amount_per_person = 8) (h3 : total_people = total_price / amount_per_person) : 
  total_people - 1 = 4 :=
by
  sorry

end bob_friends_l154_15465


namespace incorrect_statement_among_ABCD_l154_15411

theorem incorrect_statement_among_ABCD :
  ¬ (-3 = Real.sqrt ((-3)^2)) :=
by
  sorry

end incorrect_statement_among_ABCD_l154_15411


namespace time_to_cover_length_l154_15447

-- Definitions from conditions
def escalator_speed : Real := 15 -- ft/sec
def escalator_length : Real := 180 -- feet
def person_speed : Real := 3 -- ft/sec

-- Combined speed definition
def combined_speed : Real := escalator_speed + person_speed

-- Lean theorem statement proving the time taken
theorem time_to_cover_length : escalator_length / combined_speed = 10 := by
  sorry

end time_to_cover_length_l154_15447


namespace pears_weight_l154_15489

theorem pears_weight (x : ℕ) (h : 2 * x + 50 = 250) : x = 100 :=
sorry

end pears_weight_l154_15489


namespace johns_brother_age_l154_15413

variable (B : ℕ)
variable (J : ℕ)

-- Conditions given in the problem
def condition1 : Prop := J = 6 * B - 4
def condition2 : Prop := J + B = 10

-- The statement we want to prove, which is the answer to the problem:
theorem johns_brother_age (h1 : condition1 B J) (h2 : condition2 B J) : B = 2 := 
by 
  sorry

end johns_brother_age_l154_15413


namespace min_value_of_f_value_of_a_l154_15490

-- Definition of the function f
def f (x : ℝ) : ℝ := abs (x + 2) + 2 * abs (x - 1)

-- Problem: Prove that the minimum value of f(x) is 3
theorem min_value_of_f : ∃ x : ℝ, f x = 3 := sorry

-- Additional definitions for the second part of the problem
def g (x a : ℝ) : ℝ := f x + x - a

-- Problem: Given that the solution set of g(x,a) < 0 is (m, n) and n - m = 6, prove that a = 8
theorem value_of_a (a : ℝ) (m n : ℝ) (h : ∀ x : ℝ, g x a < 0 ↔ m < x ∧ x < n) (h_interval : n - m = 6) : a = 8 := sorry

end min_value_of_f_value_of_a_l154_15490


namespace speed_of_train_is_correct_l154_15440

-- Given conditions
def length_of_train : ℝ := 250
def length_of_bridge : ℝ := 120
def time_to_cross_bridge : ℝ := 20

-- Derived definition
def total_distance : ℝ := length_of_train + length_of_bridge

-- Goal to be proved
theorem speed_of_train_is_correct : total_distance / time_to_cross_bridge = 18.5 := 
by
  sorry

end speed_of_train_is_correct_l154_15440


namespace abs_expr_evaluation_l154_15425

theorem abs_expr_evaluation : abs (abs (-abs (-1 + 2) - 2) + 3) = 6 := by
  sorry

end abs_expr_evaluation_l154_15425


namespace remainder_is_83_l154_15435

-- Define the condition: the values for the division
def value1 : ℤ := 2021
def value2 : ℤ := 102

-- State the theorem: remainder when 2021 is divided by 102 is 83
theorem remainder_is_83 : value1 % value2 = 83 := by
  sorry

end remainder_is_83_l154_15435


namespace kids_played_on_tuesday_l154_15436

-- Definitions of the conditions
def kids_played_on_wednesday (julia : Type) : Nat := 4
def kids_played_on_monday (julia : Type) : Nat := 6
def difference_monday_wednesday (julia : Type) : Nat := 2

-- Define the statement to prove
theorem kids_played_on_tuesday (julia : Type) :
  (kids_played_on_monday julia - difference_monday_wednesday julia) = kids_played_on_wednesday julia :=
by
  sorry

end kids_played_on_tuesday_l154_15436


namespace construction_company_doors_needed_l154_15481

-- Definitions based on conditions
def num_floors_per_building : ℕ := 20
def num_apartments_per_floor : ℕ := 8
def num_buildings : ℕ := 3
def num_doors_per_apartment : ℕ := 10

-- Total number of apartments
def total_apartments : ℕ :=
  num_floors_per_building * num_apartments_per_floor * num_buildings

-- Total number of doors
def total_doors_needed : ℕ :=
  num_doors_per_apartment * total_apartments

-- Theorem statement to prove the number of doors needed
theorem construction_company_doors_needed :
  total_doors_needed = 4800 :=
sorry

end construction_company_doors_needed_l154_15481


namespace arithmetic_seq_sum_l154_15427

variable {a_n : ℕ → ℕ}
variable (S_n : ℕ → ℕ)
variable (q : ℕ)
variable (a_1 : ℕ)

axiom h1 : a_n 2 = 2
axiom h2 : a_n 6 = 32
axiom h3 : ∀ n, S_n n = a_1 * (1 - q ^ n) / (1 - q)

theorem arithmetic_seq_sum : S_n 100 = 2^100 - 1 :=
by
  sorry

end arithmetic_seq_sum_l154_15427


namespace sum_of_powers_divisible_by_6_l154_15428

theorem sum_of_powers_divisible_by_6 (a1 a2 a3 a4 : ℤ)
  (h : a1^3 + a2^3 + a3^3 + a4^3 = 0) (k : ℕ) (hk : k % 2 = 1) :
  6 ∣ (a1^k + a2^k + a3^k + a4^k) :=
sorry

end sum_of_powers_divisible_by_6_l154_15428


namespace one_elephant_lake_empty_in_365_days_l154_15441

variables (C K V : ℝ)
variables (t : ℝ)

noncomputable def lake_empty_one_day (C K V : ℝ) := 183 * C = V + K
noncomputable def lake_empty_five_days (C K V : ℝ) := 185 * C = V + 5 * K

noncomputable def elephant_time (C K V t : ℝ) : Prop :=
  (t * C = V + t * K) → (t = 365)

theorem one_elephant_lake_empty_in_365_days (C K V t : ℝ) :
  (lake_empty_one_day C K V) →
  (lake_empty_five_days C K V) →
  (elephant_time C K V t) := by
  intros h1 h2 h3
  sorry

end one_elephant_lake_empty_in_365_days_l154_15441


namespace hungarian_1905_l154_15406

open Nat

theorem hungarian_1905 (n p : ℕ) : (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + p * y = n ∧ x + y = p^z) ↔ 
  (p > 1 ∧ (n - 1) % (p - 1) = 0 ∧ ¬ ∃ k : ℕ, n = p^k) :=
by
  sorry

end hungarian_1905_l154_15406


namespace employees_bonus_l154_15482

theorem employees_bonus (x y z : ℝ) 
  (h1 : x + y + z = 2970) 
  (h2 : y = (1 / 3) * x + 180) 
  (h3 : z = (1 / 3) * y + 130) :
  x = 1800 ∧ y = 780 ∧ z = 390 :=
by
  sorry

end employees_bonus_l154_15482


namespace determine_squirrel_color_l154_15400

-- Define the types for Squirrel species and the nuts in hollows
inductive Squirrel
| red
| gray

def tells_truth (s : Squirrel) : Prop :=
  s = Squirrel.red

def lies (s : Squirrel) : Prop :=
  s = Squirrel.gray

-- Statements made by the squirrel in front of the second hollow
def statement1 (s : Squirrel) (no_nuts_in_first : Prop) : Prop :=
  tells_truth s → no_nuts_in_first ∧ (lies s → ¬no_nuts_in_first)

def statement2 (s : Squirrel) (nuts_in_either : Prop) : Prop :=
  tells_truth s → nuts_in_either ∧ (lies s → ¬nuts_in_either)

-- Given a squirrel that says the statements and the information about truth and lies
theorem determine_squirrel_color (s : Squirrel) (no_nuts_in_first : Prop) (nuts_in_either : Prop) :
  (statement1 s no_nuts_in_first) ∧ (statement2 s nuts_in_either) → s = Squirrel.red :=
by
  sorry

end determine_squirrel_color_l154_15400


namespace min_n_for_constant_term_l154_15478

theorem min_n_for_constant_term (n : ℕ) (h : 0 < n) : 
  (∃ (r : ℕ), 0 = n - 4 * r / 3) → n = 4 :=
by
  sorry

end min_n_for_constant_term_l154_15478


namespace simplify_expression_l154_15429

theorem simplify_expression : 
  (81 ^ (1 / Real.logb 5 9) + 3 ^ (3 / Real.logb (Real.sqrt 6) 3)) / 409 * 
  ((Real.sqrt 7) ^ (2 / Real.logb 25 7) - 125 ^ (Real.logb 25 6)) = 1 :=
by 
  sorry

end simplify_expression_l154_15429


namespace club_president_vice_president_combinations_144_l154_15431

variables (boys_total girls_total : Nat)
variables (senior_boys junior_boys senior_girls junior_girls : Nat)
variables (choose_president_vice_president : Nat)

-- Define the conditions
def club_conditions : Prop :=
  boys_total = 12 ∧
  girls_total = 12 ∧
  senior_boys = 6 ∧
  junior_boys = 6 ∧
  senior_girls = 6 ∧
  junior_girls = 6

-- Define the proof problem
def president_vice_president_combinations (boys_total girls_total senior_boys junior_boys senior_girls junior_girls : Nat) : Nat :=
  2 * senior_boys * junior_boys + 2 * senior_girls * junior_girls

-- The main theorem to prove
theorem club_president_vice_president_combinations_144 :
  club_conditions boys_total girls_total senior_boys junior_boys senior_girls junior_girls →
  president_vice_president_combinations boys_total girls_total senior_boys junior_boys senior_girls junior_girls = 144 :=
sorry

end club_president_vice_president_combinations_144_l154_15431


namespace sequence_property_l154_15495

theorem sequence_property : 
  ∀ (a : ℕ → ℝ), 
    a 1 = 1 →
    a 2 = 1 → 
    (∀ n, a (n + 2) = a (n + 1) + 1 / a n) →
    a 180 > 19 :=
by
  intros a h1 h2 h3
  sorry

end sequence_property_l154_15495


namespace another_seat_in_sample_l154_15496

-- Definition of the problem
def total_students := 56
def sample_size := 4
def sample_set : Finset ℕ := {3, 17, 45}

-- Lean 4 statement for the proof problem
theorem another_seat_in_sample :
  (sample_set = sample_set ∪ {31}) ∧
  (31 ∉ sample_set) ∧
  (∀ x ∈ sample_set ∪ {31}, x ≤ total_students) :=
by
  sorry

end another_seat_in_sample_l154_15496


namespace bottles_left_l154_15442

-- Define initial conditions
def bottlesInRefrigerator : Nat := 4
def bottlesInPantry : Nat := 4
def bottlesBought : Nat := 5
def bottlesDrank : Nat := 3

-- Goal: Prove the total number of bottles left
theorem bottles_left : bottlesInRefrigerator + bottlesInPantry + bottlesBought - bottlesDrank = 10 :=
by
  sorry

end bottles_left_l154_15442


namespace mike_spend_on_plants_l154_15467

def Mike_buys : Prop :=
  let rose_bushes_total := 6
  let rose_bush_cost := 75
  let friend_rose_bushes := 2
  let self_rose_bushes := rose_bushes_total - friend_rose_bushes
  let self_rose_bush_cost := self_rose_bushes * rose_bush_cost
  let tiger_tooth_aloe_total := 2
  let aloe_cost := 100
  let self_aloe_cost := tiger_tooth_aloe_total * aloe_cost
  self_rose_bush_cost + self_aloe_cost = 500

theorem mike_spend_on_plants :
  Mike_buys := by
  sorry

end mike_spend_on_plants_l154_15467


namespace total_onions_grown_l154_15445

theorem total_onions_grown :
  let onions_per_day_nancy := 3
  let days_worked_nancy := 4
  let onions_per_day_dan := 4
  let days_worked_dan := 6
  let onions_per_day_mike := 5
  let days_worked_mike := 5
  let onions_per_day_sasha := 6
  let days_worked_sasha := 4
  let onions_per_day_becky := 2
  let days_worked_becky := 6

  let total_onions_nancy := onions_per_day_nancy * days_worked_nancy
  let total_onions_dan := onions_per_day_dan * days_worked_dan
  let total_onions_mike := onions_per_day_mike * days_worked_mike
  let total_onions_sasha := onions_per_day_sasha * days_worked_sasha
  let total_onions_becky := onions_per_day_becky * days_worked_becky

  let total_onions := total_onions_nancy + total_onions_dan + total_onions_mike + total_onions_sasha + total_onions_becky

  total_onions = 97 :=
by
  -- proof goes here
  sorry

end total_onions_grown_l154_15445
