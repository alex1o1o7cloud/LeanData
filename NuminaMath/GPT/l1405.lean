import Mathlib

namespace find_fourth_vertex_l1405_140531

-- Given three vertices of a tetrahedron
def v1 : ℤ × ℤ × ℤ := (1, 1, 2)
def v2 : ℤ × ℤ × ℤ := (4, 2, 1)
def v3 : ℤ × ℤ × ℤ := (3, 1, 5)

-- The side length squared of the tetrahedron (computed from any pair of given points)
def side_length_squared : ℤ := 11

-- The goal is to find the fourth vertex with integer coordinates which maintains the distance
def is_fourth_vertex (x y z : ℤ) : Prop :=
  (x - 1)^2 + (y - 1)^2 + (z - 2)^2 = side_length_squared ∧
  (x - 4)^2 + (y - 2)^2 + (z - 1)^2 = side_length_squared ∧
  (x - 3)^2 + (y - 1)^2 + (z - 5)^2 = side_length_squared

theorem find_fourth_vertex : is_fourth_vertex 4 1 3 :=
  sorry

end find_fourth_vertex_l1405_140531


namespace max_min_values_l1405_140592

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_values :
  (∀ x ∈ (Set.Icc 0 2), f x ≤ 5) ∧ (∃ x ∈ (Set.Icc 0 2), f x = 5) ∧
  (∀ x ∈ (Set.Icc 0 2), f x ≥ -15) ∧ (∃ x ∈ (Set.Icc 0 2), f x = -15) :=
by
  sorry

end max_min_values_l1405_140592


namespace sum_of_digits_of_power_eight_2010_l1405_140572

theorem sum_of_digits_of_power_eight_2010 :
  let n := 2010
  let a := 8
  let tens_digit := (a ^ n / 10) % 10
  let units_digit := a ^ n % 10
  tens_digit + units_digit = 1 :=
by
  sorry

end sum_of_digits_of_power_eight_2010_l1405_140572


namespace ellipse_eccentricity_l1405_140536

theorem ellipse_eccentricity
  {a b n : ℝ}
  (h1 : a > b) (h2 : b > 0)
  (h3 : ∃ (P : ℝ × ℝ), P.1 = n ∧ P.2 = 4 ∧ (n^2 / a^2 + 16 / b^2 = 1))
  (F1 F2 : ℝ × ℝ)
  (h4 : F1 = (c, 0))        -- Placeholders for focus coordinates of the ellipse
  (h5 : F2 = (-c, 0))
  (h6 : ∃ c, 4*c = (3 / 2) * (a + c))
  : 3 * c = 5 * a → c / a = 3 / 5 :=
by
  sorry

end ellipse_eccentricity_l1405_140536


namespace solution_set_inequality_l1405_140574

theorem solution_set_inequality (x : ℝ) : 
  (x - 1) / (x + 2) > 0 ↔ x > 1 ∨ x < -2 :=
by {
  sorry -- proof omitted
}

end solution_set_inequality_l1405_140574


namespace find_phi_l1405_140520

theorem find_phi (ϕ : ℝ) (h1 : |ϕ| < π / 2)
  (h2 : ∃ k : ℤ, 3 * (π / 12) + ϕ = k * π + π / 2) :
  ϕ = π / 4 :=
by sorry

end find_phi_l1405_140520


namespace remainder_when_n_add_3006_divided_by_6_l1405_140511

theorem remainder_when_n_add_3006_divided_by_6 (n : ℤ) (h : n % 6 = 1) : (n + 3006) % 6 = 1 := by
  sorry

end remainder_when_n_add_3006_divided_by_6_l1405_140511


namespace expected_allergies_correct_expected_both_correct_l1405_140539

noncomputable def p_allergies : ℚ := 2 / 7
noncomputable def sample_size : ℕ := 350
noncomputable def expected_allergies : ℚ := (2 / 7) * 350

noncomputable def p_left_handed : ℚ := 3 / 10
noncomputable def expected_both : ℚ := (3 / 10) * (2 / 7) * 350

theorem expected_allergies_correct : expected_allergies = 100 := by
  sorry

theorem expected_both_correct : expected_both = 30 := by
  sorry

end expected_allergies_correct_expected_both_correct_l1405_140539


namespace range_of_a_l1405_140530

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 1 < 0) ↔ (a < -2 ∨ a > 2) :=
sorry

end range_of_a_l1405_140530


namespace total_stocks_l1405_140571

-- Define the conditions as given in the math problem
def closed_higher : ℕ := 1080
def ratio : ℝ := 1.20

-- Using ℕ for the number of stocks that closed lower
def closed_lower (x : ℕ) : Prop := 1080 = x * ratio ∧ closed_higher = x + x * (1 / 5)

-- Definition to compute the total number of stocks on the stock exchange
def total_number_of_stocks (x : ℕ) : ℕ := closed_higher + x

-- The main theorem to be proved
theorem total_stocks (x : ℕ) (h : closed_lower x) : total_number_of_stocks x = 1980 :=
sorry

end total_stocks_l1405_140571


namespace no_faces_painted_two_or_three_faces_painted_l1405_140527

-- Define the dimensions of the cuboid
def cuboid_length : ℕ := 3
def cuboid_width : ℕ := 4
def cuboid_height : ℕ := 5

-- Define the number of small cubes
def small_cubes_total : ℕ := 60

-- Define the number of small cubes with no faces painted
def small_cubes_no_faces_painted : ℕ := (cuboid_length - 2) * (cuboid_width - 2) * (cuboid_height - 2)

-- Define the number of small cubes with 2 faces painted
def small_cubes_two_faces_painted : ℕ := (cuboid_length - 2) * cuboid_width +
                                          (cuboid_width - 2) * cuboid_length +
                                          (cuboid_height - 2) * cuboid_width

-- Define the number of small cubes with 3 faces painted
def small_cubes_three_faces_painted : ℕ := 8

-- Define the probabilities
def probability_no_faces_painted : ℚ := small_cubes_no_faces_painted / small_cubes_total
def probability_two_or_three_faces_painted : ℚ := (small_cubes_two_faces_painted + small_cubes_three_faces_painted) / small_cubes_total

-- Theorems to prove
theorem no_faces_painted (h : cuboid_length = 3 ∧ cuboid_width = 4 ∧ cuboid_height = 5 ∧ 
                           small_cubes_total = 60 ∧ small_cubes_no_faces_painted = 6) :
  probability_no_faces_painted = 1 / 10 := by
  sorry

theorem two_or_three_faces_painted (h : cuboid_length = 3 ∧ cuboid_width = 4 ∧ cuboid_height = 5 ∧ 
                                    small_cubes_total = 60 ∧ small_cubes_two_faces_painted = 24 ∧
                                    small_cubes_three_faces_painted = 8) :
  probability_two_or_three_faces_painted = 8 / 15 := by
  sorry

end no_faces_painted_two_or_three_faces_painted_l1405_140527


namespace students_in_both_clubs_l1405_140509

theorem students_in_both_clubs
  (T R B total_club_students : ℕ)
  (hT : T = 85) (hR : R = 120)
  (hTotal : T + R - B = total_club_students)
  (hTotalVal : total_club_students = 180) :
  B = 25 :=
by
  -- Placeholder for proof
  sorry

end students_in_both_clubs_l1405_140509


namespace locus_of_vertex_P_l1405_140541

noncomputable def M : ℝ × ℝ := (0, 5)
noncomputable def N : ℝ × ℝ := (0, -5)
noncomputable def perimeter : ℝ := 36

theorem locus_of_vertex_P : ∃ (P : ℝ × ℝ), 
  (∃ (a b : ℝ), a = 13 ∧ b = 12 ∧ P ≠ (0,0) ∧
  (a^2 = b^2 + 5^2) ∧ 
  (perimeter = 2 * a + (5 - (-5))) ∧ 
  ((P.1)^2 / 144 + (P.2)^2 / 169 = 1)) :=
sorry

end locus_of_vertex_P_l1405_140541


namespace total_full_parking_spots_correct_l1405_140576

-- Define the number of parking spots on each level
def total_parking_spots (level : ℕ) : ℕ :=
  100 + (level - 1) * 50

-- Define the number of open spots on each level
def open_parking_spots (level : ℕ) : ℕ :=
  if level = 1 then 58
  else if level <= 4 then 58 - 3 * (level - 1)
  else 49 + 10 * (level - 4)

-- Define the number of full parking spots on each level
def full_parking_spots (level : ℕ) : ℕ :=
  total_parking_spots level - open_parking_spots level

-- Sum up the full parking spots on all 7 levels to get the total full spots
def total_full_parking_spots : ℕ :=
  List.sum (List.map full_parking_spots [1, 2, 3, 4, 5, 6, 7])

-- Theorem to prove the total number of full parking spots
theorem total_full_parking_spots_correct : total_full_parking_spots = 1329 :=
by
  sorry

end total_full_parking_spots_correct_l1405_140576


namespace arithmetic_sequence_problem_l1405_140525

variable {a : ℕ → ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n m : ℕ, a (n+1) = a n + d

theorem arithmetic_sequence_problem
  (h_arith : is_arithmetic_sequence a)
  (h1 : a 1 + a 2 + a 3 = 32)
  (h2 : a 11 + a 12 + a 13 = 118) :
  a 4 + a 10 = 50 :=
sorry

end arithmetic_sequence_problem_l1405_140525


namespace rate_per_meter_for_fencing_l1405_140588

/-- The length of a rectangular plot is 10 meters more than its width. 
    The cost of fencing the plot along its perimeter at a certain rate per meter is Rs. 1430. 
    The perimeter of the plot is 220 meters. 
    Prove that the rate per meter for fencing the plot is 6.5 Rs. 
 -/
theorem rate_per_meter_for_fencing (width length perimeter cost : ℝ)
  (h_length : length = width + 10)
  (h_perimeter : perimeter = 2 * (width + length))
  (h_perimeter_value : perimeter = 220)
  (h_cost : cost = 1430) :
  (cost / perimeter) = 6.5 := by
  sorry

end rate_per_meter_for_fencing_l1405_140588


namespace twenty_three_percent_of_number_is_forty_six_l1405_140581

theorem twenty_three_percent_of_number_is_forty_six (x : ℝ) (h : (23 / 100) * x = 46) : x = 200 :=
sorry

end twenty_three_percent_of_number_is_forty_six_l1405_140581


namespace vector_dot_product_parallel_l1405_140596

noncomputable def a : ℝ × ℝ := (1, 2)
noncomputable def b (m : ℝ) : ℝ × ℝ := (m, -1)
noncomputable def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = k • v

theorem vector_dot_product_parallel (m : ℝ) (h_parallel : is_parallel a (a.1 + m, a.2 + (-1))) :
  (a.1 * m + a.2 * (-1) = -5 / 2) :=
sorry

end vector_dot_product_parallel_l1405_140596


namespace cost_whitewashing_l1405_140555

theorem cost_whitewashing
  (length : ℝ) (breadth : ℝ) (height : ℝ)
  (door_height : ℝ) (door_width : ℝ)
  (window_height : ℝ) (window_width : ℝ)
  (num_windows : ℕ) (cost_per_square_foot : ℝ)
  (room_dimensions : length = 25 ∧ breadth = 15 ∧ height = 12)
  (door_dimensions : door_height = 6 ∧ door_width = 3)
  (window_dimensions : window_height = 4 ∧ window_width = 3)
  (num_windows_condition : num_windows = 3)
  (cost_condition : cost_per_square_foot = 8) :
  (2 * (length + breadth) * height - (door_height * door_width + num_windows * window_height * window_width)) * cost_per_square_foot = 7248 := 
by
  sorry

end cost_whitewashing_l1405_140555


namespace consecutive_even_integers_sum_l1405_140548

theorem consecutive_even_integers_sum :
  ∀ (y : Int), (y = 2 * (y + 2)) → y + (y + 2) = -6 :=
by
  intro y
  intro h
  sorry

end consecutive_even_integers_sum_l1405_140548


namespace sequence_polynomial_exists_l1405_140552

noncomputable def sequence_exists (k : ℕ) : Prop :=
∃ u : ℕ → ℝ,
  (∀ n : ℕ, u (n + 1) - u n = (n : ℝ) ^ k) ∧
  (∃ p : Polynomial ℝ, (∀ n : ℕ, u n = Polynomial.eval (n : ℝ) p) ∧ p.degree = k + 1 ∧ p.leadingCoeff = 1 / (k + 1))

theorem sequence_polynomial_exists (k : ℕ) : sequence_exists k :=
sorry

end sequence_polynomial_exists_l1405_140552


namespace max_number_of_books_laughlin_can_buy_l1405_140519

-- Definitions of costs and the budget constraint
def individual_book_cost : ℕ := 3
def four_book_bundle_cost : ℕ := 10
def seven_book_bundle_cost : ℕ := 15
def budget : ℕ := 20

-- Condition that Laughlin must buy at least one 4-book bundle
def minimum_required_four_book_bundles : ℕ := 1

-- Define the function to calculate the maximum number of books Laughlin can buy
def max_books (budget : ℕ) (individual_book_cost : ℕ) 
              (four_book_bundle_cost : ℕ) (seven_book_bundle_cost : ℕ) 
              (min_four_book_bundles : ℕ) : ℕ :=
  let remaining_budget_after_four_bundle := budget - (min_four_book_bundles * four_book_bundle_cost)
  if remaining_budget_after_four_bundle >= seven_book_bundle_cost then
    min_four_book_bundles * 4 + 7
  else if remaining_budget_after_four_bundle >= individual_book_cost then
    min_four_book_bundles * 4 + remaining_budget_after_four_bundle / individual_book_cost
  else
    min_four_book_bundles * 4

-- Proof statement: Laughlin can buy a maximum of 7 books
theorem max_number_of_books_laughlin_can_buy : 
  max_books budget individual_book_cost four_book_bundle_cost seven_book_bundle_cost minimum_required_four_book_bundles = 7 :=
by
  sorry

end max_number_of_books_laughlin_can_buy_l1405_140519


namespace contradiction_proof_l1405_140532

theorem contradiction_proof (a b : ℝ) (h : a + b ≥ 0) : ¬ (a < 0 ∧ b < 0) :=
by
  sorry

end contradiction_proof_l1405_140532


namespace area_of_union_of_triangle_and_reflection_l1405_140589

-- Define points in ℝ²
structure Point where
  x : ℝ
  y : ℝ

-- Define the vertices of the original triangle
def A : Point := ⟨2, 3⟩
def B : Point := ⟨4, -1⟩
def C : Point := ⟨7, 0⟩

-- Define the vertices of the reflected triangle
def A' : Point := ⟨-2, 3⟩
def B' : Point := ⟨-4, -1⟩
def C' : Point := ⟨-7, 0⟩

-- Calculate the area of a triangle given three points
def triangleArea (P Q R : Point) : ℝ :=
  0.5 * |P.x * (Q.y - R.y) + Q.x * (R.y - P.y) + R.x * (P.y - Q.y)|

-- Statement to prove: the area of the union of the original and reflected triangles
theorem area_of_union_of_triangle_and_reflection :
  triangleArea A B C + triangleArea A' B' C' = 14 := 
sorry

end area_of_union_of_triangle_and_reflection_l1405_140589


namespace distance_between_parallel_lines_l1405_140595

theorem distance_between_parallel_lines (A B C1 C2 : ℝ) (hA : A = 2) (hB : B = 4)
  (hC1 : C1 = -8) (hC2 : C2 = 7) : 
  (|C2 - C1| / (Real.sqrt (A^2 + B^2)) = 3 * Real.sqrt 5 / 2) :=
by
  rw [hA, hB, hC1, hC2]
  sorry

end distance_between_parallel_lines_l1405_140595


namespace triangle_area_solution_l1405_140562

noncomputable def triangle_area (a b : ℝ) : ℝ := 
  let r := 6 -- radius of each circle
  let d := 2 -- derived distance
  let s := 2 * Real.sqrt 3 * d -- side length of the equilateral triangle
  let area := (Real.sqrt 3 / 4) * s^2 
  area

theorem triangle_area_solution : ∃ a b : ℝ, 
  triangle_area a b = 3 * Real.sqrt 3 ∧ 
  a + b = 27 := 
by 
  exists 27
  exists 3
  sorry

end triangle_area_solution_l1405_140562


namespace polygon_diagonals_l1405_140557

theorem polygon_diagonals (n : ℕ) (k_0 k_1 k_2 : ℕ)
  (h1 : 2 * k_2 + k_1 = n)
  (h2 : k_2 + k_1 + k_0 = n - 2) :
  k_2 ≥ 2 :=
sorry

end polygon_diagonals_l1405_140557


namespace sum_of_numbers_is_216_l1405_140501

-- Define the conditions and what needs to be proved.
theorem sum_of_numbers_is_216 
  (x : ℕ) 
  (h_lcm : Nat.lcm (2 * x) (Nat.lcm (3 * x) (7 * x)) = 126) : 
  2 * x + 3 * x + 7 * x = 216 :=
by
  sorry

end sum_of_numbers_is_216_l1405_140501


namespace train_speed_is_correct_l1405_140505

-- Define the conditions
def length_of_train : ℕ := 140 -- length in meters
def time_to_cross_pole : ℕ := 7 -- time in seconds

-- Define the expected speed in km/h
def expected_speed_in_kmh : ℕ := 72 -- speed in km/h

-- Prove that the speed of the train in km/h is 72
theorem train_speed_is_correct :
  (length_of_train / time_to_cross_pole) * 36 / 10 = expected_speed_in_kmh :=
by
  sorry

end train_speed_is_correct_l1405_140505


namespace book_pricing_and_min_cost_l1405_140570

-- Define the conditions
def price_relation (a : ℝ) (ps_price : ℝ) : Prop :=
  ps_price = 1.2 * a

def book_count_relation (a : ℝ) (lit_count ps_count : ℕ) : Prop :=
  lit_count = 1200 / a ∧ ps_count = 1200 / (1.2 * a) ∧ lit_count - ps_count = 10

def min_cost_condition (x : ℕ) : Prop :=
  x ≤ 600

def total_cost (x : ℕ) : ℝ :=
  20 * x + 24 * (1000 - x)

-- The theorem combining all parts
theorem book_pricing_and_min_cost:
  ∃ (a : ℝ) (ps_price : ℝ) (lit_count ps_count : ℕ),
    price_relation a ps_price ∧
    book_count_relation a lit_count ps_count ∧
    a = 20 ∧ ps_price = 24 ∧
    (∀ (x : ℕ), min_cost_condition x → total_cost x ≥ 21600) ∧
    (total_cost 600 = 21600) :=
by
  sorry

end book_pricing_and_min_cost_l1405_140570


namespace problem_statement_l1405_140598

noncomputable def f (x a : ℝ) : ℝ := x^2 + (2 * a - 8) * x

theorem problem_statement
  (f : ℝ → ℝ → ℝ)
  (sol_set : Set ℝ)
  (cond1 : ∀ a : ℝ, sol_set = {x : ℝ | -1 ≤ x ∧ x ≤ 5} → ∀ x : ℝ, f x a ≤ 5 ↔ x ∈ sol_set)
  (cond2 : ∀ x : ℝ, ∀ m : ℝ, f x 2 ≥ m^2 - 4 * m - 9) :
  (∃ a : ℝ, a = 2) ∧ (∀ m : ℝ, -1 ≤ m ∧ m ≤ 5) :=
by
  sorry

end problem_statement_l1405_140598


namespace decimal_to_fraction_l1405_140585

theorem decimal_to_fraction (x : ℝ) (h : x = 2.35) : ∃ (a b : ℤ), (b ≠ 0) ∧ (a / b = x) ∧ (a = 47) ∧ (b = 20) := by
  sorry

end decimal_to_fraction_l1405_140585


namespace jerseys_sold_l1405_140507

theorem jerseys_sold (unit_price_jersey : ℕ) (total_revenue_jersey : ℕ) (n : ℕ) 
  (h_unit_price : unit_price_jersey = 165) 
  (h_total_revenue : total_revenue_jersey = 25740) 
  (h_eq : n * unit_price_jersey = total_revenue_jersey) : 
  n = 156 :=
by
  rw [h_unit_price, h_total_revenue] at h_eq
  sorry

end jerseys_sold_l1405_140507


namespace train_speed_clicks_l1405_140538

theorem train_speed_clicks (x : ℝ) (rail_length_feet : ℝ := 40) (clicks_per_mile : ℝ := 5280/ 40) :
  15 ≤ (2400/5280) * 60  * clicks_per_mile ∧ (2400/5280) * 60 * clicks_per_mile ≤ 30 :=
by {
  sorry
}

end train_speed_clicks_l1405_140538


namespace min_squares_to_cover_staircase_l1405_140524

-- Definition of the staircase and the constraints
def is_staircase (n : ℕ) (s : ℕ → ℕ) : Prop :=
  ∀ i, i < n → s i = i + 1

-- The proof problem statement
theorem min_squares_to_cover_staircase : 
  ∀ n : ℕ, n = 15 →
  ∀ s : ℕ → ℕ, is_staircase n s →
  ∃ k : ℕ, k = 15 ∧ (∀ i, i < n → ∃ a b : ℕ, a ≤ i ∧ b ≤ s a ∧ ∃ (l : ℕ), l = 1) :=
by
  sorry

end min_squares_to_cover_staircase_l1405_140524


namespace grandpa_age_times_jungmin_age_l1405_140580

-- Definitions based on the conditions
def grandpa_age_last_year : ℕ := 71
def jungmin_age_last_year : ℕ := 8
def grandpa_age_this_year : ℕ := grandpa_age_last_year + 1
def jungmin_age_this_year : ℕ := jungmin_age_last_year + 1

-- The statement to prove
theorem grandpa_age_times_jungmin_age :
  grandpa_age_this_year / jungmin_age_this_year = 8 :=
by
  sorry

end grandpa_age_times_jungmin_age_l1405_140580


namespace susan_fraction_apples_given_out_l1405_140558

theorem susan_fraction_apples_given_out (frank_apples : ℕ) (frank_sold_fraction : ℚ) 
  (total_remaining_apples : ℕ) (susan_multiple : ℕ) 
  (H1 : frank_apples = 36) 
  (H2 : susan_multiple = 3) 
  (H3 : frank_sold_fraction = 1 / 3) 
  (H4 : total_remaining_apples = 78) :
  let susan_apples := susan_multiple * frank_apples
  let frank_sold_apples := frank_sold_fraction * frank_apples
  let frank_remaining_apples := frank_apples - frank_sold_apples
  let total_before_susan_gave_out := susan_apples + frank_remaining_apples
  let susan_gave_out := total_before_susan_gave_out - total_remaining_apples
  let susan_gave_fraction := susan_gave_out / susan_apples
  susan_gave_fraction = 1 / 2 :=
by
  sorry

end susan_fraction_apples_given_out_l1405_140558


namespace neg_p_false_sufficient_but_not_necessary_for_p_or_q_l1405_140594

variable (p q : Prop)

theorem neg_p_false_sufficient_but_not_necessary_for_p_or_q :
  (¬ p = false) → (p ∨ q) ∧ ¬((p ∨ q) → (¬ p = false)) :=
by
  sorry

end neg_p_false_sufficient_but_not_necessary_for_p_or_q_l1405_140594


namespace necessary_but_not_sufficient_conditions_l1405_140518

theorem necessary_but_not_sufficient_conditions (x y : ℝ) :
  (|x| ≤ 1 ∧ |y| ≤ 1) → x^2 + y^2 ≤ 1 ∨ ¬(x^2 + y^2 ≤ 1) → 
  (|x| ≤ 1 ∧ |y| ≤ 1) → (x^2 + y^2 ≤ 1 → (|x| ≤ 1 ∧ |y| ≤ 1)) :=
by
  sorry

end necessary_but_not_sufficient_conditions_l1405_140518


namespace luke_can_see_silvia_for_22_point_5_minutes_l1405_140569

/--
Luke is initially 0.75 miles behind Silvia. Luke rollerblades at 10 mph and Silvia cycles 
at 6 mph. Luke can see Silvia until she is 0.75 miles behind him. Prove that Luke can see 
Silvia for a total of 22.5 minutes.
-/
theorem luke_can_see_silvia_for_22_point_5_minutes :
    let distance := (3 / 4 : ℝ)
    let luke_speed := (10 : ℝ)
    let silvia_speed := (6 : ℝ)
    let relative_speed := luke_speed - silvia_speed
    let time_to_reach := distance / relative_speed
    let total_time := 2 * time_to_reach * 60 
    total_time = 22.5 :=
by
    sorry

end luke_can_see_silvia_for_22_point_5_minutes_l1405_140569


namespace inequality_for_positive_reals_l1405_140597

variable {a b c : ℝ}
variable {k : ℕ}

theorem inequality_for_positive_reals 
  (hab : a > 0) 
  (hbc : b > 0) 
  (hac : c > 0) 
  (hprod : a * b * c = 1) 
  (hk : k ≥ 2) 
  : (a ^ k) / (a + b) + (b ^ k) / (b + c) + (c ^ k) / (c + a) ≥ 3 / 2 := 
sorry

end inequality_for_positive_reals_l1405_140597


namespace common_difference_arithmetic_sequence_l1405_140547

noncomputable def a_n (n : ℕ) : ℤ := 5 - 4 * n

theorem common_difference_arithmetic_sequence :
  ∀ n ≥ 1, a_n n - a_n (n - 1) = -4 :=
by
  intros n hn
  unfold a_n
  sorry

end common_difference_arithmetic_sequence_l1405_140547


namespace Petya_wins_l1405_140561

theorem Petya_wins (n : ℕ) (h₁ : n = 2016) : (∀ m : ℕ, m < n → ∀ k : ℕ, k ∣ m ∧ k ≠ m → m - k = 1 → false) :=
sorry

end Petya_wins_l1405_140561


namespace max_sides_in_subpolygon_l1405_140517

/-- In a convex 1950-sided polygon with all its diagonals drawn, the polygon with the greatest number of sides among these smaller polygons can have at most 1949 sides. -/
theorem max_sides_in_subpolygon (n : ℕ) (hn : n = 1950) : 
  ∃ p : ℕ, p = 1949 ∧ ∀ m, m ≤ n-2 → m ≤ 1949 :=
sorry

end max_sides_in_subpolygon_l1405_140517


namespace find_sin_θ_l1405_140545

open Real

noncomputable def θ_in_range_and_sin_2θ (θ : ℝ) : Prop :=
  (θ ∈ Set.Icc (π / 4) (π / 2)) ∧ (sin (2 * θ) = 3 * sqrt 7 / 8)

theorem find_sin_θ (θ : ℝ) (h : θ_in_range_and_sin_2θ θ) : sin θ = 3 / 4 :=
  sorry

end find_sin_θ_l1405_140545


namespace truck_travel_distance_l1405_140559

noncomputable def truck_distance (gallons: ℕ) : ℕ :=
  let efficiency_10_gallons := 300 / 10 -- miles per gallon
  let efficiency_initial := efficiency_10_gallons
  let efficiency_decreased := efficiency_initial * 9 / 10 -- 10% decrease
  if gallons <= 12 then
    gallons * efficiency_initial
  else
    12 * efficiency_initial + (gallons - 12) * efficiency_decreased

theorem truck_travel_distance (gallons: ℕ) :
  gallons = 15 → truck_distance gallons = 441 :=
by
  intros h
  rw [h]
  -- skipping proof
  sorry

end truck_travel_distance_l1405_140559


namespace paintings_on_Sep27_l1405_140550

-- Definitions for the problem conditions
def total_days := 6
def paintings_per_2_days := (6 : ℕ)
def paintings_per_3_days := (8 : ℕ)
def paintings_P22_to_P26 := 30

-- Function to calculate paintings over a given period
def paintings_in_days (days : ℕ) (frequency : ℕ) : ℕ := days / frequency

-- Function to calculate total paintings from the given artists
def total_paintings (d : ℕ) (p2 : ℕ) (p3 : ℕ) : ℕ :=
  p2 * paintings_in_days d 2 + p3 * paintings_in_days d 3

-- Calculate total paintings in 6 days
def total_paintings_in_6_days := total_paintings total_days paintings_per_2_days paintings_per_3_days

-- Proof problem: Show the number of paintings on the last day (September 27)
theorem paintings_on_Sep27 : total_paintings_in_6_days - paintings_P22_to_P26 = 4 :=
by
  sorry

end paintings_on_Sep27_l1405_140550


namespace prime_square_minus_one_divisible_by_24_l1405_140546

theorem prime_square_minus_one_divisible_by_24 (p : ℕ) (hp : Prime p) (hp_ge_5 : 5 ≤ p) : 24 ∣ (p^2 - 1) := 
by 
sorry

end prime_square_minus_one_divisible_by_24_l1405_140546


namespace tank_capacity_l1405_140575

theorem tank_capacity (c w : ℝ) 
  (h1 : w / c = 1 / 7) 
  (h2 : (w + 5) / c = 1 / 5) : 
  c = 87.5 := 
by
  sorry

end tank_capacity_l1405_140575


namespace smallest_S_value_l1405_140529

def num_list := {x : ℕ // 1 ≤ x ∧ x ≤ 9}

def S (a b c : num_list) (d e f : num_list) (g h i : num_list) : ℕ :=
  a.val * b.val * c.val + d.val * e.val * f.val + g.val * h.val * i.val

theorem smallest_S_value :
  ∃ a b c d e f g h i : num_list,
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i ∧
  S a b c d e f g h i = 214 :=
sorry

end smallest_S_value_l1405_140529


namespace ab_power_2023_l1405_140573

theorem ab_power_2023 (a b : ℤ) (h : |a + 2| + (b - 1) ^ 2 = 0) : (a + b) ^ 2023 = -1 :=
by
  sorry

end ab_power_2023_l1405_140573


namespace sum_one_to_twenty_nine_l1405_140568

theorem sum_one_to_twenty_nine : (29 / 2) * (1 + 29) = 435 := by
  -- proof
  sorry

end sum_one_to_twenty_nine_l1405_140568


namespace sticker_probability_l1405_140553

theorem sticker_probability 
  (n : ℕ) (k : ℕ) (uncollected : ℕ) (collected : ℕ) (C : ℕ → ℕ → ℕ) :
  n = 18 → k = 10 → uncollected = 6 → collected = 12 → 
  (C uncollected uncollected) * (C collected (k - uncollected)) = 495 → 
  C n k = 43758 → 
  (495 : ℚ) / 43758 = 5 / 442 := 
by
  intros h_n h_k h_uncollected h_collected h_C1 h_C2
  sorry

end sticker_probability_l1405_140553


namespace volume_frustum_fraction_l1405_140564

-- Define the base edge and initial altitude of the pyramid.
def base_edge := 32 -- in inches
def altitude_original := 1 -- in feet

-- Define the fractional part representing the altitude of the smaller pyramid.
def altitude_fraction := 1/4

-- Define the volume of the original pyramid being V.
noncomputable def volume_original : ℝ := (1/3) * (base_edge ^ 2) * altitude_original

-- Define the volume of the smaller pyramid being removed.
noncomputable def volume_smaller : ℝ := (1/3) * ((altitude_fraction * base_edge) ^ 2) * (altitude_fraction * altitude_original)

-- We now state the proof
theorem volume_frustum_fraction : 
  (volume_original - volume_smaller) / volume_original = 63/64 :=
by
  sorry

end volume_frustum_fraction_l1405_140564


namespace minimum_cuts_for_48_pieces_l1405_140544

theorem minimum_cuts_for_48_pieces 
  (rearrange_without_folding : Prop)
  (can_cut_multiple_layers_simultaneously : Prop)
  (straight_line_cut : Prop)
  (cut_doubles_pieces : ∀ n, ∃ m, m = 2 * n) :
  ∃ n, (2^n ≥ 48 ∧ ∀ m, (m < n → 2^m < 48)) ∧ n = 6 := 
by 
  sorry

end minimum_cuts_for_48_pieces_l1405_140544


namespace sum_of_coefficients_l1405_140515

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def polynomial (a : ℝ) (x : ℝ) : ℝ :=
  (2 + a * x) * (1 + x)^5

def x2_coefficient_condition (a : ℝ) : Prop :=
  2 * binomial_coefficient 5 2 + a * binomial_coefficient 5 1 = 15

theorem sum_of_coefficients (a : ℝ) (h : x2_coefficient_condition a) : 
  polynomial a 1 = 64 := 
sorry

end sum_of_coefficients_l1405_140515


namespace perpendicular_bisector_l1405_140577

theorem perpendicular_bisector (x y : ℝ) :
  (x - 2 * y + 1 = 0 ∧ -1 ≤ x ∧ x ≤ 3) → (2 * x + y - 3 = 0) :=
by
  sorry

end perpendicular_bisector_l1405_140577


namespace height_of_parallelogram_l1405_140590

theorem height_of_parallelogram (A B h : ℝ) (hA : A = 72) (hB : B = 12) (h_area : A = B * h) : h = 6 := by
  sorry

end height_of_parallelogram_l1405_140590


namespace smallest_collected_l1405_140522

noncomputable def Yoongi_collections : ℕ := 4
noncomputable def Jungkook_collections : ℕ := 6 / 3
noncomputable def Yuna_collections : ℕ := 5

theorem smallest_collected : min (min Yoongi_collections Jungkook_collections) Yuna_collections = 2 :=
by
  sorry

end smallest_collected_l1405_140522


namespace percentage_slump_in_business_l1405_140526

theorem percentage_slump_in_business (X Y : ℝ) (h1 : 0.04 * X = 0.05 * Y) : 
  (1 - Y / X) * 100 = 20 :=
by
  sorry

end percentage_slump_in_business_l1405_140526


namespace Gabrielle_sells_8_crates_on_Wednesday_l1405_140566

-- Definitions based on conditions from part a)
def crates_sold_on_Monday := 5
def crates_sold_on_Tuesday := 2 * crates_sold_on_Monday
def crates_sold_on_Thursday := crates_sold_on_Tuesday / 2
def total_crates_sold := 28
def crates_sold_on_Wednesday := total_crates_sold - (crates_sold_on_Monday + crates_sold_on_Tuesday + crates_sold_on_Thursday)

-- The theorem to prove the question == answer given conditions
theorem Gabrielle_sells_8_crates_on_Wednesday : crates_sold_on_Wednesday = 8 := by
  sorry

end Gabrielle_sells_8_crates_on_Wednesday_l1405_140566


namespace price_of_each_armchair_l1405_140502

theorem price_of_each_armchair
  (sofa_price : ℕ)
  (coffee_table_price : ℕ)
  (total_invoice : ℕ)
  (num_armchairs : ℕ)
  (h_sofa : sofa_price = 1250)
  (h_coffee_table : coffee_table_price = 330)
  (h_invoice : total_invoice = 2430)
  (h_num_armchairs : num_armchairs = 2) :
  (total_invoice - (sofa_price + coffee_table_price)) / num_armchairs = 425 := 
by 
  sorry

end price_of_each_armchair_l1405_140502


namespace probability_Hugo_first_roll_is_six_l1405_140508

/-
In a dice game, each of 5 players, including Hugo, rolls a standard 6-sided die. 
The winner is the player who rolls the highest number. 
In the event of a tie for the highest roll, those involved in the tie roll again until a clear winner emerges.
-/
variable (HugoRoll : Nat) (A1 B1 C1 D1 : Nat)
variable (W : Bool)

-- Conditions in the problem
def isWinner (HugoRoll : Nat) (W : Bool) : Prop := (W = true)
def firstRollAtLeastFour (HugoRoll : Nat) : Prop := HugoRoll >= 4
def firstRollIsSix (HugoRoll : Nat) : Prop := HugoRoll = 6

-- Hypotheses: Hugo's event conditions
axiom HugoWonAndRollsAtLeastFour : isWinner HugoRoll W ∧ firstRollAtLeastFour HugoRoll

-- Target probability based on problem statement
noncomputable def probability (p : ℚ) : Prop := p = 625 / 4626

-- Main statement
theorem probability_Hugo_first_roll_is_six (HugoRoll : Nat) (A1 B1 C1 D1 : Nat) (W : Bool) :
  isWinner HugoRoll W ∧ firstRollAtLeastFour HugoRoll → 
  probability (625 / 4626) := by
  sorry


end probability_Hugo_first_roll_is_six_l1405_140508


namespace base9_minus_base6_l1405_140586

-- Definitions from conditions
def base9_to_base10 (n : Nat) : Nat :=
  match n with
  | 325 => 3 * 9^2 + 2 * 9^1 + 5 * 9^0
  | _ => 0

def base6_to_base10 (n : Nat) : Nat :=
  match n with
  | 231 => 2 * 6^2 + 3 * 6^1 + 1 * 6^0
  | _ => 0

-- Main theorem statement
theorem base9_minus_base6 : base9_to_base10 325 - base6_to_base10 231 = 175 :=
by
  sorry

end base9_minus_base6_l1405_140586


namespace marble_count_l1405_140549

variable (initial_mar: Int) (lost_mar: Int)

def final_mar (initial_mar: Int) (lost_mar: Int) : Int :=
  initial_mar - lost_mar

theorem marble_count : final_mar 16 7 = 9 := by
  trivial

end marble_count_l1405_140549


namespace systematic_sampling_id_fourth_student_l1405_140563

theorem systematic_sampling_id_fourth_student (n : ℕ) (a b c d : ℕ) (h1 : n = 54) 
(h2 : a = 3) (h3 : b = 29) (h4 : c = 42) (h5 : d = a + 13) : d = 16 :=
by
  sorry

end systematic_sampling_id_fourth_student_l1405_140563


namespace equal_animals_per_aquarium_l1405_140514

theorem equal_animals_per_aquarium (aquariums animals : ℕ) (h1 : aquariums = 26) (h2 : animals = 52) (h3 : ∀ a, a = animals / aquariums) : a = 2 := 
by
  sorry

end equal_animals_per_aquarium_l1405_140514


namespace mark_more_hours_l1405_140534

-- Definitions based on the conditions
variables (Pat Kate Mark Alex : ℝ)
variables (total_hours : ℝ)
variables (h1 : Pat + Kate + Mark + Alex = 350)
variables (h2 : Pat = 2 * Kate)
variables (h3 : Pat = (1 / 3) * Mark)
variables (h4 : Alex = 1.5 * Kate)

-- Theorem statement with the desired proof target
theorem mark_more_hours (Pat Kate Mark Alex : ℝ) (h1 : Pat + Kate + Mark + Alex = 350) 
(h2 : Pat = 2 * Kate) (h3 : Pat = (1 / 3) * Mark) (h4 : Alex = 1.5 * Kate) : 
Mark - (Kate + Alex) = 116.66666666666667 := sorry

end mark_more_hours_l1405_140534


namespace pyramids_from_cuboid_l1405_140582

-- Define the vertices of a cuboid
def vertices_of_cuboid : ℕ := 8

-- Define the edges of a cuboid
def edges_of_cuboid : ℕ := 12

-- Define the faces of a cuboid
def faces_of_cuboid : ℕ := 6

-- Define the combinatoric calculation
def combinations (n k : ℕ) : ℕ := (n.choose k)

-- Define the total number of tetrahedrons formed
def total_tetrahedrons : ℕ := combinations 7 3 - faces_of_cuboid * combinations 4 3

-- Define the expected result
def expected_tetrahedrons : ℕ := 106

-- The theorem statement to prove that the total number of tetrahedrons is 106
theorem pyramids_from_cuboid : total_tetrahedrons = expected_tetrahedrons :=
by
  sorry

end pyramids_from_cuboid_l1405_140582


namespace hall_length_width_difference_l1405_140500

variable (L W : ℕ)

theorem hall_length_width_difference (h₁ : W = 1 / 2 * L) (h₂ : L * W = 800) :
  L - W = 20 :=
sorry

end hall_length_width_difference_l1405_140500


namespace clark_discount_l1405_140565

theorem clark_discount (price_per_part : ℕ) (number_of_parts : ℕ) (amount_paid : ℕ)
  (h1 : price_per_part = 80)
  (h2 : number_of_parts = 7)
  (h3 : amount_paid = 439) : 
  (number_of_parts * price_per_part) - amount_paid = 121 := by
  sorry

end clark_discount_l1405_140565


namespace probability_of_less_than_5_is_one_half_l1405_140504

noncomputable def probability_of_less_than_5 : ℚ :=
  let total_outcomes := 8
  let successful_outcomes := 4
  successful_outcomes / total_outcomes

theorem probability_of_less_than_5_is_one_half :
  probability_of_less_than_5 = 1 / 2 :=
by
  -- proof omitted
  sorry

end probability_of_less_than_5_is_one_half_l1405_140504


namespace value_of_m_l1405_140512

-- Define the function given m
def f (x m : ℝ) : ℝ := x^2 - 2 * (abs x) + 2 - m

-- State the theorem to be proved
theorem value_of_m (m : ℝ) :
  (∃ x1 x2 x3 : ℝ, f x1 m = 0 ∧ f x2 m = 0 ∧ f x3 m = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1) →
  m = 2 :=
by
  sorry

end value_of_m_l1405_140512


namespace solve_equation_l1405_140560

theorem solve_equation : ∀ x : ℝ, (2 / 3 * x - 2 = 4) → x = 9 :=
by
  intro x
  intro h
  sorry

end solve_equation_l1405_140560


namespace repeating_decimal_sum_l1405_140506

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_4 : ℚ := 4 / 9
noncomputable def repeating_decimal_7 : ℚ := 7 / 9

theorem repeating_decimal_sum : 
  repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_4 - repeating_decimal_7 = -1 / 3 :=
by {
  sorry
}

end repeating_decimal_sum_l1405_140506


namespace probability_sales_greater_than_10000_l1405_140554

/-- Define the probability that the sales of new energy vehicles in a randomly selected city are greater than 10000 -/
theorem probability_sales_greater_than_10000 :
  (1 / 2) * (2 / 10) + (1 / 2) * (6 / 10) = 2 / 5 :=
by sorry

end probability_sales_greater_than_10000_l1405_140554


namespace area_of_polygon_l1405_140543

theorem area_of_polygon (side_length n : ℕ) (h1 : n = 36) (h2 : 36 * side_length = 72) (h3 : ∀ i, i < n → (∃ a, ∃ b, (a + b = 4) ∧ (i = 4 * a + b))) :
  (n / 4) * side_length ^ 2 = 144 :=
by
  sorry

end area_of_polygon_l1405_140543


namespace max_det_value_l1405_140579

theorem max_det_value :
  ∃ θ : ℝ, 
    (1 * ((5 + Real.sin θ) * 9 - 6 * 8) 
     - 2 * (4 * 9 - 6 * (7 + Real.cos θ)) 
     + 3 * (4 * 8 - (5 + Real.sin θ) * (7 + Real.cos θ))) 
     = 93 :=
sorry

end max_det_value_l1405_140579


namespace reciprocal_eq_self_l1405_140513

theorem reciprocal_eq_self {x : ℝ} (h : x ≠ 0) : (1 / x = x) → (x = 1 ∨ x = -1) :=
by
  intro h1
  sorry

end reciprocal_eq_self_l1405_140513


namespace food_expenditure_increase_l1405_140537

-- Conditions
def linear_relationship (x : ℝ) : ℝ := 0.254 * x + 0.321

-- Proof statement
theorem food_expenditure_increase (x : ℝ) : linear_relationship (x + 1) - linear_relationship x = 0.254 :=
by
  sorry

end food_expenditure_increase_l1405_140537


namespace geometric_sequence_decreasing_iff_l1405_140510

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

noncomputable def is_decreasing_sequence (a : ℕ → ℝ) : Prop := ∀ n : ℕ, a n > a (n + 1)

theorem geometric_sequence_decreasing_iff (a : ℕ → ℝ) (h : is_geometric_sequence a) :
  (a 0 > a 1 ∧ a 1 > a 2) ↔ is_decreasing_sequence a :=
by
  sorry

end geometric_sequence_decreasing_iff_l1405_140510


namespace find_k_l1405_140584

theorem find_k (k : ℚ) : (∀ x y : ℚ, (x, y) = (2, 1) → 3 * k * x - k = -4 * y - 2) → k = -(6 / 5) :=
by
  intro h
  have key := h 2 1 rfl
  have : 3 * k * 2 - k = -4 * 1 - 2 := key
  linarith

end find_k_l1405_140584


namespace relative_speed_of_trains_l1405_140599

def kmph_to_mps (speed : ℕ) : ℕ := speed * 1000 / 3600

theorem relative_speed_of_trains 
  (speed_train1_kmph : ℕ) 
  (speed_train2_kmph : ℕ) 
  (h1 : speed_train1_kmph = 216) 
  (h2 : speed_train2_kmph = 180) : 
  kmph_to_mps speed_train1_kmph - kmph_to_mps speed_train2_kmph = 10 := 
by 
  sorry

end relative_speed_of_trains_l1405_140599


namespace subset_sum_bounds_l1405_140535

theorem subset_sum_bounds (M m n : ℕ) (A : Finset ℕ)
  (h1 : 1 ≤ m) (h2 : m ≤ n) (h3 : 1 ≤ M) (h4 : M ≤ (m * (m + 1)) / 2) (hA : A.card = m) (hA_subset : ∀ x ∈ A, x ∈ Finset.range (n + 1)) :
  ∃ B ⊆ A, 0 ≤ (B.sum id) - M ∧ (B.sum id) - M ≤ n - m :=
by
  sorry

end subset_sum_bounds_l1405_140535


namespace fourth_graders_bought_more_markers_l1405_140591

-- Define the conditions
def cost_per_marker : ℕ := 20
def total_payment_fifth_graders : ℕ := 180
def total_payment_fourth_graders : ℕ := 200

-- Compute the number of markers bought by fifth and fourth graders
def markers_bought_by_fifth_graders : ℕ := total_payment_fifth_graders / cost_per_marker
def markers_bought_by_fourth_graders : ℕ := total_payment_fourth_graders / cost_per_marker

-- Statement to prove
theorem fourth_graders_bought_more_markers : 
  markers_bought_by_fourth_graders - markers_bought_by_fifth_graders = 1 := by
  sorry

end fourth_graders_bought_more_markers_l1405_140591


namespace multiply_polynomials_l1405_140540

theorem multiply_polynomials (x : ℝ) :
  (x^4 + 8 * x^2 + 64) * (x^2 - 8) = x^4 + 16 * x^2 :=
by
  sorry

end multiply_polynomials_l1405_140540


namespace evaluate_expr_correct_l1405_140567

def evaluate_expr : Prop :=
  (8 : ℝ) / (4 * 25) = (0.8 : ℝ) / (0.4 * 25)

theorem evaluate_expr_correct : evaluate_expr :=
by
  sorry

end evaluate_expr_correct_l1405_140567


namespace smallest_tree_height_correct_l1405_140593

-- Defining the conditions
def TallestTreeHeight : ℕ := 108
def MiddleTreeHeight (tallest : ℕ) : ℕ := (tallest / 2) - 6
def SmallestTreeHeight (middle : ℕ) : ℕ := middle / 4

-- Proof statement
theorem smallest_tree_height_correct :
  SmallestTreeHeight (MiddleTreeHeight TallestTreeHeight) = 12 :=
by
  -- Here we would put the proof, but we are skipping it with sorry.
  sorry

end smallest_tree_height_correct_l1405_140593


namespace arithmetic_sequence_sum_and_mean_l1405_140503

theorem arithmetic_sequence_sum_and_mean :
  let a1 := 1
  let d := 2
  let an := 21
  let n := 11
  let S := (n / 2) * (a1 + an)
  S = 121 ∧ (S / n) = 11 :=
by
  let a1 := 1
  let d := 2
  let an := 21
  let n := 11
  let S := (n / 2) * (a1 + an)
  have h1 : S = 121 := sorry
  have h2 : (S / n) = 11 := by
    rw [h1]
    exact sorry
  exact ⟨h1, h2⟩

end arithmetic_sequence_sum_and_mean_l1405_140503


namespace function_relationship_area_60_maximum_area_l1405_140587

-- Definitions and conditions
def perimeter := 32
def side_length (x : ℝ) : ℝ := 16 - x  -- One side of the rectangle
def area (x : ℝ) : ℝ := x * (16 - x)

-- Theorem 1: Function relationship between y and x
theorem function_relationship (x : ℝ) (hx : 0 < x ∧ x < 16) : area x = -x^2 + 16 * x :=
by
  sorry

-- Theorem 2: Values of x when the area is 60 square meters
theorem area_60 (x : ℝ) (hx1 : area x = 60) : x = 6 ∨ x = 10 :=
by
  sorry

-- Theorem 3: Maximum area
theorem maximum_area : ∃ x, area x = 64 ∧ x = 8 :=
by
  sorry

end function_relationship_area_60_maximum_area_l1405_140587


namespace number_from_division_l1405_140583

theorem number_from_division (number : ℝ) (h : number / 2000 = 0.012625) : number = 25.25 :=
by
  sorry

end number_from_division_l1405_140583


namespace total_leaves_on_farm_l1405_140528

noncomputable def number_of_branches : ℕ := 10
noncomputable def sub_branches_per_branch : ℕ := 40
noncomputable def leaves_per_sub_branch : ℕ := 60
noncomputable def number_of_trees : ℕ := 4

theorem total_leaves_on_farm :
  number_of_branches * sub_branches_per_branch * leaves_per_sub_branch * number_of_trees = 96000 :=
by
  sorry

end total_leaves_on_farm_l1405_140528


namespace coterminal_angle_neg_60_eq_300_l1405_140521

theorem coterminal_angle_neg_60_eq_300 :
  ∃ k : ℤ, 0 ≤ k * 360 - 60 ∧ k * 360 - 60 < 360 ∧ (k * 360 - 60 = 300) := by
  sorry

end coterminal_angle_neg_60_eq_300_l1405_140521


namespace ball_bouncing_height_l1405_140556

theorem ball_bouncing_height : ∃ (b : ℕ), 400 * (3/4 : ℝ)^b < 50 ∧ ∀ n < b, 400 * (3/4 : ℝ)^n ≥ 50 :=
by
  use 8
  sorry

end ball_bouncing_height_l1405_140556


namespace sum_of_powers_of_four_to_50_l1405_140516

theorem sum_of_powers_of_four_to_50 :
  2 * (Finset.sum (Finset.range 51) (λ x => x^4)) = 1301700 := by
  sorry

end sum_of_powers_of_four_to_50_l1405_140516


namespace largest_of_five_l1405_140551

def a : ℝ := 0.994
def b : ℝ := 0.9399
def c : ℝ := 0.933
def d : ℝ := 0.9940
def e : ℝ := 0.9309

theorem largest_of_five : (a > b ∧ a > c ∧ a ≥ d ∧ a > e) := by
  -- We add sorry here to skip the proof
  sorry

end largest_of_five_l1405_140551


namespace percent_difference_l1405_140542

theorem percent_difference : 0.12 * 24.2 - 0.10 * 14.2 = 1.484 := by
  sorry

end percent_difference_l1405_140542


namespace bus_fare_with_train_change_in_total_passengers_l1405_140578

variables (p : ℝ) (q : ℝ) (TC : ℝ → ℝ)
variables (p_train : ℝ) (train_capacity : ℝ)

-- Demand function
def demand_function (p : ℝ) : ℝ := 4200 - 100 * p

-- Train fare is fixed
def train_fare : ℝ := 4

-- Train capacity
def train_cap : ℝ := 800

-- Bus total cost function
def total_cost (y : ℝ) : ℝ := 10 * y + 225

-- Case when there is competition (train available)
def optimal_bus_fare_with_train : ℝ := 22

-- Case when there is no competition (train service is closed)
def optimal_bus_fare_without_train : ℝ := 26

-- Change in the number of passengers when the train service closes
def change_in_passengers : ℝ := 400

-- Theorems to prove
theorem bus_fare_with_train : optimal_bus_fare_with_train = 22 := sorry
theorem change_in_total_passengers : change_in_passengers = 400 := sorry

end bus_fare_with_train_change_in_total_passengers_l1405_140578


namespace peggy_dolls_ratio_l1405_140523

noncomputable def peggy_dolls_original := 6
noncomputable def peggy_dolls_from_grandmother := 30
noncomputable def peggy_dolls_total := 51

theorem peggy_dolls_ratio :
  ∃ x, peggy_dolls_original + peggy_dolls_from_grandmother + x = peggy_dolls_total ∧ x / peggy_dolls_from_grandmother = 1 / 2 :=
by {
  sorry
}

end peggy_dolls_ratio_l1405_140523


namespace total_pieces_of_gum_l1405_140533

def packages := 43
def pieces_per_package := 23
def extra_pieces := 8

theorem total_pieces_of_gum :
  (packages * pieces_per_package) + extra_pieces = 997 := sorry

end total_pieces_of_gum_l1405_140533
