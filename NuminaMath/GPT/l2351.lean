import Mathlib

namespace csc_neg_45_eq_neg_sqrt_2_l2351_235117

noncomputable def csc (θ : Real) : Real := 1 / Real.sin θ

theorem csc_neg_45_eq_neg_sqrt_2 :
  csc (-Real.pi / 4) = -Real.sqrt 2 := by
  sorry

end csc_neg_45_eq_neg_sqrt_2_l2351_235117


namespace determine_xy_l2351_235154

noncomputable section

open Real

def op_defined (ab xy : ℝ × ℝ) : ℝ × ℝ :=
  (ab.1 * xy.1 + ab.2 * xy.2, ab.1 * xy.2 + ab.2 * xy.1)

theorem determine_xy (x y : ℝ) :
  (∀ (a b : ℝ), op_defined (a, b) (x, y) = (a, b)) → (x = 1 ∧ y = 0) :=
by
  sorry

end determine_xy_l2351_235154


namespace solution_of_system_l2351_235153

theorem solution_of_system (x y : ℝ) (h1 : x + 2 * y = 8) (h2 : 2 * x + y = 1) : x + y = 3 :=
sorry

end solution_of_system_l2351_235153


namespace line_x_intercept_l2351_235169

theorem line_x_intercept {x1 y1 x2 y2 : ℝ} (h : (x1, y1) = (4, 6)) (h2 : (x2, y2) = (8, 2)) :
  ∃ x : ℝ, (y1 - y2) / (x1 - x2) * x + 6 - ((y1 - y2) / (x1 - x2)) * 4 = 0 ∧ x = 10 :=
by
  sorry

end line_x_intercept_l2351_235169


namespace like_terms_sum_three_l2351_235155

theorem like_terms_sum_three (m n : ℤ) (h1 : 2 * m = 4 - n) (h2 : m = n - 1) : m + n = 3 :=
sorry

end like_terms_sum_three_l2351_235155


namespace increase_in_daily_mess_expenses_l2351_235145

theorem increase_in_daily_mess_expenses (A X : ℝ)
  (h1 : 35 * A = 420)
  (h2 : 42 * (A - 1) = 420 + X) :
  X = 42 :=
by
  sorry

end increase_in_daily_mess_expenses_l2351_235145


namespace adult_ticket_cost_l2351_235105

-- Definitions based on given conditions.
def children_ticket_cost : ℝ := 7.5
def total_bill : ℝ := 138
def total_tickets : ℕ := 12
def additional_children_tickets : ℕ := 8

-- Proof statement: Prove the cost of each adult ticket.
theorem adult_ticket_cost (x : ℕ) (A : ℝ)
  (h1 : x + (x + additional_children_tickets) = total_tickets)
  (h2 : x * A + (x + additional_children_tickets) * children_ticket_cost = total_bill) :
  A = 31.50 :=
  sorry

end adult_ticket_cost_l2351_235105


namespace contradiction_proof_l2351_235149

theorem contradiction_proof (a b : ℝ) (h : a ≥ b) (h_pos : b > 0) (h_contr : a^2 < b^2) : false :=
by {
  sorry
}

end contradiction_proof_l2351_235149


namespace isabella_more_than_sam_l2351_235131

variable (I S G : ℕ)

def Giselle_money : G = 120 := by sorry
def Isabella_more_than_Giselle : I = G + 15 := by sorry
def total_donation : I + S + G = 345 := by sorry

theorem isabella_more_than_sam : I - S = 45 := by
sorry

end isabella_more_than_sam_l2351_235131


namespace days_with_equal_sun_tue_l2351_235124

theorem days_with_equal_sun_tue (days_in_month : ℕ) (weekdays : ℕ) (d1 d2 : ℕ) (h1 : days_in_month = 30)
  (h2 : weekdays = 7) (h3 : d1 = 4) (h4 : d2 = 2) :
  ∃ count, count = 3 := by
  sorry

end days_with_equal_sun_tue_l2351_235124


namespace correct_calculation_l2351_235116

theorem correct_calculation (x : ℝ) : 
(x + x = 2 * x) ∧
(x * x = x^2) ∧
(2 * x * x^2 = 2 * x^3) ∧
(x^6 / x^3 = x^3) →
(2 * x * x^2 = 2 * x^3) := 
by
  intro h
  exact h.2.2.1

end correct_calculation_l2351_235116


namespace binary_addition_is_correct_l2351_235141

theorem binary_addition_is_correct :
  (0b101101 + 0b1011 + 0b11001 + 0b1110101 + 0b1111) = 0b10010001 :=
by sorry

end binary_addition_is_correct_l2351_235141


namespace total_handshakes_l2351_235114

-- There are 5 members on each of the two basketball teams.
def teamMembers : Nat := 5

-- There are 2 referees.
def referees : Nat := 2

-- Each player from one team shakes hands with each player from the other team.
def handshakesBetweenTeams : Nat := teamMembers * teamMembers

-- Each player shakes hands with each referee.
def totalPlayers : Nat := 2 * teamMembers
def handshakesWithReferees : Nat := totalPlayers * referees

-- Prove that the total number of handshakes is 45.
theorem total_handshakes : handshakesBetweenTeams + handshakesWithReferees = 45 := by
  -- Total handshakes is the sum of handshakes between teams and handshakes with referees.
  sorry

end total_handshakes_l2351_235114


namespace dot_product_correct_l2351_235175

-- Define the vectors as given conditions
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (1, -2)

-- State the theorem to prove the dot product
theorem dot_product_correct : a.1 * b.1 + a.2 * b.2 = -4 := by
  -- Proof steps go here
  sorry

end dot_product_correct_l2351_235175


namespace range_of_a_l2351_235193

def f (a x : ℝ) : ℝ := x^2 - a*x + a + 3
def g (a x : ℝ) : ℝ := x - a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬(f a x < 0 ∧ g a x < 0)) ↔ a ∈ Set.Icc (-3 : ℝ) 6 :=
sorry

end range_of_a_l2351_235193


namespace theo_cookies_eaten_in_9_months_l2351_235172

-- Define the basic variable values as per the conditions
def cookiesPerTime : Nat := 25
def timesPerDay : Nat := 5
def daysPerMonth : Nat := 27
def numMonths : Nat := 9

-- Define the total number of cookies Theo can eat in 9 months
def totalCookiesIn9Months : Nat :=
  cookiesPerTime * timesPerDay * daysPerMonth * numMonths

-- The theorem stating the answer
theorem theo_cookies_eaten_in_9_months :
  totalCookiesIn9Months = 30375 := by
  -- Proof will go here
  sorry

end theo_cookies_eaten_in_9_months_l2351_235172


namespace cars_people_count_l2351_235133

-- Define the problem conditions
def cars_people_conditions (x y : ℕ) : Prop :=
  y = 3 * (x - 2) ∧ y = 2 * x + 9

-- Define the theorem stating that there exist numbers of cars and people that satisfy the conditions
theorem cars_people_count (x y : ℕ) : cars_people_conditions x y ↔ (y = 3 * (x - 2) ∧ y = 2 * x + 9) := by
  -- skip the proof
  sorry

end cars_people_count_l2351_235133


namespace problem_1_problem_2_l2351_235195

universe u

/-- Assume the universal set U is the set of real numbers -/
def U : Set ℝ := Set.univ

/-- Define set A -/
def A : Set ℝ := {x : ℝ | x ≥ 1}

/-- Define set B -/
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

/-- Prove the intersection of A and B -/
theorem problem_1 : (A ∩ B) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

/-- Prove the complement of the union of A and B -/
theorem problem_2 : (U \ (A ∪ B)) = {x : ℝ | x < -1} :=
sorry

end problem_1_problem_2_l2351_235195


namespace proof_problem_l2351_235139

variables (a b : ℝ)
variable (h : a ≠ b)
variable (h1 : a * Real.exp a = b * Real.exp b)
variable (p : Prop := Real.log a + a = Real.log b + b)
variable (q : Prop := (a + 1) * (b + 1) < 0)

theorem proof_problem : p ∨ q :=
sorry

end proof_problem_l2351_235139


namespace quotient_of_powers_l2351_235181

theorem quotient_of_powers:
  (50 : ℕ) = 2 * 5^2 →
  (25 : ℕ) = 5^2 →
  (50^50 / 25^25 : ℕ) = 100^25 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end quotient_of_powers_l2351_235181


namespace select_16_genuine_coins_l2351_235166

theorem select_16_genuine_coins (coins : Finset ℕ) (h_coins_count : coins.card = 40) 
  (counterfeit : Finset ℕ) (h_counterfeit_count : counterfeit.card = 3)
  (h_counterfeit_lighter : ∀ c ∈ counterfeit, ∀ g ∈ (coins \ counterfeit), c < g) :
  ∃ genuine : Finset ℕ, genuine.card = 16 ∧ 
    (∀ h1 h2 h3 : Finset ℕ, h1.card = 20 → h2.card = 10 → h3.card = 8 →
      ((h1 ⊆ coins ∧ h2 ⊆ h1 ∧ h3 ⊆ (h1 \ counterfeit)) ∨
       (h1 ⊆ coins ∧ h2 ⊆ (h1 \ counterfeit) ∧ h3 ⊆ (h2 \ counterfeit))) →
      genuine ⊆ coins \ counterfeit) :=
sorry

end select_16_genuine_coins_l2351_235166


namespace min_value_of_expression_l2351_235146

theorem min_value_of_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x : ℝ, x = 6 * (12 : ℝ)^(1/6) ∧
  (∀ a b c, 0 < a ∧ 0 < b ∧ 0 < c → 
  x ≤ (a + 2 * b) / c + (2 * a + c) / b + (b + 3 * c) / a) :=
sorry

end min_value_of_expression_l2351_235146


namespace parcel_cost_l2351_235110

theorem parcel_cost (P : ℤ) (hP : P ≥ 1) : 
  (P ≤ 5 → C = 15 + 4 * (P - 1)) ∧ (P > 5 → C = 15 + 4 * (P - 1) - 10) :=
sorry

end parcel_cost_l2351_235110


namespace megatek_manufacturing_percentage_l2351_235142

theorem megatek_manufacturing_percentage (total_degrees sector_degrees : ℝ)
    (h_circle: total_degrees = 360)
    (h_sector: sector_degrees = 252) :
    (sector_degrees / total_degrees) * 100 = 70 :=
by
  sorry

end megatek_manufacturing_percentage_l2351_235142


namespace number_of_boys_l2351_235165

-- Definitions of the conditions
def total_members (B G : ℕ) : Prop := B + G = 26
def meeting_attendance (B G : ℕ) : Prop := (1 / 2 : ℚ) * G + B = 16

-- Theorem statement
theorem number_of_boys (B G : ℕ) (h1 : total_members B G) (h2 : meeting_attendance B G) : B = 6 := by
  sorry

end number_of_boys_l2351_235165


namespace license_plate_increase_l2351_235113

theorem license_plate_increase :
  let old_license_plates := 26^2 * 10^3
  let new_license_plates := 26^2 * 10^4
  new_license_plates / old_license_plates = 10 :=
by
  sorry

end license_plate_increase_l2351_235113


namespace relationship_between_k_and_c_l2351_235100

-- Define the functions and given conditions
def y1 (x : ℝ) (c : ℝ) : ℝ := x^2 + 2*x + c
def y2 (x : ℝ) (k : ℝ) : ℝ := k*x + 2

-- Define the vertex of y1
def vertex_y1 (c : ℝ) : ℝ × ℝ := (-1, c - 1)

-- State the main theorem
theorem relationship_between_k_and_c (k c : ℝ) (hk : k ≠ 0) :
  y2 (vertex_y1 c).1 k = (vertex_y1 c).2 → c + k = 3 :=
by
  sorry

end relationship_between_k_and_c_l2351_235100


namespace jake_watched_friday_l2351_235197

theorem jake_watched_friday
  (monday_hours : ℕ)
  (tuesday_hours : ℕ)
  (wednesday_hours : ℕ)
  (thursday_hours : ℕ)
  (total_hours : ℕ)
  (day_hours : ℕ := 24) :
  monday_hours = (day_hours / 2) →
  tuesday_hours = 4 →
  wednesday_hours = (day_hours / 4) →
  thursday_hours = ((monday_hours + tuesday_hours + wednesday_hours) / 2) →
  total_hours = 52 →
  (total_hours - (monday_hours + tuesday_hours + wednesday_hours + thursday_hours)) = 19 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end jake_watched_friday_l2351_235197


namespace hours_felt_good_l2351_235106

variable (x : ℝ)

theorem hours_felt_good (h1 : 15 * x + 10 * (8 - x) = 100) : x == 4 := 
by
  sorry

end hours_felt_good_l2351_235106


namespace congruent_triangles_count_l2351_235136

open Set

variables (g l : Line) (A B C : Point)

def number_of_congruent_triangles (g l : Line) (A B C : Point) : ℕ :=
  16

theorem congruent_triangles_count (g l : Line) (A B C : Point) :
  number_of_congruent_triangles g l A B C = 16 :=
sorry

end congruent_triangles_count_l2351_235136


namespace union_sets_l2351_235101

open Set

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5, 6}

theorem union_sets : (A ∪ B) = {1, 2, 3, 4, 5, 6} :=
by
  sorry

end union_sets_l2351_235101


namespace f_bounded_by_inverse_l2351_235122

theorem f_bounded_by_inverse (f : ℕ → ℝ) (h_pos : ∀ n, 0 < f n) (h_rec : ∀ n, (f n)^2 ≤ f n - f (n + 1)) :
  ∀ n, f n < 1 / (n + 1) :=
by
  sorry

end f_bounded_by_inverse_l2351_235122


namespace range_of_a_l2351_235140

theorem range_of_a (a : ℝ) : 
  (∀ x : ℕ, (1 ≤ x ∧ x ≤ 4) → ax + 4 ≥ 0) → (-1 ≤ a ∧ a < -4/5) :=
by
  sorry

end range_of_a_l2351_235140


namespace exists_a_b_k_l2351_235109

theorem exists_a_b_k (m : ℕ) (hm : 0 < m) : 
  ∃ a b k : ℤ, 
    (a % 2 = 1) ∧ 
    (b % 2 = 1) ∧ 
    (0 ≤ k) ∧ 
    (2 * m = a^19 + b^99 + k * 2^1999) :=
sorry

end exists_a_b_k_l2351_235109


namespace geography_book_price_l2351_235138

open Real

-- Define the problem parameters
def num_english_books : ℕ := 35
def num_geography_books : ℕ := 35
def cost_english : ℝ := 7.50
def total_cost : ℝ := 630.00

-- Define the unknown we need to prove
def cost_geography : ℝ := 10.50

theorem geography_book_price :
  num_english_books * cost_english + num_geography_books * cost_geography = total_cost :=
by
  -- No need to include the proof steps
  sorry

end geography_book_price_l2351_235138


namespace solution_to_abs_eq_l2351_235121

theorem solution_to_abs_eq :
  ∀ x : ℤ, abs ((-5) + x) = 11 → (x = 16 ∨ x = -6) :=
by sorry

end solution_to_abs_eq_l2351_235121


namespace subsets_bound_l2351_235137

variable {n : ℕ} (S : Finset (Fin n)) (m : ℕ) (A : ℕ → Finset (Fin n))

theorem subsets_bound {n : ℕ} (hn : n ≥ 2) (hA : ∀ i, 1 ≤ i ∧ i ≤ m → (A i).card ≥ 2)
  (h_inter : ∀ i j k, 1 ≤ i ∧ i ≤ m → 1 ≤ j ∧ j ≤ m → 1 ≤ k ∧ k ≤ m →
    (A i) ∩ (A j) ≠ ∅ ∧ (A i) ∩ (A k) ≠ ∅ ∧ (A j) ∩ (A k) ≠ ∅ → (A i) ∩ (A j) ∩ (A k) ≠ ∅) :
  m ≤ 2 ^ (n - 1) - 1 := 
sorry

end subsets_bound_l2351_235137


namespace solution_correct_l2351_235147

def mixed_number_to_fraction (a b c : ℕ) : ℚ :=
  (a * b + c) / b

def percentage_to_decimal (fraction : ℚ) : ℚ :=
  fraction / 100

def evaluate_expression : ℚ :=
  let part1 := 63 * 5 + 4
  let part2 := 48 * 7 + 3
  let part3 := 17 * 3 + 2
  let term1 := (mixed_number_to_fraction 63 5 4) * 3150
  let term2 := (mixed_number_to_fraction 48 7 3) * 2800
  let term3 := (mixed_number_to_fraction 17 3 2) * 945 / 2
  term1 - term2 + term3

theorem solution_correct :
  (percentage_to_decimal (mixed_number_to_fraction 63 5 4) * 3150) -
  (percentage_to_decimal (mixed_number_to_fraction 48 7 3) * 2800) +
  (percentage_to_decimal (mixed_number_to_fraction 17 3 2) * 945 / 2) = 737.175 := 
sorry

end solution_correct_l2351_235147


namespace original_number_divisibility_l2351_235134

theorem original_number_divisibility (N : ℤ) : (∃ k : ℤ, N = 9 * k + 3) ↔ (∃ m : ℤ, (N + 3) = 9 * m) := sorry

end original_number_divisibility_l2351_235134


namespace jerry_can_throw_things_l2351_235103

def points_for_interrupting : ℕ := 5
def points_for_insulting : ℕ := 10
def points_for_throwing : ℕ := 25
def office_points_threshold : ℕ := 100
def interruptions : ℕ := 2
def insults : ℕ := 4

theorem jerry_can_throw_things : 
  (office_points_threshold - (points_for_interrupting * interruptions + points_for_insulting * insults)) / points_for_throwing = 2 :=
by 
  sorry

end jerry_can_throw_things_l2351_235103


namespace all_numbers_equal_l2351_235199

theorem all_numbers_equal
  (n : ℕ)
  (h n_eq_20 : n = 20)
  (a : ℕ → ℝ)
  (h_avg : ∀ i : ℕ, i < n → a i = (a ((i+n-1) % n) + a ((i+1) % n)) / 2) :
  ∀ i j : ℕ, i < n → j < n → a i = a j :=
by {
  -- Proof steps go here.
  sorry
}

end all_numbers_equal_l2351_235199


namespace unique_friends_count_l2351_235162

-- Definitions from conditions
def M : ℕ := 10
def P : ℕ := 20
def G : ℕ := 5
def M_P : ℕ := 4
def M_G : ℕ := 2
def P_G : ℕ := 0
def M_P_G : ℕ := 2

-- Theorem we need to prove
theorem unique_friends_count : (M + P + G - M_P - M_G - P_G + M_P_G) = 31 := by
  sorry

end unique_friends_count_l2351_235162


namespace find_n_l2351_235176

theorem find_n : ∀ n : ℚ, (1 / (n + 2) + 2 / (n + 2) + 3 * n / (n + 2) = 5) → (n = -7 / 2) := by
  intro n h
  sorry

end find_n_l2351_235176


namespace calculate_expression_l2351_235156

theorem calculate_expression :
  (56 * 0.57 * 0.85) / (2.8 * 19 * 1.7) = 0.3 :=
by
  sorry

end calculate_expression_l2351_235156


namespace value_of_x_minus_y_l2351_235187

theorem value_of_x_minus_y (x y : ℚ) 
    (h₁ : 3 * x - 5 * y = 5) 
    (h₂ : x / (x + y) = 5 / 7) : x - y = 3 := by
  sorry

end value_of_x_minus_y_l2351_235187


namespace alex_growth_rate_l2351_235174

noncomputable def growth_rate_per_hour_hanging_upside_down
  (current_height : ℝ)
  (required_height : ℝ)
  (normal_growth_per_month : ℝ)
  (hanging_hours_per_month : ℝ)
  (answer : ℝ) : Prop :=
  current_height + 12 * normal_growth_per_month + 12 * hanging_hours_per_month * answer = required_height

theorem alex_growth_rate 
  (current_height : ℝ) 
  (required_height : ℝ)
  (normal_growth_per_month : ℝ)
  (hanging_hours_per_month : ℝ)
  (answer : ℝ) :
  current_height = 48 → 
  required_height = 54 → 
  normal_growth_per_month = 1/3 → 
  hanging_hours_per_month = 2 → 
  growth_rate_per_hour_hanging_upside_down current_height required_height normal_growth_per_month hanging_hours_per_month answer ↔ answer = 1/12 :=
by sorry

end alex_growth_rate_l2351_235174


namespace gumballs_per_pair_of_earrings_l2351_235170

theorem gumballs_per_pair_of_earrings : 
  let day1_earrings := 3
  let day2_earrings := 2 * day1_earrings
  let day3_earrings := day2_earrings - 1
  let total_earrings := day1_earrings + day2_earrings + day3_earrings
  let days := 42
  let gumballs_per_day := 3
  let total_gumballs := days * gumballs_per_day
  (total_gumballs / total_earrings) = 9 :=
by
  -- Definitions
  let day1_earrings := 3
  let day2_earrings := 2 * day1_earrings
  let day3_earrings := day2_earrings - 1
  let total_earrings := day1_earrings + day2_earrings + day3_earrings
  let days := 42
  let gumballs_per_day := 3
  let total_gumballs := days * gumballs_per_day
  -- Theorem statement
  sorry

end gumballs_per_pair_of_earrings_l2351_235170


namespace max_gcd_sequence_l2351_235120

noncomputable def a (n : ℕ) : ℕ := n^3 + 4
noncomputable def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_gcd_sequence : (∀ n : ℕ, 0 < n → d n ≤ 433) ∧ (∃ n : ℕ, 0 < n ∧ d n = 433) :=
by sorry

end max_gcd_sequence_l2351_235120


namespace closest_to_zero_is_13_l2351_235159

noncomputable def a (n : ℕ) : ℤ := 88 - 7 * n

theorem closest_to_zero_is_13 : ∀ (n : ℕ), 1 ≤ n → 81 + (n - 1) * (-7) = a n →
  (∀ m : ℕ, (m : ℤ) ≤ (88 : ℤ) / 7 → abs (a m) > abs (a 13)) :=
  sorry

end closest_to_zero_is_13_l2351_235159


namespace triangle_side_relation_triangle_perimeter_l2351_235111

theorem triangle_side_relation (a b c : ℝ) (A B C : ℝ)
  (h1 : a / (Real.sin A) = b / (Real.sin B)) (h2 : a / (Real.sin A) = c / (Real.sin C))
  (h3 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) :
  2 * a ^ 2 = b ^ 2 + c ^ 2 := sorry

theorem triangle_perimeter (a b c : ℝ) (A : ℝ) (hcosA : Real.cos A = 25 / 31)
  (h1 : a / (Real.sin A) = b / (Real.sin B)) (h2 : a / (Real.sin A) = c / (Real.sin C))
  (h3 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) (ha : a = 5) :
  a + b + c = 14 := sorry

end triangle_side_relation_triangle_perimeter_l2351_235111


namespace math_problem_l2351_235157

theorem math_problem : (300 + 5 * 8) / (2^3) = 42.5 := by
  sorry

end math_problem_l2351_235157


namespace required_hemispherical_containers_l2351_235163

noncomputable def initial_volume : ℝ := 10940
noncomputable def initial_temperature : ℝ := 20
noncomputable def final_temperature : ℝ := 25
noncomputable def expansion_coefficient : ℝ := 0.002
noncomputable def container_volume : ℝ := 4
noncomputable def usable_capacity : ℝ := 0.8

noncomputable def volume_expansion : ℝ := initial_volume * (final_temperature - initial_temperature) * expansion_coefficient
noncomputable def final_volume : ℝ := initial_volume + volume_expansion
noncomputable def usable_volume_per_container : ℝ := container_volume * usable_capacity
noncomputable def number_of_containers_needed : ℝ := final_volume / usable_volume_per_container

theorem required_hemispherical_containers : ⌈number_of_containers_needed⌉ = 3453 :=
by 
  sorry

end required_hemispherical_containers_l2351_235163


namespace tickets_system_l2351_235190

variable (x y : ℕ)

theorem tickets_system (h1 : x + y = 20) (h2 : 2800 * x + 6400 * y = 74000) :
  (x + y = 20) ∧ (2800 * x + 6400 * y = 74000) :=
by {
  exact (And.intro h1 h2)
}

end tickets_system_l2351_235190


namespace maria_uses_666_blocks_l2351_235164

theorem maria_uses_666_blocks :
  let original_volume := 15 * 12 * 7
  let interior_length := 15 - 2 * 1.5
  let interior_width := 12 - 2 * 1.5
  let interior_height := 7 - 1.5
  let interior_volume := interior_length * interior_width * interior_height
  let blocks_volume := original_volume - interior_volume
  blocks_volume = 666 :=
by
  sorry

end maria_uses_666_blocks_l2351_235164


namespace certain_number_divided_by_two_l2351_235108

theorem certain_number_divided_by_two (x : ℝ) (h : x / 2 + x + 2 = 62) : x = 40 :=
sorry

end certain_number_divided_by_two_l2351_235108


namespace sequence_match_l2351_235186

-- Define the sequence sum S_n
def S_n (n : ℕ) : ℕ := 2^(n + 1) - 1

-- Define the sequence a_n based on the problem statement
def a_n (n : ℕ) : ℕ :=
  if n = 1 then 3
  else 2^n

-- The theorem stating that sequence a_n satisfies the given sum condition S_n
theorem sequence_match (n : ℕ) : a_n n = if n = 1 then 3 else 2^n :=
  sorry

end sequence_match_l2351_235186


namespace simplify_expression_l2351_235160

theorem simplify_expression {x a : ℝ} (h1 : x > a) (h2 : x ≠ 0) (h3 : a ≠ 0) :
  (x * (x^2 - a^2)⁻¹ + 1) / (a * (x - a)⁻¹ + (x - a)^(1 / 2))
  / ((a^2 * (x + a)^(1 / 2)) / (x - (x^2 - a^2)^(1 / 2)) + 1 / (x^2 - a * x))
  = 2 / (x^2 - a^2) :=
by sorry

end simplify_expression_l2351_235160


namespace area_of_shaded_rectangle_l2351_235102

-- Definition of side length of the squares
def side_length : ℕ := 12

-- Definition of the dimensions of the overlapped rectangle
def rectangle_length : ℕ := 20
def rectangle_width : ℕ := side_length

-- Theorem stating the area of the shaded rectangle PBCS
theorem area_of_shaded_rectangle
  (squares_identical : ∀ (a b c d p q r s : ℕ),
    a = side_length → b = side_length →
    p = side_length → q = side_length →
    rectangle_width * (rectangle_length - side_length) = 48) :
  rectangle_width * (rectangle_length - side_length) = 48 :=
by sorry -- Proof omitted

end area_of_shaded_rectangle_l2351_235102


namespace arithmetic_sequence_geometric_l2351_235125

noncomputable def sequence_arith_to_geom (a1 d : ℤ) (h_d : d ≠ 0) (n : ℕ) : ℤ :=
a1 + (n - 1) * d

theorem arithmetic_sequence_geometric (a1 d : ℤ) (h_d : d ≠ 0) (n : ℕ) :
  (n = 16)
    ↔ (((a1 + 3 * d) / (a1 + 2 * d) = (a1 + 6 * d) / (a1 + 3 * d)) ∧ 
        ((a1 + 6 * d) / (a1 + 3 * d) = (a1 + (n - 1) * d) / (a1 + 6 * d))) :=
by
  sorry

end arithmetic_sequence_geometric_l2351_235125


namespace watch_cost_price_l2351_235118

open Real

theorem watch_cost_price (CP SP1 SP2 : ℝ)
    (h1 : SP1 = CP * 0.85)
    (h2 : SP2 = CP * 1.10)
    (h3 : SP2 = SP1 + 450) : CP = 1800 :=
by
  sorry

end watch_cost_price_l2351_235118


namespace one_of_18_consecutive_is_divisible_l2351_235130

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define what it means for one number to be divisible by another
def divisible (a b : ℕ) : Prop :=
  b ≠ 0 ∧ a % b = 0

-- The main theorem
theorem one_of_18_consecutive_is_divisible : 
  ∀ (n : ℕ), 100 ≤ n ∧ n + 17 ≤ 999 → ∃ (k : ℕ), n ≤ k ∧ k ≤ (n + 17) ∧ divisible k (sum_of_digits k) :=
by
  intros n h
  sorry

end one_of_18_consecutive_is_divisible_l2351_235130


namespace find_m_l2351_235173

theorem find_m (m : ℝ) (a b : ℝ × ℝ) (k : ℝ) (ha : a = (1, 1)) (hb : b = (m, 2)) 
  (h_parallel : 2 • a + b = k • a) : m = 2 :=
sorry

end find_m_l2351_235173


namespace range_of_a_l2351_235115

noncomputable def problem (x y z : ℝ) (a : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ (x + y + z = 1) ∧ 
  (a / (x * y * z) = 1/x + 1/y + 1/z - 2) 

theorem range_of_a (x y z a : ℝ) (h : problem x y z a) : 
  0 < a ∧ a ≤ 7/27 :=
sorry

end range_of_a_l2351_235115


namespace find_number_l2351_235126

axiom condition_one (x y : ℕ) : 10 * x + y = 3 * (x + y) + 7
axiom condition_two (x y : ℕ) : x^2 + y^2 - x * y = 10 * x + y

theorem find_number : 
  ∃ (x y : ℕ), (10 * x + y = 37) → (10 * x + y = 3 * (x + y) + 7 ∧ x^2 + y^2 - x * y = 10 * x + y) := 
by 
  sorry

end find_number_l2351_235126


namespace line_intersects_plane_at_angle_l2351_235178

def direction_vector : ℝ × ℝ × ℝ := (1, -1, 2)
def normal_vector : ℝ × ℝ × ℝ := (-2, 2, -4)

theorem line_intersects_plane_at_angle :
  let a := direction_vector
  let u := normal_vector
  a ≠ (0, 0, 0) → u ≠ (0, 0, 0) →
  ∃ θ : ℝ, 0 < θ ∧ θ < π :=
by
  sorry

end line_intersects_plane_at_angle_l2351_235178


namespace no_positive_reals_satisfy_equations_l2351_235198

theorem no_positive_reals_satisfy_equations :
  ¬ ∃ (a b c d : ℝ), (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (0 < d) ∧
  (a / b + b / c + c / d + d / a = 6) ∧ (b / a + c / b + d / c + a / d = 32) :=
by sorry

end no_positive_reals_satisfy_equations_l2351_235198


namespace sum_of_squares_diagonals_of_rhombus_l2351_235191

theorem sum_of_squares_diagonals_of_rhombus (d1 d2 : ℝ) (h : (d1 / 2)^2 + (d2 / 2)^2 = 4) : d1^2 + d2^2 = 16 :=
sorry

end sum_of_squares_diagonals_of_rhombus_l2351_235191


namespace gasoline_tank_capacity_l2351_235152

theorem gasoline_tank_capacity
  (y : ℝ)
  (h_initial: y * (5 / 6) - y * (1 / 3) = 20) :
  y = 40 :=
sorry

end gasoline_tank_capacity_l2351_235152


namespace fibonacci_arithmetic_sequence_l2351_235128

def fibonacci : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_arithmetic_sequence (a b c n : ℕ) 
  (h1 : fibonacci 1 = 1)
  (h2 : fibonacci 2 = 1)
  (h3 : ∀ n ≥ 3, fibonacci n = fibonacci (n - 1) + fibonacci (n - 2))
  (h4 : a + b + c = 2500)
  (h5 : (a, b, c) = (n, n + 3, n + 5)) :
  a = 831 := 
sorry

end fibonacci_arithmetic_sequence_l2351_235128


namespace simplify_and_evaluate_expression_l2351_235104

theorem simplify_and_evaluate_expression (x y : ℝ) (h1 : x = 1/2) (h2 : y = -2) :
  ((x + 2 * y) ^ 2 - (x + y) * (x - y)) / (2 * y) = -4 := by
  sorry

end simplify_and_evaluate_expression_l2351_235104


namespace range_of_t_l2351_235119

theorem range_of_t (a b : ℝ) 
  (h1 : a^2 + a * b + b^2 = 1) 
  (h2 : ∃ t : ℝ, t = a * b - a^2 - b^2) : 
  ∀ t, t = a * b - a^2 - b^2 → -3 ≤ t ∧ t ≤ -1/3 :=
by sorry

end range_of_t_l2351_235119


namespace solution_set_of_inequality_l2351_235188

variable {a x : ℝ}

theorem solution_set_of_inequality (h : 2 * a + 1 < 0) : 
  {x : ℝ | x^2 - 4 * a * x - 5 * a^2 > 0} = {x | x < 5 * a ∨ x > -a} := by
  sorry

end solution_set_of_inequality_l2351_235188


namespace focus_of_parabola_x_squared_eq_neg_4_y_l2351_235194

theorem focus_of_parabola_x_squared_eq_neg_4_y:
  (∃ F : ℝ × ℝ, (F = (0, -1)) ∧ (∀ x y : ℝ, x^2 = -4 * y → F = (0, y + 1))) :=
sorry

end focus_of_parabola_x_squared_eq_neg_4_y_l2351_235194


namespace rational_xyz_squared_l2351_235196

theorem rational_xyz_squared
  (x y z : ℝ)
  (hx : ∃ r1 : ℚ, x + y * z = r1)
  (hy : ∃ r2 : ℚ, y + z * x = r2)
  (hz : ∃ r3 : ℚ, z + x * y = r3)
  (hxy : x^2 + y^2 = 1) :
  ∃ r4 : ℚ, x * y * z^2 = r4 := 
sorry

end rational_xyz_squared_l2351_235196


namespace freight_capacity_equation_l2351_235148

theorem freight_capacity_equation
  (x : ℝ)
  (h1 : ∀ (capacity_large capacity_small : ℝ), capacity_large = capacity_small + 4)
  (h2 : ∀ (n_large n_small : ℕ), (n_large : ℝ) = 80 / (x + 4) ∧ (n_small : ℝ) = 60 / x → n_large = n_small) :
  (80 / (x + 4)) = (60 / x) :=
by
  sorry

end freight_capacity_equation_l2351_235148


namespace max_xyz_l2351_235112

theorem max_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 2) 
(h5 : x^2 + y^2 + z^2 = x * z + y * z + x * y) : xyz ≤ (8 / 27) :=
sorry

end max_xyz_l2351_235112


namespace find_y_value_l2351_235135

theorem find_y_value :
  (∃ m b : ℝ, (∀ x y : ℝ, (x = 2 ∧ y = 5) ∨ (x = 6 ∧ y = 17) ∨ (x = 10 ∧ y = 29) → y = m * x + b))
  → (∃ y : ℝ, x = 40 → y = 119) := by
  sorry

end find_y_value_l2351_235135


namespace complex_number_simplification_l2351_235123

theorem complex_number_simplification (i : ℂ) (h : i^2 = -1) : 
  (↑(1 : ℂ) - i) / (↑(1 : ℂ) + i) ^ 2017 = -i :=
sorry

end complex_number_simplification_l2351_235123


namespace find_common_ratio_l2351_235144

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, S n = a 1 * (1 - q ^ n) / (1 - q)

variables (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 q : ℝ)

noncomputable def a_5_condition : Prop :=
  a 5 = 2 * S 4 + 3

noncomputable def a_6_condition : Prop :=
  a 6 = 2 * S 5 + 3

theorem find_common_ratio (h1 : a_5_condition a S) (h2 : a_6_condition a S)
  (hg : geometric_sequence a q) (hs : sum_of_first_n_terms a S q) :
  q = 3 :=
sorry

end find_common_ratio_l2351_235144


namespace sacks_required_in_4_weeks_l2351_235185

-- Definitions for the weekly requirements of each bakery
def weekly_sacks_bakery1 : Nat := 2
def weekly_sacks_bakery2 : Nat := 4
def weekly_sacks_bakery3 : Nat := 12

-- Total weeks considered
def weeks : Nat := 4

-- Calculating the total sacks needed for all bakeries over the given weeks
def total_sacks_needed : Nat :=
  (weekly_sacks_bakery1 * weeks) +
  (weekly_sacks_bakery2 * weeks) +
  (weekly_sacks_bakery3 * weeks)

-- The theorem to be proven
theorem sacks_required_in_4_weeks :
  total_sacks_needed = 72 :=
by
  sorry

end sacks_required_in_4_weeks_l2351_235185


namespace pentagon_coloring_valid_l2351_235180

-- Define the colors
inductive Color
| Red
| Blue

-- Define the vertices as a type
inductive Vertex
| A | B | C | D | E

open Vertex Color

-- Define an edge as a pair of vertices
def Edge := Vertex × Vertex

-- Define the coloring function
def color : Edge → Color := sorry

-- Define the pentagon
def pentagon_edges : List Edge :=
  [(A, B), (B, C), (C, D), (D, E), (E, A), (A, C), (A, D), (A, E), (B, D), (B, E), (C, E)]

-- Define the condition for a valid triangle coloring
def valid_triangle_coloring (e1 e2 e3 : Edge) : Prop :=
  (color e1 = Red ∧ (color e2 = Blue ∨ color e3 = Blue)) ∨
  (color e2 = Red ∧ (color e1 = Blue ∨ color e3 = Blue)) ∨
  (color e3 = Red ∧ (color e1 = Blue ∨ color e2 = Blue))

-- Define the condition for all triangles formed by the vertices of the pentagon
def all_triangles_valid : Prop :=
  ∀ v1 v2 v3 : Vertex,
    v1 ≠ v2 → v2 ≠ v3 → v1 ≠ v3 →
    valid_triangle_coloring (v1, v2) (v2, v3) (v1, v3)

-- Statement: Prove that there are 12 valid ways to color the pentagon
theorem pentagon_coloring_valid : (∃ (coloring : Edge → Color), all_triangles_valid) :=
  sorry

end pentagon_coloring_valid_l2351_235180


namespace ratio_of_surface_areas_of_spheres_l2351_235167

theorem ratio_of_surface_areas_of_spheres (r1 r2 : ℝ) (h : r1 / r2 = 1 / 3) : 
  (4 * Real.pi * r1^2) / (4 * Real.pi * r2^2) = 1 / 9 := by
  sorry

end ratio_of_surface_areas_of_spheres_l2351_235167


namespace arrangement_count_l2351_235107

def no_adjacent_students_arrangements (teachers students : ℕ) : ℕ :=
  if teachers = 3 ∧ students = 3 then 144 else 0

theorem arrangement_count :
  no_adjacent_students_arrangements 3 3 = 144 :=
by
  sorry

end arrangement_count_l2351_235107


namespace shape_is_cylinder_l2351_235127

def is_cylinder (c : ℝ) (r θ z : ℝ) : Prop :=
  c > 0 ∧ r = c ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ True

theorem shape_is_cylinder (c : ℝ) (r θ z : ℝ) (h : c > 0) :
  is_cylinder c r θ z :=
by
  -- Proof is omitted
  sorry

end shape_is_cylinder_l2351_235127


namespace remainder_not_power_of_4_l2351_235168

theorem remainder_not_power_of_4 : ∃ n : ℕ, n ≥ 2 ∧ ¬ (∃ k : ℕ, (2^2^n) % (2^n - 1) = 4^k) := sorry

end remainder_not_power_of_4_l2351_235168


namespace trinomials_real_roots_inequality_l2351_235161

theorem trinomials_real_roots_inequality :
  (∃ (p q : ℤ), 1 ≤ p ∧ p ≤ 1997 ∧ 1 ≤ q ∧ q ≤ 1997 ∧ 
   ¬ (∃ m n : ℤ, (1 ≤ m ∧ m ≤ 1997) ∧ (1 ≤ n ∧ n ≤ 1997) ∧ (m + n = p) ∧ (m * n = q))) >
  (∃ (p q : ℤ), 1 ≤ p ∧ p ≤ 1997 ∧ 1 ≤ q ∧ q ≤ 1997 ∧ 
   ∃ m n : ℤ, (1 ≤ m ∧ m ≤ 1997) ∧ (1 ≤ n ∧ n ≤ 1997) ∧ (m + n = p) ∧ (m * n = q)) :=
sorry

end trinomials_real_roots_inequality_l2351_235161


namespace vans_capacity_l2351_235192

def students : ℕ := 33
def adults : ℕ := 9
def vans : ℕ := 6

def total_people : ℕ := students + adults
def people_per_van : ℕ := total_people / vans

theorem vans_capacity : people_per_van = 7 := by
  sorry

end vans_capacity_l2351_235192


namespace base_8_subtraction_l2351_235182

theorem base_8_subtraction : 
  let x := 0o1234   -- 1234 in base 8
  let y := 0o765    -- 765 in base 8
  let result := 0o225 -- 225 in base 8
  x - y = result := by sorry

end base_8_subtraction_l2351_235182


namespace frac_calc_l2351_235171

theorem frac_calc : (2 / 9) * (5 / 11) + 1 / 3 = 43 / 99 :=
by sorry

end frac_calc_l2351_235171


namespace length_of_PQ_l2351_235143

-- Definitions for the problem conditions
variable (XY UV PQ : ℝ)
variable (hXY_fixed : XY = 120)
variable (hUV_fixed : UV = 90)
variable (hParallel : XY = UV ∧ UV = PQ) -- Ensures XY || UV || PQ

-- The statement to prove
theorem length_of_PQ : PQ = 360 / 7 := by
  -- Definitions for similarity ratios and solving steps can be assumed here
  sorry

end length_of_PQ_l2351_235143


namespace total_area_of_rectangles_l2351_235129

/-- The combined area of two adjacent rectangular regions given their conditions -/
theorem total_area_of_rectangles (u v w z : ℝ) 
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) (hz : w < z) : 
  (u + v) * z = (u + v) * w + (u + v) * (z - w) :=
by
  sorry

end total_area_of_rectangles_l2351_235129


namespace janet_earnings_per_hour_l2351_235189

theorem janet_earnings_per_hour :
  let P := 0.25
  let T := 10
  3600 / T * P = 90 :=
by
  let P := 0.25
  let T := 10
  sorry

end janet_earnings_per_hour_l2351_235189


namespace pieces_present_l2351_235179

-- Define the pieces and their counts in a standard chess set
def total_pieces := 32
def missing_pieces := 12
def missing_kings := 1
def missing_queens := 2
def missing_knights := 3
def missing_pawns := 6

-- The theorem statement that we need to prove
theorem pieces_present : 
  (total_pieces - (missing_kings + missing_queens + missing_knights + missing_pawns)) = 20 :=
by
  sorry

end pieces_present_l2351_235179


namespace scientific_notation_of_0_0000021_l2351_235158

theorem scientific_notation_of_0_0000021 :
  0.0000021 = 2.1 * 10 ^ (-6) :=
sorry

end scientific_notation_of_0_0000021_l2351_235158


namespace parallel_lines_m_eq_l2351_235132

theorem parallel_lines_m_eq (m : ℝ) : 
  (∃ k : ℝ, (x y : ℝ) → 2 * x + (m + 1) * y + 4 = k * (m * x + 3 * y - 2)) → 
  (m = 2 ∨ m = -3) :=
by
  intro h
  sorry

end parallel_lines_m_eq_l2351_235132


namespace polynomial_bound_swap_l2351_235151

variable (a b c : ℝ)

theorem polynomial_bound_swap (h : ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  ∀ (x : ℝ), |x| ≤ 1 → |c * x^2 + b * x + a| ≤ 2 := by
  sorry

end polynomial_bound_swap_l2351_235151


namespace part1_tangent_line_at_1_part2_monotonic_intervals_part3_range_of_a_l2351_235184

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * x + 1
noncomputable def F (x : ℝ) (a : ℝ) : ℝ := f x - g x a

-- (Ⅰ) Equation of the tangent line to y = f(x) at x = 1
theorem part1_tangent_line_at_1 : ∀ x, (f 1 + (1 / 1) * (x - 1)) = x - 1 := sorry

-- (Ⅱ) Intervals where F(x) is monotonic
theorem part2_monotonic_intervals (a : ℝ) : 
  (a ≤ 0 → ∀ x > 0, F x a > 0) ∧ 
  (a > 0 → (∀ x > 0, x < (1 / a) → F x a > 0) ∧ (∀ x > 1 / a, F x a < 0)) := sorry

-- (Ⅲ) Range of a for which f(x) is below g(x) for all x > 0
theorem part3_range_of_a (a : ℝ) : (∀ x > 0, f x < g x a) ↔ a ∈ Set.Ioi (Real.exp (-2)) := sorry

end part1_tangent_line_at_1_part2_monotonic_intervals_part3_range_of_a_l2351_235184


namespace equation_of_AB_l2351_235177

-- Definitions based on the conditions
def circle_C (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 3

def midpoint_M (p : ℝ × ℝ) : Prop :=
  p = (1, 0)

-- The theorem to be proved
theorem equation_of_AB (x y : ℝ) (M : ℝ × ℝ) :
  circle_C x y ∧ midpoint_M M → x - y = 1 :=
by
  sorry

end equation_of_AB_l2351_235177


namespace total_servings_l2351_235150

-- Definitions for the conditions

def servings_per_carrot : ℕ := 4
def plants_per_plot : ℕ := 9
def servings_multiplier_corn : ℕ := 5
def servings_multiplier_green_bean : ℤ := 2

-- Proof statement
theorem total_servings : 
  (plants_per_plot * servings_per_carrot) + 
  (plants_per_plot * (servings_per_carrot * servings_multiplier_corn)) + 
  (plants_per_plot * (servings_per_carrot * servings_multiplier_corn / servings_multiplier_green_bean)) = 
  306 :=
by
  sorry

end total_servings_l2351_235150


namespace value_of_a_is_3_l2351_235183

def symmetric_about_x1 (a : ℝ) : Prop :=
  ∀ x : ℝ, |x + 1| + |x - a| = |2 - x + 1| + |2 - x - a|

theorem value_of_a_is_3 : symmetric_about_x1 3 :=
sorry

end value_of_a_is_3_l2351_235183
