import Mathlib

namespace certain_number_z_l818_81846

theorem certain_number_z (x y z : ℝ) (h1 : 0.5 * x = y + z) (h2 : x - 2 * y = 40) : z = 20 :=
by 
  sorry

end certain_number_z_l818_81846


namespace diana_can_paint_statues_l818_81834

theorem diana_can_paint_statues : (3 / 6) / (1 / 6) = 3 := 
by 
  sorry

end diana_can_paint_statues_l818_81834


namespace simplify_expression_l818_81862

theorem simplify_expression (x : ℝ) : 120 * x - 72 * x + 15 * x - 9 * x = 54 * x := 
by
  sorry

end simplify_expression_l818_81862


namespace no_unhappy_days_l818_81864

theorem no_unhappy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := 
by 
  sorry

end no_unhappy_days_l818_81864


namespace sheets_borrowed_l818_81865

-- Definitions based on conditions
def total_pages : ℕ := 60  -- Hiram's algebra notes are 60 pages
def total_sheets : ℕ := 30  -- printed on 30 sheets of paper
def average_remaining : ℕ := 23  -- the average of the page numbers on all remaining sheets is 23

-- Let S_total be the sum of all page numbers initially
def S_total := (total_pages * (1 + total_pages)) / 2

-- Let c be the number of consecutive sheets borrowed
-- Let b be the number of sheets before the borrowed sheets
-- Calculate S_borrowed based on problem conditions
def S_borrowed (c b : ℕ) := 2 * c * (b + c) + c

-- Calculate the remaining sum and corresponding mean
def remaining_sum (c b : ℕ) := S_total - S_borrowed c b
def remaining_mean (c : ℕ) := (total_sheets * 2 - 2 * c)

-- The theorem we want to prove
theorem sheets_borrowed (c : ℕ) (h : 1830 - S_borrowed c 10 = 23 * (60 - 2 * c)) : c = 15 :=
  sorry

end sheets_borrowed_l818_81865


namespace fraction_of_25_l818_81843

theorem fraction_of_25 (x : ℝ) (h1 : 0.65 * 40 = 26) (h2 : 26 = x * 25 + 6) : x = 4 / 5 :=
sorry

end fraction_of_25_l818_81843


namespace rings_on_fingers_arrangement_l818_81857

-- Definitions based on the conditions
def rings : ℕ := 5
def fingers : ℕ := 5

-- Theorem statement
theorem rings_on_fingers_arrangement : (fingers ^ rings) = 5 ^ 5 := by
  sorry  -- Proof skipped

end rings_on_fingers_arrangement_l818_81857


namespace total_number_of_flowers_is_correct_l818_81872

-- Define the conditions
def number_of_pots : ℕ := 544
def flowers_per_pot : ℕ := 32
def total_flowers : ℕ := number_of_pots * flowers_per_pot

-- State the theorem to be proved
theorem total_number_of_flowers_is_correct :
  total_flowers = 17408 :=
by
  sorry

end total_number_of_flowers_is_correct_l818_81872


namespace plot_area_is_correct_l818_81871

noncomputable def scaled_area_in_acres
  (scale_cm_miles : ℕ)
  (area_conversion_factor_miles_acres : ℕ)
  (bottom_cm : ℕ)
  (top_cm : ℕ)
  (height_cm : ℕ) : ℕ :=
  let area_cm_squared := (1 / 2) * (bottom_cm + top_cm) * height_cm
  let area_in_squared_miles := area_cm_squared * (scale_cm_miles * scale_cm_miles)
  area_in_squared_miles * area_conversion_factor_miles_acres

theorem plot_area_is_correct :
  scaled_area_in_acres 3 640 18 14 12 = 1105920 :=
by
  sorry

end plot_area_is_correct_l818_81871


namespace complex_quadrant_l818_81895

theorem complex_quadrant (i : ℂ) (hi : i * i = -1) (z : ℂ) (hz : z = 1 / (1 - i)) : 
  (z.re > 0 ∧ z.im > 0) :=
by
  sorry

end complex_quadrant_l818_81895


namespace negation_of_implication_l818_81825

-- Definitions based on the conditions from part (a)
def original_prop (x : ℝ) : Prop := x > 5 → x > 0
def negation_candidate_A (x : ℝ) : Prop := x ≤ 5 → x ≤ 0

-- The goal is to prove that the negation of the original proposition
-- is equivalent to option A, that is:
theorem negation_of_implication (x : ℝ) : (¬ (x > 5 → x > 0)) = (x ≤ 5 → x ≤ 0) :=
by
  sorry

end negation_of_implication_l818_81825


namespace cost_of_fixing_clothes_l818_81829

def num_shirts : ℕ := 10
def num_pants : ℕ := 12
def time_per_shirt : ℝ := 1.5
def time_per_pant : ℝ := 3.0
def rate_per_hour : ℝ := 30.0

theorem cost_of_fixing_clothes : 
  let total_time := (num_shirts * time_per_shirt) + (num_pants * time_per_pant)
  let total_cost := total_time * rate_per_hour
  total_cost = 1530 :=
by 
  sorry

end cost_of_fixing_clothes_l818_81829


namespace find_x_l818_81851

theorem find_x (x : ℝ) (h : ⌈x⌉ * x = 182) : x = 13 :=
sorry

end find_x_l818_81851


namespace smallest_pos_int_for_congruence_l818_81831

theorem smallest_pos_int_for_congruence :
  ∃ (n : ℕ), 5 * n % 33 = 980 % 33 ∧ n > 0 ∧ n = 19 := 
by {
  sorry
}

end smallest_pos_int_for_congruence_l818_81831


namespace system_of_equations_solution_l818_81868

theorem system_of_equations_solution (x y z : ℝ) :
  (x = 6 + Real.sqrt 29 ∧ y = (5 - 2 * (6 + Real.sqrt 29)) / 3 ∧ z = (4 - (6 + Real.sqrt 29)) / 3 ∧
   x + y + z = 3 ∧ x + 2 * y - z = 2 ∧ x + y * z + z * x = 3) ∨
  (x = 6 - Real.sqrt 29 ∧ y = (5 - 2 * (6 - Real.sqrt 29)) / 3 ∧ z = (4 - (6 - Real.sqrt 29)) / 3 ∧
   x + y + z = 3 ∧ x + 2 * y - z = 2 ∧ x + y * z + z * x = 3) :=
sorry

end system_of_equations_solution_l818_81868


namespace number_of_true_propositions_l818_81821

variable {a b c : ℝ}

theorem number_of_true_propositions :
  (2 = (if (a > b → a * c ^ 2 > b * c ^ 2) then 1 else 0) +
       (if (a * c ^ 2 > b * c ^ 2 → a > b) then 1 else 0) +
       (if (¬(a * c ^ 2 > b * c ^ 2) → ¬(a > b)) then 1 else 0) +
       (if (¬(a > b) → ¬(a * c ^ 2 > b * c ^ 2)) then 1 else 0)) :=
sorry

end number_of_true_propositions_l818_81821


namespace remainder_of_2519_div_8_l818_81826

theorem remainder_of_2519_div_8 : 2519 % 8 = 7 := 
by 
  sorry

end remainder_of_2519_div_8_l818_81826


namespace number_of_squares_with_prime_condition_l818_81804

theorem number_of_squares_with_prime_condition : 
  ∃! (n : ℕ), ∃ (p : ℕ), Prime p ∧ n^2 = p + 4 := 
sorry

end number_of_squares_with_prime_condition_l818_81804


namespace john_writes_book_every_2_months_l818_81894

theorem john_writes_book_every_2_months
    (years_writing : ℕ)
    (average_earnings_per_book : ℕ)
    (total_earnings : ℕ)
    (H1 : years_writing = 20)
    (H2 : average_earnings_per_book = 30000)
    (H3 : total_earnings = 3600000) : 
    (years_writing * 12 / (total_earnings / average_earnings_per_book)) = 2 :=
by
    sorry

end john_writes_book_every_2_months_l818_81894


namespace compare_expressions_l818_81812

theorem compare_expressions (a b : ℝ) : (a + 3) * (a - 5) < (a + 2) * (a - 4) :=
by {
  sorry
}

end compare_expressions_l818_81812


namespace intersection_equality_l818_81867

def setA := {x : ℝ | (x - 1) * (3 - x) < 0}
def setB := {x : ℝ | -3 ≤ x ∧ x ≤ 3}

theorem intersection_equality : setA ∩ setB = {x : ℝ | -3 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_equality_l818_81867


namespace expansion_of_expression_l818_81860

theorem expansion_of_expression (x : ℝ) :
  let a := 15 * x^2 + 5 - 3 * x
  let b := 3 * x^3
  a * b = 45 * x^5 - 9 * x^4 + 15 * x^3 := by
  sorry

end expansion_of_expression_l818_81860


namespace equation_one_equation_two_l818_81814

-- Equation (1): Show that for the equation ⟦ ∀ x, (x / (2 * x - 1) + 2 / (1 - 2 * x) = 3 ↔ x = 1 / 5) ⟧
theorem equation_one (x : ℝ) : (x / (2 * x - 1) + 2 / (1 - 2 * x) = 3) ↔ (x = 1 / 5) :=
sorry

-- Equation (2): Show that for the equation ⟦ ∀ x, ((4 / (x^2 - 4) - 1 / (x - 2) = 0) ↔ false) ⟧
theorem equation_two (x : ℝ) : (4 / (x^2 - 4) - 1 / (x - 2) = 0) ↔ false :=
sorry

end equation_one_equation_two_l818_81814


namespace basketball_free_throws_l818_81833

theorem basketball_free_throws:
  ∀ (a b x : ℕ),
    3 * b = 4 * a →
    x = 2 * a →
    2 * a + 3 * b + x = 65 →
    x = 18 := 
by
  intros a b x h1 h2 h3
  sorry

end basketball_free_throws_l818_81833


namespace negation_proposition_l818_81838

theorem negation_proposition (x : ℝ) :
  ¬(∀ x : ℝ, x^2 - x + 3 > 0) ↔ ∃ x : ℝ, x^2 - x + 3 ≤ 0 := 
by { sorry }

end negation_proposition_l818_81838


namespace cannot_determine_total_movies_l818_81884

def number_of_books : ℕ := 22
def books_read : ℕ := 12
def books_to_read : ℕ := 10
def movies_watched : ℕ := 56

theorem cannot_determine_total_movies (n : ℕ) (h1 : books_read + books_to_read = number_of_books) : n ≠ movies_watched → n = 56 → False := 
by 
  intro h2 h3
  sorry

end cannot_determine_total_movies_l818_81884


namespace speed_in_still_water_l818_81849

theorem speed_in_still_water (upstream downstream : ℝ) (h_upstream : upstream = 37) (h_downstream : downstream = 53) : 
  (upstream + downstream) / 2 = 45 := 
by
  sorry

end speed_in_still_water_l818_81849


namespace original_rectangle_area_is_56_l818_81879

-- Conditions
def original_rectangle_perimeter := 30 -- cm
def smaller_rectangle_perimeter := 16 -- cm
def side_length_square := (original_rectangle_perimeter - smaller_rectangle_perimeter) / 2 -- Using the reduction logic

-- Computing the length and width of the original rectangle.
def width_original_rectangle := side_length_square
def length_original_rectangle := smaller_rectangle_perimeter / 2

-- The goal is to prove that the area of the original rectangle is 56 cm^2.

theorem original_rectangle_area_is_56:
  (length_original_rectangle - width_original_rectangle + width_original_rectangle) = 8 -- finding the length
  ∧ (length_original_rectangle * width_original_rectangle) = 56 := by
  sorry

end original_rectangle_area_is_56_l818_81879


namespace roots_cubic_inv_sum_l818_81817

theorem roots_cubic_inv_sum (a b c r s : ℝ) (h_eq : ∃ (r s : ℝ), r^2 * a + b * r - c = 0 ∧ s^2 * a + b * s - c = 0) :
  (1 / r^3) + (1 / s^3) = (b^3 + 3 * a * b * c) / c^3 :=
by
  sorry

end roots_cubic_inv_sum_l818_81817


namespace pen_average_price_l818_81811

theorem pen_average_price (pens_purchased pencils_purchased : ℕ) (total_cost pencil_avg_price : ℝ)
  (H0 : pens_purchased = 30) (H1 : pencils_purchased = 75) 
  (H2 : total_cost = 690) (H3 : pencil_avg_price = 2) :
  (total_cost - (pencils_purchased * pencil_avg_price)) / pens_purchased = 18 :=
by
  rw [H0, H1, H2, H3]
  sorry

end pen_average_price_l818_81811


namespace negative_solution_exists_l818_81822

theorem negative_solution_exists (a b c x y : ℝ) :
  (a * x + b * y = c ∧ b * x + c * y = a ∧ c * x + a * y = b) ∧ (x < 0 ∧ y < 0) ↔ a + b + c = 0 :=
sorry

end negative_solution_exists_l818_81822


namespace dessert_eating_contest_l818_81802

theorem dessert_eating_contest (a b c : ℚ) 
  (h1 : a = 5/6) 
  (h2 : b = 7/8) 
  (h3 : c = 1/2) :
  b - a = 1/24 ∧ a - c = 1/3 := 
by 
  sorry

end dessert_eating_contest_l818_81802


namespace train_length_approx_l818_81876

noncomputable def length_of_train (distance_km : ℝ) (time_min : ℝ) (time_sec : ℝ) : ℝ :=
  let distance_m := distance_km * 1000 -- Convert km to meters
  let time_s := time_min * 60 -- Convert min to seconds
  let speed := distance_m / time_s -- Speed in meters/second
  speed * time_sec -- Length of train in meters

theorem train_length_approx :
  length_of_train 10 15 10 = 111.1 :=
by
  sorry

end train_length_approx_l818_81876


namespace simplify_expression_l818_81828

theorem simplify_expression : 
  (20 * (9 / 14) * (1 / 18) : ℚ) = (5 / 7) := 
by 
  sorry

end simplify_expression_l818_81828


namespace Alice_more_nickels_l818_81891

-- Define quarters each person has
def Alice_quarters (q : ℕ) : ℕ := 10 * q + 2
def Bob_quarters (q : ℕ) : ℕ := 2 * q + 10

-- Prove that Alice has 40(q - 1) more nickels than Bob
theorem Alice_more_nickels (q : ℕ) : 
  (5 * (Alice_quarters q - Bob_quarters q)) = 40 * (q - 1) :=
by
  sorry

end Alice_more_nickels_l818_81891


namespace initial_walking_speed_l818_81890

variable (v : ℝ)

theorem initial_walking_speed :
  (13.5 / v - 13.5 / 6 = 27 / 60) → v = 5 :=
by
  intro h
  sorry

end initial_walking_speed_l818_81890


namespace sum_fraction_nonnegative_le_one_l818_81819

theorem sum_fraction_nonnegative_le_one 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (habc : a + b + c = 2) :
  a * b / (c^2 + 1) + b * c / (a^2 + 1) + c * a / (b^2 + 1) ≤ 1 :=
sorry

end sum_fraction_nonnegative_le_one_l818_81819


namespace committee_size_l818_81878

theorem committee_size (n : ℕ) (h : 2 * n = 6) (p : ℚ) (h_prob : p = 2/5) : n = 3 :=
by
  -- problem conditions
  have h1 : 2 * n = 6 := h
  have h2 : p = 2/5 := h_prob
  -- skip the proof details
  sorry

end committee_size_l818_81878


namespace max_ounces_among_items_l818_81883

theorem max_ounces_among_items
  (budget : ℝ)
  (candy_cost : ℝ)
  (candy_ounces : ℝ)
  (candy_stock : ℕ)
  (chips_cost : ℝ)
  (chips_ounces : ℝ)
  (chips_stock : ℕ)
  : budget = 7 → candy_cost = 1.25 → candy_ounces = 12 →
    candy_stock = 5 → chips_cost = 1.40 → chips_ounces = 17 → chips_stock = 4 →
    max (min ((budget / candy_cost) * candy_ounces) (candy_stock * candy_ounces))
        (min ((budget / chips_cost) * chips_ounces) (chips_stock * chips_ounces)) = 68 := 
by
  intros h_budget h_candy_cost h_candy_ounces h_candy_stock h_chips_cost h_chips_ounces h_chips_stock
  sorry

end max_ounces_among_items_l818_81883


namespace each_child_play_time_l818_81807

theorem each_child_play_time (n_children : ℕ) (game_time : ℕ) (children_per_game : ℕ)
  (h1 : n_children = 8) (h2 : game_time = 120) (h3 : children_per_game = 2) :
  ((children_per_game * game_time) / n_children) = 30 :=
  sorry

end each_child_play_time_l818_81807


namespace ajay_saves_each_month_l818_81832

def monthly_income : ℝ := 90000
def spend_household : ℝ := 0.50 * monthly_income
def spend_clothes : ℝ := 0.25 * monthly_income
def spend_medicines : ℝ := 0.15 * monthly_income
def total_spent : ℝ := spend_household + spend_clothes + spend_medicines
def amount_saved : ℝ := monthly_income - total_spent

theorem ajay_saves_each_month : amount_saved = 9000 :=
by sorry

end ajay_saves_each_month_l818_81832


namespace coloring_connected_circles_diff_colors_l818_81888

def num_ways_to_color_five_circles : ℕ :=
  36

theorem coloring_connected_circles_diff_colors (A B C D E : Type) (colors : Fin 3) 
  (connected : (A → B → C → D → E → Prop)) : num_ways_to_color_five_circles = 36 :=
by sorry

end coloring_connected_circles_diff_colors_l818_81888


namespace length_of_crease_l818_81830

theorem length_of_crease (θ : ℝ) : 
  let B := 5
  let DM := 5 * (Real.tan θ)
  DM = 5 * (Real.tan θ) := 
by 
  sorry

end length_of_crease_l818_81830


namespace final_statue_weight_l818_81856

-- Define the initial weight of the statue
def initial_weight : ℝ := 250

-- Define the percentage of weight remaining after each week
def remaining_after_week1 (w : ℝ) : ℝ := 0.70 * w
def remaining_after_week2 (w : ℝ) : ℝ := 0.80 * w
def remaining_after_week3 (w : ℝ) : ℝ := 0.75 * w

-- Define the final weight of the statue after three weeks
def final_weight : ℝ := 
  remaining_after_week3 (remaining_after_week2 (remaining_after_week1 initial_weight))

-- Prove the weight of the final statue is 105 kg
theorem final_statue_weight : final_weight = 105 := 
  by
    sorry

end final_statue_weight_l818_81856


namespace smallest_number_increased_by_nine_divisible_by_8_11_24_l818_81835

theorem smallest_number_increased_by_nine_divisible_by_8_11_24 :
  ∃ x : ℕ, (x + 9) % 8 = 0 ∧ (x + 9) % 11 = 0 ∧ (x + 9) % 24 = 0 ∧ x = 255 :=
by
  sorry

end smallest_number_increased_by_nine_divisible_by_8_11_24_l818_81835


namespace champion_is_C_l818_81803

-- Definitions of statements made by Zhang, Wang, and Li
def zhang_statement (winner : String) : Bool := winner = "A" ∨ winner = "B"
def wang_statement (winner : String) : Bool := winner ≠ "C"
def li_statement (winner : String) : Bool := winner ≠ "A" ∧ winner ≠ "B"

-- Predicate that indicates exactly one of the statements is correct
def exactly_one_correct (winner : String) : Prop :=
  (zhang_statement winner ∧ ¬wang_statement winner ∧ ¬li_statement winner) ∨
  (¬zhang_statement winner ∧ wang_statement winner ∧ ¬li_statement winner) ∨
  (¬zhang_statement winner ∧ ¬wang_statement winner ∧ li_statement winner)

-- The theorem stating the correct answer to the problem
theorem champion_is_C : (exactly_one_correct "C") :=
  by
    sorry  -- Proof goes here

-- Note: The import statement and sorry definition are included to ensure the code builds.

end champion_is_C_l818_81803


namespace sugar_already_put_in_l818_81898

-- Define the conditions
def totalSugarRequired : Nat := 14
def sugarNeededToAdd : Nat := 12
def sugarAlreadyPutIn (total : Nat) (needed : Nat) : Nat := total - needed

--State the theorem
theorem sugar_already_put_in :
  sugarAlreadyPutIn totalSugarRequired sugarNeededToAdd = 2 := 
  by
    -- Providing 'sorry' as a placeholder for the actual proof
    sorry

end sugar_already_put_in_l818_81898


namespace coprime_integers_lt_15_l818_81841

theorem coprime_integers_lt_15 : ∃ (S : Finset ℕ), S.card = 8 ∧ (∀ a ∈ S, a < 15 ∧ Nat.gcd a 15 = 1) :=
by
  sorry

end coprime_integers_lt_15_l818_81841


namespace students_drawn_from_class_A_l818_81823

-- Given conditions
def classA_students : Nat := 40
def classB_students : Nat := 50
def total_sample : Nat := 18

-- Predicate that checks if the number of students drawn from Class A is correct
theorem students_drawn_from_class_A (students_from_A : Nat) : students_from_A = 9 :=
by
  sorry

end students_drawn_from_class_A_l818_81823


namespace positive_m_for_one_root_l818_81881

theorem positive_m_for_one_root (m : ℝ) (h : (6 * m)^2 - 4 * 1 * 2 * m = 0) : m = 2 / 9 :=
by
  sorry

end positive_m_for_one_root_l818_81881


namespace instantaneous_velocity_at_2_l818_81855

def displacement (t : ℝ) : ℝ := 100 * t - 5 * t^2

noncomputable def instantaneous_velocity_at (s : ℝ → ℝ) (t : ℝ) : ℝ :=
  (deriv s) t

theorem instantaneous_velocity_at_2 : instantaneous_velocity_at displacement 2 = 80 :=
by
  sorry

end instantaneous_velocity_at_2_l818_81855


namespace smallest_angle_l818_81801

theorem smallest_angle (largest_angle : ℝ) (a b : ℝ) (h1 : largest_angle = 120) (h2 : 3 * a = 2 * b) (h3 : largest_angle + a + b = 180) : b = 24 := by
  sorry

end smallest_angle_l818_81801


namespace angle_SRT_l818_81805

-- Define angles in degrees
def angle_P : ℝ := 50
def angle_Q : ℝ := 60
def angle_R : ℝ := 40

-- Define the problem: Prove that angle SRT is 30 degrees given the above conditions
theorem angle_SRT : 
  (angle_P = 50 ∧ angle_Q = 60 ∧ angle_R = 40) → (∃ angle_SRT : ℝ, angle_SRT = 30) :=
by
  intros h
  sorry

end angle_SRT_l818_81805


namespace balance_balls_l818_81853

theorem balance_balls (R O G B : ℝ) (h₁ : 4 * R = 8 * G) (h₂ : 3 * O = 6 * G) (h₃ : 8 * G = 6 * B) :
  3 * R + 2 * O + 4 * B = (46 / 3) * G :=
by
  -- Using the given conditions to derive intermediate results (included in the detailed proof, not part of the statement)
  sorry

end balance_balls_l818_81853


namespace arithmetic_sequence_a10_l818_81836

theorem arithmetic_sequence_a10 (a : ℕ → ℕ) (d : ℕ) 
  (h1 : a 1 = 1) (h3 : a 3 = 5) 
  (h_diff : d = (a 3 - a 1) / (3 - 1)) :
  a 10 = 19 := 
by 
  sorry

end arithmetic_sequence_a10_l818_81836


namespace eq_value_of_2a_plus_b_l818_81840

theorem eq_value_of_2a_plus_b (a b : ℝ) (h : abs (a + 2) + (b - 5)^2 = 0) : 2 * a + b = 1 := by
  sorry

end eq_value_of_2a_plus_b_l818_81840


namespace simplify_expression_l818_81885

theorem simplify_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 6) + 7) + 2 = -x^4 + 3 * x^3 - 6 * x^2 + 7 * x + 2 := 
by 
  sorry

end simplify_expression_l818_81885


namespace relationship_xy_qz_l818_81809

theorem relationship_xy_qz
  (a c b d : ℝ)
  (x y q z : ℝ)
  (h1 : a^(2 * x) = c^(2 * q) ∧ c^(2 * q) = b^2)
  (h2 : c^(3 * y) = a^(3 * z) ∧ a^(3 * z) = d^2) :
  x * y = q * z :=
by
  sorry

end relationship_xy_qz_l818_81809


namespace find_m_value_l818_81863

theorem find_m_value (f g : ℝ → ℝ) (m : ℝ)
    (hf : ∀ x, f x = 3 * x ^ 2 - 1 / x + 4)
    (hg : ∀ x, g x = x ^ 2 - m)
    (hfg : f 3 - g 3 = 5) :
    m = -50 / 3 :=
  sorry

end find_m_value_l818_81863


namespace trig_identity_l818_81882

theorem trig_identity : Real.sin (35 * Real.pi / 6) + Real.cos (-11 * Real.pi / 3) = 0 := by
  sorry

end trig_identity_l818_81882


namespace platform_length_l818_81875

theorem platform_length
  (L_train : ℕ) (T_platform : ℕ) (T_pole : ℕ) (P : ℕ)
  (h1 : L_train = 300)
  (h2 : T_platform = 39)
  (h3 : T_pole = 10)
  (h4 : L_train / T_pole * T_platform = L_train + P) :
  P = 870 := 
sorry

end platform_length_l818_81875


namespace employed_females_percentage_l818_81866

theorem employed_females_percentage (total_population_percent employed_population_percent employed_males_percent : ℝ) :
  employed_population_percent = 70 → employed_males_percent = 21 →
  (employed_population_percent - employed_males_percent) / employed_population_percent * 100 = 70 :=
by
  -- Assume the total population percentage is 100%, which allows us to work directly with percentages.
  let employed_population_percent := 70
  let employed_males_percent := 21
  sorry

end employed_females_percentage_l818_81866


namespace notepad_duration_l818_81845

theorem notepad_duration (a8_papers_per_a4 : ℕ)
  (a4_papers : ℕ)
  (notes_per_day : ℕ)
  (notes_per_side : ℕ) :
  a8_papers_per_a4 = 16 →
  a4_papers = 8 →
  notes_per_day = 15 →
  notes_per_side = 2 →
  (a4_papers * a8_papers_per_a4 * notes_per_side) / notes_per_day = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end notepad_duration_l818_81845


namespace trig_identity_l818_81880

theorem trig_identity (α : ℝ) (h : Real.tan α = 3 / 4) : 
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := 
by
  sorry

end trig_identity_l818_81880


namespace largest_possible_value_l818_81806

theorem largest_possible_value (X Y Z m: ℕ) 
  (hX_range: 0 ≤ X ∧ X ≤ 4) 
  (hY_range: 0 ≤ Y ∧ Y ≤ 4) 
  (hZ_range: 0 ≤ Z ∧ Z ≤ 4) 
  (h1: m = 25 * X + 5 * Y + Z)
  (h2: m = 81 * Z + 9 * Y + X):
  m = 121 :=
by
  -- The proof goes here
  sorry

end largest_possible_value_l818_81806


namespace buckets_needed_to_fill_tank_l818_81839

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem buckets_needed_to_fill_tank :
  let radius_tank := 8
  let height_tank := 32
  let radius_bucket := 8
  let volume_bucket := volume_of_sphere radius_bucket
  let volume_tank := volume_of_cylinder radius_tank height_tank
  volume_tank / volume_bucket = 3 :=
by sorry

end buckets_needed_to_fill_tank_l818_81839


namespace max_rectangle_area_l818_81887

theorem max_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 48) : x * y ≤ 144 :=
by
  sorry

end max_rectangle_area_l818_81887


namespace cos_double_angle_l818_81847

theorem cos_double_angle (α : ℝ) (h : Real.sin (Real.pi / 2 + α) = 3 / 5) : Real.cos (2 * α) = -7 / 25 :=
sorry

end cos_double_angle_l818_81847


namespace fraction_of_q_age_l818_81854

theorem fraction_of_q_age (P Q : ℕ) (h1 : P / Q = 3 / 4) (h2 : P + Q = 28) : (P - 0) / (Q - 0) = 3 / 4 :=
by
  sorry

end fraction_of_q_age_l818_81854


namespace problem1_solution_problem2_solution_l818_81886

-- Problem 1
theorem problem1_solution (x : ℝ) : (2 * x - 3) * (x + 1) < 0 ↔ (-1 < x) ∧ (x < 3 / 2) :=
sorry

-- Problem 2
theorem problem2_solution (x : ℝ) : (4 * x - 1) / (x + 2) ≥ 0 ↔ (x < -2) ∨ (x >= 1 / 4) :=
sorry

end problem1_solution_problem2_solution_l818_81886


namespace sum_of_cubes_l818_81820

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -3) : a^3 + b^3 = 26 :=
sorry

end sum_of_cubes_l818_81820


namespace length_of_rectangle_l818_81827

theorem length_of_rectangle (L : ℝ) (W : ℝ) (A_triangle : ℝ) (hW : W = 4) (hA_triangle : A_triangle = 60)
  (hRatio : (L * W) / A_triangle = 2 / 5) : L = 6 :=
by
  sorry

end length_of_rectangle_l818_81827


namespace sequence_a_10_l818_81808

theorem sequence_a_10 : ∀ {a : ℕ → ℕ}, (a 1 = 1) → (∀ n, a (n+1) = a n + 2^n) → (a 10 = 1023) :=
by
  intros a h1 h_rec
  sorry

end sequence_a_10_l818_81808


namespace negation_of_p_l818_81869

open Real

-- Define the original proposition p
def p := ∀ x : ℝ, 0 < x → x^2 > log x

-- State the theorem with its negation
theorem negation_of_p : ¬p ↔ ∃ x : ℝ, 0 < x ∧ x^2 ≤ log x :=
by
  sorry

end negation_of_p_l818_81869


namespace find_a_values_for_eccentricity_l818_81873

theorem find_a_values_for_eccentricity (a : ℝ) : 
  ( ∃ a : ℝ, ((∀ x y : ℝ, (x^2 / (a+8) + y^2 / 9 = 1)) ∧ (e = 1/2) ) 
  → (a = 4 ∨ a = -5/4)) := 
sorry

end find_a_values_for_eccentricity_l818_81873


namespace rectangle_area_given_perimeter_l818_81896

theorem rectangle_area_given_perimeter (x : ℝ) (h_perim : 8 * x = 160) : (2 * x) * (2 * x) = 1600 := by
  -- Definitions derived from conditions
  let length := 2 * x
  let width := 2 * x
  -- Proof transformed to a Lean statement
  have h1 : length = 40 := by sorry
  have h2 : width = 40 := by sorry
  have h_area : length * width = 1600 := by sorry
  exact h_area

end rectangle_area_given_perimeter_l818_81896


namespace no_rational_x_y_m_n_with_conditions_l818_81842

noncomputable def f (t : ℚ) : ℚ := t^3 + t

theorem no_rational_x_y_m_n_with_conditions :
  ¬ ∃ (x y : ℚ) (m n : ℕ), xy = 3 ∧ m > 0 ∧ n > 0 ∧
    (f^[m] x = f^[n] y) := 
sorry

end no_rational_x_y_m_n_with_conditions_l818_81842


namespace B_subscribed_fraction_correct_l818_81897

-- Define the total capital and the shares of A, C
variables (X : ℝ) (profit : ℝ) (A_share : ℝ) (C_share : ℝ)

-- Define the conditions as given in the problem
def A_capital_share := 1 / 3
def C_capital_share := 1 / 5
def total_profit := 2430
def A_profit_share := 810

-- Define the calculation of B's share
def B_capital_share := 1 - (A_capital_share + C_capital_share)

-- Define the expected correct answer for B's share
def expected_B_share := 7 / 15

-- Theorem statement
theorem B_subscribed_fraction_correct :
  B_capital_share = expected_B_share :=
by
  sorry

end B_subscribed_fraction_correct_l818_81897


namespace maximum_M_value_l818_81837

theorem maximum_M_value (x y z u M : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 0 < u)
  (h5 : x - 2 * y = z - 2 * u) (h6 : 2 * y * z = u * x) (h7 : z ≥ y) 
  : ∃ M, M ≤ z / y ∧ M ≤ 6 + 4 * Real.sqrt 2 :=
sorry

end maximum_M_value_l818_81837


namespace melanie_total_payment_l818_81813

noncomputable def totalCost (rentalCostPerDay : ℝ) (insuranceCostPerDay : ℝ) (mileageCostPerMile : ℝ) (days : ℕ) (miles : ℕ) : ℝ :=
  (rentalCostPerDay * days) + (insuranceCostPerDay * days) + (mileageCostPerMile * miles)

theorem melanie_total_payment :
  totalCost 30 5 0.25 3 350 = 192.5 :=
by
  sorry

end melanie_total_payment_l818_81813


namespace g_value_at_2_l818_81859

def g (x : ℝ) (d : ℝ) : ℝ := 3 * x^5 - 2 * x^4 + d * x - 8

theorem g_value_at_2 (d : ℝ) (h : g (-2) d = 4) : g 2 d = -84 := by
  sorry

end g_value_at_2_l818_81859


namespace scientific_notation_l818_81852

theorem scientific_notation (n : ℝ) (h : n = 40.9 * 10^9) : n = 4.09 * 10^10 :=
by sorry

end scientific_notation_l818_81852


namespace blue_line_length_correct_l818_81889

def white_line_length : ℝ := 7.67
def difference_in_length : ℝ := 4.33
def blue_line_length : ℝ := 3.34

theorem blue_line_length_correct :
  white_line_length - difference_in_length = blue_line_length :=
by
  sorry

end blue_line_length_correct_l818_81889


namespace ways_to_stand_l818_81893

-- Definitions derived from conditions
def num_steps : ℕ := 7
def max_people_per_step : ℕ := 2

-- Define a function to count the number of different ways
noncomputable def count_ways : ℕ :=
  336

-- The statement to be proven in Lean 4
theorem ways_to_stand : count_ways = 336 :=
  sorry

end ways_to_stand_l818_81893


namespace length_of_living_room_l818_81818

theorem length_of_living_room
  (l : ℝ) -- length of the living room
  (w : ℝ) -- width of the living room
  (boxes_coverage : ℝ) -- area covered by one box
  (initial_area : ℝ) -- area already covered
  (additional_boxes : ℕ) -- additional boxes required
  (total_area : ℝ) -- total area required
  (w_condition : w = 20)
  (boxes_coverage_condition : boxes_coverage = 10)
  (initial_area_condition : initial_area = 250)
  (additional_boxes_condition : additional_boxes = 7)
  (total_area_condition : total_area = l * w)
  (full_coverage_condition : additional_boxes * boxes_coverage + initial_area = total_area) :
  l = 16 := by
  sorry

end length_of_living_room_l818_81818


namespace ratio_of_spinsters_to_cats_l818_81874

-- Definitions for the conditions given:
def S : ℕ := 12 -- 12 spinsters
def C : ℕ := S + 42 -- 42 more cats than spinsters
def ratio (a b : ℕ) : ℚ := a / b -- Ratio definition

-- The theorem stating the required equivalence:
theorem ratio_of_spinsters_to_cats :
  ratio S C = 2 / 9 :=
by
  -- This proof has been omitted for the purpose of this exercise.
  sorry

end ratio_of_spinsters_to_cats_l818_81874


namespace num_perfect_cubes_between_bounds_l818_81850

   noncomputable def lower_bound := 2^8 + 1
   noncomputable def upper_bound := 2^18 + 1

   theorem num_perfect_cubes_between_bounds : 
     ∃ (k : ℕ), k = 58 ∧ (∀ (n : ℕ), (lower_bound ≤ n^3 ∧ n^3 ≤ upper_bound) ↔ (7 ≤ n ∧ n ≤ 64)) :=
   sorry
   
end num_perfect_cubes_between_bounds_l818_81850


namespace susan_spent_total_l818_81870

-- Definitions for the costs and quantities
def pencil_cost : ℝ := 0.25
def pen_cost : ℝ := 0.80
def total_items : ℕ := 36
def pencils_bought : ℕ := 16

-- Question: How much did Susan spend?
theorem susan_spent_total : (pencil_cost * pencils_bought + pen_cost * (total_items - pencils_bought)) = 20 :=
by
    -- definition goes here
    sorry

end susan_spent_total_l818_81870


namespace plates_probability_l818_81861

noncomputable def number_of_plates := 12
noncomputable def red_plates := 6
noncomputable def light_blue_plates := 3
noncomputable def dark_blue_plates := 3
noncomputable def total_pairs := number_of_plates * (number_of_plates - 1) / 2
noncomputable def red_pairs := red_plates * (red_plates - 1) / 2
noncomputable def light_blue_pairs := light_blue_plates * (light_blue_plates - 1) / 2
noncomputable def dark_blue_pairs := dark_blue_plates * (dark_blue_plates - 1) / 2
noncomputable def mixed_blue_pairs := light_blue_plates * dark_blue_plates
noncomputable def total_satisfying_pairs := red_pairs + light_blue_pairs + dark_blue_pairs + mixed_blue_pairs
noncomputable def desired_probability := (total_satisfying_pairs : ℚ) / total_pairs

theorem plates_probability :
  desired_probability = 5 / 11 :=
by
  -- Add the proof here
  sorry

end plates_probability_l818_81861


namespace complement_union_l818_81810

-- Definition of the universal set U
def U : Set ℤ := {x | x^2 - 5 * x - 6 ≤ 0}

-- Definition of set A
def A : Set ℤ := {x | x * (2 - x) ≥ 0}

-- Definition of set B
def B : Set ℤ := {1, 2, 3}

-- The proof statement
theorem complement_union (h : U = {x | x^2 - 5 * x - 6 ≤ 0} ∧ 
                           A = {x | x * (2 - x) ≥ 0} ∧ 
                           B = {1, 2, 3}) : 
  U \ (A ∪ B) = {-1, 4, 5, 6} :=
by {
  sorry
}

end complement_union_l818_81810


namespace marbles_before_purchase_l818_81844

-- Lean 4 statement for the problem
theorem marbles_before_purchase (bought : ℝ) (total_now : ℝ) (initial : ℝ) 
    (h1 : bought = 134.0) 
    (h2 : total_now = 321) 
    (h3 : total_now = initial + bought) : 
    initial = 187 :=
by 
    sorry

end marbles_before_purchase_l818_81844


namespace plane_equation_and_gcd_l818_81892

variable (x y z : ℝ)

theorem plane_equation_and_gcd (A B C D : ℤ) (h1 : A = 8) (h2 : B = -6) (h3 : C = 5) (h4 : D = -125) :
    (A * x + B * y + C * z + D = 0 ↔ x = 8 ∧ y = -6 ∧ z = 5) ∧
    Int.gcd (Int.gcd A B) (Int.gcd C D) = 1 :=
by sorry

end plane_equation_and_gcd_l818_81892


namespace final_percentage_of_alcohol_l818_81899

theorem final_percentage_of_alcohol (initial_volume : ℝ) (initial_alcohol_percentage : ℝ)
  (removed_alcohol : ℝ) (added_water : ℝ) :
  initial_volume = 15 → initial_alcohol_percentage = 25 →
  removed_alcohol = 2 → added_water = 3 →
  ( ( (initial_alcohol_percentage / 100 * initial_volume - removed_alcohol) / 
    (initial_volume - removed_alcohol + added_water) ) * 100 = 10.9375) :=
by
  intros
  sorry

end final_percentage_of_alcohol_l818_81899


namespace marks_in_mathematics_l818_81824

-- Define the marks obtained in each subject and the average
def marks_in_english : ℕ := 86
def marks_in_physics : ℕ := 82
def marks_in_chemistry : ℕ := 87
def marks_in_biology : ℕ := 85
def average_marks : ℕ := 85
def number_of_subjects : ℕ := 5

-- The theorem to prove the marks in Mathematics
theorem marks_in_mathematics : ℕ :=
  let sum_of_marks := average_marks * number_of_subjects
  let sum_of_known_marks := marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology
  sum_of_marks - sum_of_known_marks

-- The expected result that we need to prove
example : marks_in_mathematics = 85 := by
  -- skip the proof
  sorry

end marks_in_mathematics_l818_81824


namespace matrix_pow_C_50_l818_81815

def C : Matrix (Fin 2) (Fin 2) ℤ := 
  !![3, 1; -4, -1]

theorem matrix_pow_C_50 : C^50 = !![101, 50; -200, -99] := 
  sorry

end matrix_pow_C_50_l818_81815


namespace sum_expression_l818_81816

theorem sum_expression (x k : ℝ) (h1 : y = 3 * x) (h2 : z = k * y) : x + y + z = (4 + 3 * k) * x :=
by
  sorry

end sum_expression_l818_81816


namespace four_letter_list_product_l818_81877

def letter_value (c : Char) : Nat :=
  if 'A' ≤ c ∧ c ≤ 'Z' then (c.toNat - 'A'.toNat + 1) else 0

def list_product (s : String) : Nat :=
  s.foldl (λ acc c => acc * letter_value c) 1

def target_product : Nat :=
  list_product "TUVW"

theorem four_letter_list_product : 
  ∀ (s1 s2 : String), s1.toList.all (λ c => 'A' ≤ c ∧ c ≤ 'Z') → s2.toList.all (λ c => 'A' ≤ c ∧ c ≤ 'Z') →
  s1.length = 4 → s2.length = 4 →
  list_product s1 = target_product → s2 = "BEHK" :=
by
  sorry

end four_letter_list_product_l818_81877


namespace inequality_proof_l818_81800

open scoped BigOperators

theorem inequality_proof {n : ℕ} (a : Fin n → ℝ) 
  (h1 : ∀ i, 0 < a i ∧ a i ≤ 1 / 2) :
  (∑ i, (a i)^2 / (∑ i, a i)^2) ≥ (∑ i, (1 - a i)^2 / (∑ i, (1 - a i))^2) := 
by 
  sorry

end inequality_proof_l818_81800


namespace laundry_loads_l818_81858

-- Definitions based on conditions
def num_families : ℕ := 3
def people_per_family : ℕ := 4
def num_people : ℕ := num_families * people_per_family

def days : ℕ := 7
def towels_per_person_per_day : ℕ := 1
def total_towels : ℕ := num_people * days * towels_per_person_per_day

def washing_machine_capacity : ℕ := 14

-- Statement to prove
theorem laundry_loads : total_towels / washing_machine_capacity = 6 := 
by
  sorry

end laundry_loads_l818_81858


namespace calculator_change_problem_l818_81848

theorem calculator_change_problem :
  let basic_cost := 8
  let scientific_cost := 2 * basic_cost
  let graphing_cost := 3 * scientific_cost
  let total_cost := basic_cost + scientific_cost + graphing_cost
  let initial_money := 100
  let change_received := initial_money - total_cost
  change_received = 28 := by
{
  let basic_cost := 8
  let scientific_cost := 2 * basic_cost
  let graphing_cost := 3 * scientific_cost
  let total_cost := basic_cost + scientific_cost + graphing_cost
  let initial_money := 100
  let change_received := initial_money - total_cost
  have h1 : scientific_cost = 16 := sorry
  have h2 : graphing_cost = 48 := sorry
  have h3 : total_cost = 72 := sorry
  have h4 : change_received = 28 := sorry
  exact h4
}

end calculator_change_problem_l818_81848
