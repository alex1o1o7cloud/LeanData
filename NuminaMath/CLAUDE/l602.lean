import Mathlib

namespace function_equality_l602_60256

theorem function_equality (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x + f y) ≥ f (f x + y)) 
  (h2 : f 0 = 0) : 
  ∀ x : ℝ, f x = x := by
sorry

end function_equality_l602_60256


namespace squirrel_pine_cones_theorem_l602_60258

/-- Represents the number of pine cones each squirrel has -/
structure SquirrelPineCones where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Redistributes pine cones according to the problem description -/
def redistribute (initial : SquirrelPineCones) : SquirrelPineCones :=
  let step1 := SquirrelPineCones.mk (initial.a - 10) (initial.b + 5) (initial.c + 5)
  let step2 := SquirrelPineCones.mk (step1.a + 9) (step1.b - 18) (step1.c + 9)
  let final_c := step2.c / 2
  SquirrelPineCones.mk (step2.a + final_c) (step2.b + final_c) final_c

/-- The theorem to be proved -/
theorem squirrel_pine_cones_theorem (initial : SquirrelPineCones) 
  (h1 : initial.a = 26)
  (h2 : initial.c = 86) :
  let final := redistribute initial
  final.a = final.b ∧ final.b = final.c := by
  sorry

end squirrel_pine_cones_theorem_l602_60258


namespace decimal_point_problem_l602_60269

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 * (1 / x)) : 
  x = Real.sqrt 30 / 100 := by
  sorry

end decimal_point_problem_l602_60269


namespace greatest_a_for_inequality_l602_60291

theorem greatest_a_for_inequality : 
  ∃ (a : ℝ), (∀ (x₁ x₂ x₃ x₄ x₅ : ℝ), 
    x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ a * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅)) ∧ 
  (∀ (b : ℝ), (∀ (x₁ x₂ x₃ x₄ x₅ : ℝ), 
    x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 ≥ b * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅)) → b ≤ a) ∧ 
  a = 2 / Real.sqrt 3 := by
sorry

end greatest_a_for_inequality_l602_60291


namespace function_composition_implies_sum_zero_l602_60254

theorem function_composition_implies_sum_zero 
  (a b c d : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : c ≠ 0) 
  (h4 : d ≠ 0) 
  (f : ℝ → ℝ) 
  (h5 : ∀ x, f x = (2*a*x + b) / (c*x + 2*d)) 
  (h6 : ∀ x, f (f x) = 3*x - 4) : 
  a + d = 0 := by
sorry

end function_composition_implies_sum_zero_l602_60254


namespace four_circles_minus_large_circle_area_l602_60221

/-- Four circles with radius r > 0 centered at (0, r), (r, 0), (0, -r), and (-r, 0) -/
def four_circles (r : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), (x - 0)^2 + (y - r)^2 ≤ r^2 ∨
                    (x - r)^2 + (y - 0)^2 ≤ r^2 ∨
                    (x - 0)^2 + (y + r)^2 ≤ r^2 ∨
                    (x + r)^2 + (y - 0)^2 ≤ r^2}

/-- Circle with radius 2r centered at (0, 0) -/
def large_circle (r : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), x^2 + y^2 ≤ (2*r)^2}

/-- Area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem to be proved -/
theorem four_circles_minus_large_circle_area (r : ℝ) (hr : r > 0) :
  area (four_circles r) - area (large_circle r \ four_circles r) = 8 * r^2 := by sorry

end four_circles_minus_large_circle_area_l602_60221


namespace pass_rate_calculation_l602_60245

/-- Pass rate calculation for a batch of parts -/
theorem pass_rate_calculation (inspected : ℕ) (qualified : ℕ) (h1 : inspected = 40) (h2 : qualified = 38) :
  (qualified : ℚ) / (inspected : ℚ) * 100 = 95 := by
  sorry

end pass_rate_calculation_l602_60245


namespace ellipse_focal_length_2_implies_a_5_l602_60202

/-- Represents an ellipse with equation x^2/a + y^2 = 1 -/
structure Ellipse where
  a : ℝ
  h_a_gt_one : a > 1

/-- The focal length of an ellipse -/
def focal_length (e : Ellipse) : ℝ := sorry

/-- Theorem: If the focal length of the ellipse is 2, then a = 5 -/
theorem ellipse_focal_length_2_implies_a_5 (e : Ellipse) 
  (h : focal_length e = 2) : e.a = 5 := by sorry

end ellipse_focal_length_2_implies_a_5_l602_60202


namespace unique_excellent_beats_all_l602_60233

-- Define the type for players
variable {Player : Type}

-- Define the relation for "beats"
variable (beats : Player → Player → Prop)

-- Define what it means to be an excellent player
def is_excellent (A : Player) : Prop :=
  ∀ B : Player, B ≠ A → (beats A B ∨ ∃ C : Player, beats C B ∧ beats A C)

-- State the theorem
theorem unique_excellent_beats_all
  (players : Set Player)
  (h_nonempty : Set.Nonempty players)
  (h_no_self_play : ∀ A : Player, ¬beats A A)
  (h_all_play : ∀ A B : Player, A ∈ players → B ∈ players → A ≠ B → (beats A B ∨ beats B A))
  (h_unique_excellent : ∃! A : Player, A ∈ players ∧ is_excellent beats A) :
  ∃ A : Player, A ∈ players ∧ is_excellent beats A ∧ ∀ B : Player, B ∈ players → B ≠ A → beats A B :=
sorry

end unique_excellent_beats_all_l602_60233


namespace alternating_series_sum_l602_60266

def alternating_series (n : ℕ) : ℤ := 
  if n % 2 = 0 then n else -n

def series_sum (n : ℕ) : ℤ := 
  (List.range n).map (λ i => alternating_series (i + 1)) |>.sum

theorem alternating_series_sum : series_sum 11001 = 16501 := by
  sorry

end alternating_series_sum_l602_60266


namespace original_number_proof_l602_60279

theorem original_number_proof (x : ℝ) : x * 1.2 = 1080 → x = 900 := by
  sorry

end original_number_proof_l602_60279


namespace total_carrots_l602_60271

theorem total_carrots (sandy sam sarah : ℕ) 
  (h1 : sandy = 6) 
  (h2 : sam = 3) 
  (h3 : sarah = 5) : 
  sandy + sam + sarah = 14 := by
  sorry

end total_carrots_l602_60271


namespace sum_a_b_equals_negative_two_l602_60234

theorem sum_a_b_equals_negative_two (a b : ℝ) :
  (a - 2)^2 + |b + 4| = 0 → a + b = -2 := by
  sorry

end sum_a_b_equals_negative_two_l602_60234


namespace woman_work_time_l602_60280

/-- Represents the time taken to complete a work unit -/
structure WorkTime where
  men : ℕ
  women : ℕ
  days : ℕ

/-- The work done by one person in one day -/
def work_per_day (gender : String) : ℚ :=
  if gender = "man" then 1 / 100
  else 1 / 225

theorem woman_work_time : ∃ (w : ℚ),
  (10 * work_per_day "man" + 15 * w) * 6 = 1 ∧
  w = work_per_day "woman" ∧
  1 / w = 225 := by
  sorry

#check woman_work_time

end woman_work_time_l602_60280


namespace min_value_expression_lower_bound_achievable_l602_60281

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (((x^2 + y^2 + z^2) * (4*x^2 + y^2 + z^2)).sqrt) / (x*y*z) ≥ (3/2 : ℝ) :=
sorry

theorem lower_bound_achievable :
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  (((x^2 + y^2 + z^2) * (4*x^2 + y^2 + z^2)).sqrt) / (x*y*z) = (3/2 : ℝ) :=
sorry

end min_value_expression_lower_bound_achievable_l602_60281


namespace max_cookie_price_l602_60205

theorem max_cookie_price (cookie_price bun_price : ℕ) : 
  (cookie_price > 0) →
  (bun_price > 0) →
  (8 * cookie_price + 3 * bun_price < 200) →
  (4 * cookie_price + 5 * bun_price > 150) →
  cookie_price ≤ 19 ∧ 
  ∃ (max_price : ℕ), max_price = 19 ∧
    ∃ (bun_price_19 : ℕ), 
      (8 * max_price + 3 * bun_price_19 < 200) ∧
      (4 * max_price + 5 * bun_price_19 > 150) := by
  sorry

end max_cookie_price_l602_60205


namespace shaded_areas_equality_l602_60222

theorem shaded_areas_equality (φ : Real) (h : 0 < φ ∧ φ < Real.pi / 2) :
  (∃ r : Real, r > 0 ∧
    (φ * r^2 / 2 = r^2 * Real.tan φ / 2 - φ * r^2 / 2)) ↔ Real.tan φ = 2 * φ := by
  sorry

end shaded_areas_equality_l602_60222


namespace m_range_l602_60206

def p (m : ℝ) : Prop := ∀ x, x^2 - 2*m*x + 1 ≥ 0

def q (m : ℝ) : Prop := m * (m - 2) < 0

def range_m (m : ℝ) : Prop := (-1 ≤ m ∧ m ≤ 0) ∨ (1 < m ∧ m < 2)

theorem m_range : 
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m → q m) → range_m m :=
sorry

end m_range_l602_60206


namespace min_value_of_f_l602_60298

/-- Given positive numbers a, b, c, x, y, z satisfying the conditions,
    the function f(x, y, z) has a minimum value of 1/2 -/
theorem min_value_of_f (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (eq1 : c * y + b * z = a)
  (eq2 : a * z + c * x = b)
  (eq3 : b * x + a * y = c) :
  ∀ (x' y' z' : ℝ), 0 < x' → 0 < y' → 0 < z' →
    x'^2 / (1 + x') + y'^2 / (1 + y') + z'^2 / (1 + z') ≥ 1/2 :=
by sorry

end min_value_of_f_l602_60298


namespace checking_account_theorem_l602_60227

/-- Given the total amount of yen and the amount in the savings account,
    calculate the amount in the checking account. -/
def checking_account_balance (total : ℕ) (savings : ℕ) : ℕ :=
  total - savings

/-- Theorem stating that given the total amount and savings account balance,
    the checking account balance is correctly calculated. -/
theorem checking_account_theorem (total savings : ℕ) 
  (h1 : total = 9844)
  (h2 : savings = 3485) :
  checking_account_balance total savings = 6359 := by
  sorry

end checking_account_theorem_l602_60227


namespace divisibility_implies_equality_l602_60255

theorem divisibility_implies_equality (a b : ℕ+) 
  (h : (4 * a.val * b.val - 1) ∣ (4 * a.val^2 - 1)^2) : a = b := by
  sorry

end divisibility_implies_equality_l602_60255


namespace specific_arrangement_probability_l602_60277

def total_tiles : ℕ := 9
def x_tiles : ℕ := 5
def o_tiles : ℕ := 4

theorem specific_arrangement_probability :
  (1 : ℚ) / Nat.choose total_tiles x_tiles = (1 : ℚ) / 126 := by
  sorry

end specific_arrangement_probability_l602_60277


namespace set_equality_implies_values_l602_60252

theorem set_equality_implies_values (x y : ℝ) : 
  ({x, y^2, 1} : Set ℝ) = ({1, 2*x, y} : Set ℝ) → x = 2 ∧ y = 2 := by
  sorry

end set_equality_implies_values_l602_60252


namespace transaction_difference_prove_transaction_difference_l602_60287

theorem transaction_difference : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun mabel_transactions anthony_transactions cal_transactions jade_transactions =>
    mabel_transactions = 90 →
    anthony_transactions = mabel_transactions + mabel_transactions / 10 →
    cal_transactions = anthony_transactions * 2 / 3 →
    jade_transactions = 80 →
    jade_transactions - cal_transactions = 14

#check transaction_difference

theorem prove_transaction_difference :
  ∃ (mabel anthony cal jade : ℕ),
    transaction_difference mabel anthony cal jade :=
by
  sorry

end transaction_difference_prove_transaction_difference_l602_60287


namespace order_relation_l602_60267

theorem order_relation (a b c : ℝ) (ha : a = Real.log (1 + Real.exp 1))
    (hb : b = Real.sqrt (Real.exp 1)) (hc : c = (2 * Real.exp 1) / 3) :
  a < b ∧ b < c := by sorry

end order_relation_l602_60267


namespace integer_solutions_of_inequalities_l602_60259

theorem integer_solutions_of_inequalities :
  {x : ℤ | 2 * x + 4 > 0 ∧ 1 + x ≥ 2 * x - 1} = {-1, 0, 1, 2} := by
  sorry

end integer_solutions_of_inequalities_l602_60259


namespace series_sum_l602_60262

/-- The sum of the series ∑_{n=0}^{∞} (-1)^n / (3n + 1) is equal to (1/3) * (ln(2) + π/√3) -/
theorem series_sum : 
  ∑' (n : ℕ), ((-1)^n : ℝ) / (3*n + 1) = (1/3) * (Real.log 2 + Real.pi / Real.sqrt 3) := by
  sorry

end series_sum_l602_60262


namespace hyperbola_focal_length_l602_60230

theorem hyperbola_focal_length (m : ℝ) : 
  (∀ x y : ℝ, x^2 / m - y^2 / 4 = 1) →  -- Equation of the hyperbola
  (∃ c : ℝ, c = 3) →                    -- Focal length is 6 (2c = 6, so c = 3)
  (∃ a b : ℝ, a^2 = m ∧ b^2 = 4 ∧ c^2 = a^2 + b^2) →  -- Relationship between a, b, c
  m = 5 := by
sorry

end hyperbola_focal_length_l602_60230


namespace team_total_score_l602_60236

def team_score (connor_score amy_score jason_score : ℕ) : ℕ :=
  connor_score + amy_score + jason_score

theorem team_total_score : 
  ∀ (connor_score amy_score jason_score : ℕ),
    connor_score = 2 →
    amy_score = connor_score + 4 →
    jason_score = 2 * amy_score →
    team_score connor_score amy_score jason_score = 20 := by
  sorry

end team_total_score_l602_60236


namespace season_games_l602_60286

/-- The number of baseball games in a month -/
def games_per_month : ℕ := 7

/-- The number of months in a season -/
def months_in_season : ℕ := 2

/-- The total number of games in a season -/
def total_games : ℕ := games_per_month * months_in_season

theorem season_games : total_games = 14 := by sorry

end season_games_l602_60286


namespace sandy_savings_l602_60237

theorem sandy_savings (last_year_salary : ℝ) : 
  let last_year_savings := 0.1 * last_year_salary
  let this_year_salary := 1.1 * last_year_salary
  let this_year_savings := 0.6599999999999999 * last_year_savings
  (this_year_savings / this_year_salary) * 100 = 6 := by
sorry

end sandy_savings_l602_60237


namespace quadratic_range_on_unit_interval_l602_60283

/-- The range of a quadratic function on [0, 1] -/
theorem quadratic_range_on_unit_interval (a b c : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∀ x, x ∈ Set.Icc 0 1 → f x ∈ Set.Icc (min (f 0) (min (f 1) (f (-b/(2*a))))) (max (f 0) (f 1))) ∧
  ((-2*a ≤ b ∧ b ≤ 0 ∧ a + b + c ≥ c) →
    Set.Icc (-b^2/(4*a) + c) (a + b + c) = Set.Icc (min (f 0) (min (f 1) (f (-b/(2*a))))) (max (f 0) (f 1))) ∧
  ((b < -2*a ∨ b > 0) →
    Set.Icc c (a + b + c) = Set.Icc (min (f 0) (min (f 1) (f (-b/(2*a))))) (max (f 0) (f 1))) :=
by sorry

end quadratic_range_on_unit_interval_l602_60283


namespace collinear_points_imply_b_value_l602_60211

/-- Given three points in 2D space, this function checks if they are collinear -/
def are_collinear (x1 y1 x2 y2 x3 y3 : ℚ) : Prop :=
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

/-- Theorem stating that if the given points are collinear, then b = -3/13 -/
theorem collinear_points_imply_b_value (b : ℚ) :
  are_collinear 4 (-6) ((-b) + 3) 4 (3*b + 4) 3 → b = -3/13 := by
  sorry

end collinear_points_imply_b_value_l602_60211


namespace express_y_in_terms_of_x_l602_60225

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x + y = 5) : y = 5 - 2 * x := by
  sorry

end express_y_in_terms_of_x_l602_60225


namespace lesser_number_proof_l602_60212

theorem lesser_number_proof (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : min a b = 25 := by
  sorry

end lesser_number_proof_l602_60212


namespace triangle_dot_product_l602_60213

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 3^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 3^2 ∧
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define point M
def point_M (A B M : ℝ × ℝ) : Prop :=
  (M.1 - B.1) = 2 * (A.1 - M.1) ∧
  (M.2 - B.2) = 2 * (A.2 - M.2)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem triangle_dot_product 
  (A B C M : ℝ × ℝ) 
  (h1 : triangle_ABC A B C) 
  (h2 : point_M A B M) : 
  dot_product (C.1 - M.1, C.2 - M.2) (C.1 - B.1, C.2 - B.2) = 3 := by
  sorry

end triangle_dot_product_l602_60213


namespace smallest_covering_triangular_l602_60232

/-- Triangular number function -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Function to determine the row of a number on the snaking board -/
def board_row (n : ℕ) : ℕ :=
  if n % 20 ≤ 10 then (n - 1) / 10 + 1 else (n + 9) / 10

/-- Theorem stating that 91 is the smallest triangular number that covers all rows -/
theorem smallest_covering_triangular : 
  (∀ k < 13, ∃ r ≤ 10, ∀ i ≤ 10, board_row (triangular k) ≠ i) ∧
  (∀ i ≤ 10, ∃ k ≤ 13, board_row (triangular k) = i) :=
sorry

end smallest_covering_triangular_l602_60232


namespace smallest_cookie_count_l602_60209

theorem smallest_cookie_count (b : ℕ) : 
  b > 0 ∧
  b % 5 = 4 ∧
  b % 6 = 3 ∧
  b % 8 = 5 ∧
  b % 9 = 7 ∧
  (∀ c : ℕ, c > 0 ∧ c % 5 = 4 ∧ c % 6 = 3 ∧ c % 8 = 5 ∧ c % 9 = 7 → b ≤ c) →
  b = 909 := by
sorry

end smallest_cookie_count_l602_60209


namespace partial_fraction_decomposition_l602_60247

theorem partial_fraction_decomposition :
  ∃ (A B C : ℚ),
    (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
      (x^2 + 5*x - 2) / (x^3 - x) = A / x + (B*x + C) / (x^2 - 1)) ∧
    A = 2 ∧ B = -1 ∧ C = 5 := by
  sorry

end partial_fraction_decomposition_l602_60247


namespace quentavious_nickels_l602_60210

/-- Represents the exchange of nickels for gum pieces -/
def exchange_nickels_for_gum (initial_nickels : ℕ) (gum_pieces : ℕ) (remaining_nickels : ℕ) : Prop :=
  initial_nickels = (gum_pieces / 2) + remaining_nickels

/-- Proves that given the conditions, Quentavious must have started with 5 nickels -/
theorem quentavious_nickels : 
  ∀ (initial_nickels : ℕ),
    exchange_nickels_for_gum initial_nickels 6 2 →
    initial_nickels = 5 := by
  sorry


end quentavious_nickels_l602_60210


namespace ratio_sum_to_y_l602_60244

theorem ratio_sum_to_y (w x y : ℝ) (hw_x : w / x = 1 / 3) (hw_y : w / y = 2 / 3) :
  (x + y) / y = 3 := by sorry

end ratio_sum_to_y_l602_60244


namespace boxes_opened_is_twelve_l602_60208

/-- Calculates the number of boxes opened given the number of samples per box,
    the number of customers who tried a sample, and the number of samples left over. -/
def boxes_opened (samples_per_box : ℕ) (customers : ℕ) (samples_left : ℕ) : ℕ :=
  (customers + samples_left) / samples_per_box

/-- Proves that given the conditions, the number of boxes opened is 12. -/
theorem boxes_opened_is_twelve :
  boxes_opened 20 235 5 = 12 := by
  sorry

end boxes_opened_is_twelve_l602_60208


namespace fashion_show_duration_l602_60243

def fashion_show_time (num_models : ℕ) (bathing_suits_per_model : ℕ) (evening_wear_per_model : ℕ) (time_per_trip : ℕ) : ℕ :=
  num_models * (bathing_suits_per_model + evening_wear_per_model) * time_per_trip

theorem fashion_show_duration :
  fashion_show_time 6 2 3 2 = 60 := by
  sorry

end fashion_show_duration_l602_60243


namespace window_width_is_30_l602_60270

/-- Represents the width of a pane of glass in the window -/
def pane_width : ℝ := 6

/-- Represents the height of a pane of glass in the window -/
def pane_height : ℝ := 3 * pane_width

/-- Represents the width of the borders around and between panes -/
def border_width : ℝ := 3

/-- Represents the number of columns of panes in the window -/
def num_columns : ℕ := 3

/-- Represents the number of rows of panes in the window -/
def num_rows : ℕ := 2

/-- Calculates the total width of the window -/
def window_width : ℝ := num_columns * pane_width + (num_columns + 1) * border_width

/-- Theorem stating that the width of the rectangular window is 30 inches -/
theorem window_width_is_30 : window_width = 30 := by sorry

end window_width_is_30_l602_60270


namespace geometric_sequence_sum_l602_60246

/-- A geometric sequence with positive terms satisfying a specific condition -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ r, ∀ n, a (n + 1) = r * a n) ∧
  (a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100)

/-- The sum of the 4th and 6th terms of the geometric sequence is 10 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) : a 4 + a 6 = 10 := by
  sorry

end geometric_sequence_sum_l602_60246


namespace sin_830_equality_l602_60226

theorem sin_830_equality (n : ℤ) : 
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.sin (830 * π / 180) → 
  n = 70 ∨ n = 110 := by
  sorry

end sin_830_equality_l602_60226


namespace min_detectors_for_specific_board_and_ship_l602_60293

/-- Represents a grid board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a ship -/
structure Ship :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a detector placement strategy -/
def DetectorStrategy := ℕ → ℕ → Bool

/-- Checks if a detector strategy can determine the ship's position -/
def can_determine_position (b : Board) (s : Ship) (strategy : DetectorStrategy) : Prop :=
  sorry

/-- The minimum number of detectors required -/
def min_detectors (b : Board) (s : Ship) : ℕ :=
  sorry

theorem min_detectors_for_specific_board_and_ship :
  let b : Board := ⟨2015, 2015⟩
  let s : Ship := ⟨1500, 1500⟩
  min_detectors b s = 1030 :=
sorry

end min_detectors_for_specific_board_and_ship_l602_60293


namespace spider_dressing_theorem_l602_60284

/-- The number of legs of the spider -/
def num_legs : ℕ := 10

/-- The number of items per leg -/
def items_per_leg : ℕ := 3

/-- The total number of items -/
def total_items : ℕ := num_legs * items_per_leg

/-- The number of possible arrangements of items for one leg -/
def arrangements_per_leg : ℕ := Nat.factorial items_per_leg

theorem spider_dressing_theorem :
  (Nat.factorial total_items) / (arrangements_per_leg ^ num_legs) = 
  (Nat.factorial (num_legs * items_per_leg)) / (Nat.factorial items_per_leg ^ num_legs) := by
  sorry

end spider_dressing_theorem_l602_60284


namespace f_of_3_equals_4_l602_60257

-- Define the function f
def f : ℝ → ℝ := fun x => (x - 1)^2

-- State the theorem
theorem f_of_3_equals_4 : f 3 = 4 := by
  sorry

end f_of_3_equals_4_l602_60257


namespace angles_on_line_y_equals_x_l602_60224

/-- The set of angles whose terminal side lies on the line y = x -/
def anglesOnLineYEqualsX : Set ℝ :=
  {α | ∃ k : ℤ, α = k * Real.pi + Real.pi / 4}

/-- The line y = x -/
def lineYEqualsX (x : ℝ) : ℝ := x

theorem angles_on_line_y_equals_x :
  {α : ℝ | ∃ (x : ℝ), Real.cos α * x = lineYEqualsX x ∧ Real.sin α * x = lineYEqualsX x} =
  anglesOnLineYEqualsX := by sorry

end angles_on_line_y_equals_x_l602_60224


namespace third_blue_after_fifth_probability_l602_60275

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 5

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 3

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + blue_marbles

/-- The probability of drawing the third blue marble after the fifth draw -/
def probability_third_blue_after_fifth : ℚ := 23 / 28

theorem third_blue_after_fifth_probability :
  probability_third_blue_after_fifth = 
    (Nat.choose 5 2 * Nat.choose 3 1 + 
     Nat.choose 5 1 * Nat.choose 3 2 + 
     Nat.choose 5 0 * Nat.choose 3 3) / 
    Nat.choose total_marbles blue_marbles :=
by sorry

end third_blue_after_fifth_probability_l602_60275


namespace prob_odd_top_face_l602_60268

/-- The number of sides on the die -/
def num_sides : ℕ := 12

/-- The total number of dots on the die initially -/
def total_dots : ℕ := (num_sides * (num_sides + 1)) / 2

/-- The number of ways to choose 2 dots from the total -/
def ways_to_choose_two_dots : ℕ := total_dots.choose 2

/-- The probability of rolling a specific face -/
def prob_single_face : ℚ := 1 / num_sides

/-- The sum of even numbers from 2 to 12 -/
def sum_even_faces : ℕ := 2 + 4 + 6 + 8 + 10 + 12

/-- Theorem: The probability of rolling an odd number of dots on the top face
    of a 12-sided die, after randomly removing two dots, is 7/3003 -/
theorem prob_odd_top_face : 
  (prob_single_face * (2 * sum_even_faces : ℚ)) / ways_to_choose_two_dots = 7 / 3003 :=
sorry

end prob_odd_top_face_l602_60268


namespace f_derivative_at_one_l602_60203

def f (x : ℝ) := (x - 2)^2

theorem f_derivative_at_one : 
  deriv f 1 = -2 := by sorry

end f_derivative_at_one_l602_60203


namespace max_value_of_s_l602_60242

theorem max_value_of_s (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 10)
  (sum_prod_eq : p*q + p*r + p*s + q*r + q*s + r*s = 20) :
  s ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end max_value_of_s_l602_60242


namespace problem_solution_l602_60299

def p (x a : ℝ) : Prop := (x - 3*a) * (x - a) < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) ≤ 0

theorem problem_solution :
  (∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x < 3)) ∧
  (∀ a : ℝ, (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x) ↔ (1 ≤ a ∧ a ≤ 2)) :=
sorry

end problem_solution_l602_60299


namespace basketball_game_proof_l602_60251

def basketball_game (basket_points : ℕ) (matthew_points : ℕ) (shawn_points : ℕ) : Prop :=
  ∃ (matthew_baskets shawn_baskets : ℕ),
    basket_points * matthew_baskets = matthew_points ∧
    basket_points * shawn_baskets = shawn_points ∧
    matthew_baskets + shawn_baskets = 5

theorem basketball_game_proof :
  basketball_game 3 9 6 := by
  sorry

end basketball_game_proof_l602_60251


namespace stratified_sampling_second_grade_l602_60285

theorem stratified_sampling_second_grade 
  (total_sample : ℕ) 
  (ratio_first : ℕ) 
  (ratio_second : ℕ) 
  (ratio_third : ℕ) :
  total_sample = 50 →
  ratio_first = 3 →
  ratio_second = 3 →
  ratio_third = 4 →
  (ratio_second : ℚ) / (ratio_first + ratio_second + ratio_third : ℚ) * total_sample = 15 :=
by
  sorry

end stratified_sampling_second_grade_l602_60285


namespace margarita_vs_ricciana_distance_l602_60241

/-- Represents the total distance of a long jump, including running and jumping. -/
structure LongJump where
  run : ℕ
  jump : ℕ

/-- Calculates the total distance of a long jump. -/
def total_distance (lj : LongJump) : ℕ := lj.run + lj.jump

theorem margarita_vs_ricciana_distance : 
  let ricciana : LongJump := { run := 20, jump := 4 }
  let margarita : LongJump := { run := 18, jump := 2 * ricciana.jump - 1 }
  total_distance margarita - total_distance ricciana = 1 := by
sorry

end margarita_vs_ricciana_distance_l602_60241


namespace daniela_shopping_cost_l602_60294

-- Define the original prices and discount rates
def shoe_price : ℝ := 50
def dress_price : ℝ := 100
def shoe_discount : ℝ := 0.4
def dress_discount : ℝ := 0.2

-- Define the number of items purchased
def num_shoes : ℕ := 2
def num_dresses : ℕ := 1

-- Calculate the discounted prices
def discounted_shoe_price : ℝ := shoe_price * (1 - shoe_discount)
def discounted_dress_price : ℝ := dress_price * (1 - dress_discount)

-- Calculate the total cost
def total_cost : ℝ := num_shoes * discounted_shoe_price + num_dresses * discounted_dress_price

-- Theorem statement
theorem daniela_shopping_cost : total_cost = 140 := by
  sorry

end daniela_shopping_cost_l602_60294


namespace divisibility_condition_l602_60235

theorem divisibility_condition (n : ℤ) : 
  (∃ a : ℤ, n - 4 = 6 * a) → 
  (∃ b : ℤ, n - 8 = 10 * b) → 
  (∃ k : ℤ, n = 30 * k - 2) :=
by sorry

end divisibility_condition_l602_60235


namespace circle_area_with_diameter_8_l602_60296

/-- The area of a circle with diameter 8 meters is 16π square meters -/
theorem circle_area_with_diameter_8 (π : ℝ) :
  let diameter : ℝ := 8
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 16 * π := by
  sorry

end circle_area_with_diameter_8_l602_60296


namespace expression_simplification_system_of_equations_solution_l602_60229

-- Part 1: Simplifying the Expression
theorem expression_simplification :
  (Real.sqrt 6 - Real.sqrt (8/3)) * Real.sqrt 3 - (2 + Real.sqrt 3) * (2 - Real.sqrt 3) = Real.sqrt 2 - 1 := by
  sorry

-- Part 2: Solving the System of Equations
theorem system_of_equations_solution :
  ∃ (x y : ℝ), 2*x - 5*y = 7 ∧ 3*x + 2*y = 1 ∧ x = 1 ∧ y = -1 := by
  sorry

end expression_simplification_system_of_equations_solution_l602_60229


namespace final_sum_theorem_l602_60260

theorem final_sum_theorem (a b S : ℝ) (h : a + b = S) :
  2 * (a + 3) + 2 * (b + 3) = 2 * S + 12 := by
  sorry

end final_sum_theorem_l602_60260


namespace conic_section_k_range_l602_60250

/-- Represents a conic section of the form x^2/2 + y^2/k = 1 -/
structure ConicSection (k : ℝ) where
  equation : ∀ (x y : ℝ), x^2/2 + y^2/k = 1

/-- Proposition p: The conic section is an ellipse with foci on the x-axis -/
def is_ellipse_x_foci (k : ℝ) : Prop :=
  0 < k ∧ k < 2

/-- Proposition q: The eccentricity of the conic section is in the interval (√2, √3) -/
def eccentricity_in_range (k : ℝ) : Prop :=
  let e := Real.sqrt ((2 - k) / 2)
  Real.sqrt 2 < e ∧ e < Real.sqrt 3

/-- The main theorem -/
theorem conic_section_k_range (k : ℝ) (E : ConicSection k) :
  (¬is_ellipse_x_foci k) ∧ eccentricity_in_range k → -4 < k ∧ k < -2 :=
sorry

end conic_section_k_range_l602_60250


namespace fourth_proportional_l602_60276

theorem fourth_proportional (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ x : ℝ, x > 0 ∧ a * x = b * c := by
sorry

end fourth_proportional_l602_60276


namespace roots_imply_f_value_l602_60265

-- Define the polynomials g and f
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 2*x + 15
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + x^3 + b*x^2 + 75*x + c

-- State the theorem
theorem roots_imply_f_value (a b c : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    g a r₁ = 0 ∧ g a r₂ = 0 ∧ g a r₃ = 0 ∧
    f b c r₁ = 0 ∧ f b c r₂ = 0 ∧ f b c r₃ = 0) →
  f b c (-1) = -2773 := by
  sorry

end roots_imply_f_value_l602_60265


namespace bus_journey_time_l602_60272

/-- Calculates the total time for a bus journey with two different speeds -/
theorem bus_journey_time (total_distance : ℝ) (speed1 speed2 : ℝ) (distance1 : ℝ) : 
  total_distance = 250 →
  speed1 = 40 →
  speed2 = 60 →
  distance1 = 148 →
  (distance1 / speed1) + ((total_distance - distance1) / speed2) = 5.4 := by
  sorry

end bus_journey_time_l602_60272


namespace colored_pencils_total_l602_60215

theorem colored_pencils_total (madeline_pencils : ℕ) 
  (h1 : madeline_pencils = 63)
  (h2 : ∃ cheryl_pencils : ℕ, cheryl_pencils = 2 * madeline_pencils)
  (h3 : ∃ cyrus_pencils : ℕ, 3 * cyrus_pencils = cheryl_pencils) :
  ∃ total_pencils : ℕ, total_pencils = madeline_pencils + cheryl_pencils + cyrus_pencils ∧ total_pencils = 231 :=
by sorry

end colored_pencils_total_l602_60215


namespace ellipse_eccentricity_l602_60218

-- Define the complex polynomial
def p (z : ℂ) : ℂ := (z - 1) * (z^2 + 2*z + 4) * (z^2 + 4*z + 6)

-- Define the set of solutions
def S : Set ℂ := {z : ℂ | p z = 0}

-- Define the ellipse passing through the solutions
def E : Set (ℝ × ℝ) :=
  {xy : ℝ × ℝ | ∃ z ∈ S, (xy.1 = z.re ∧ xy.2 = z.im)}

-- State the theorem
theorem ellipse_eccentricity :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  E = {xy : ℝ × ℝ | (xy.1 + 9/10)^2 / (361/100) + xy.2^2 / (361/120) = 1} ∧
  (a^2 - b^2) / a^2 = 1/6 := by
  sorry

end ellipse_eccentricity_l602_60218


namespace total_pokemon_cards_l602_60263

def melanie_cards : ℝ := 7.5
def benny_cards : ℝ := 9
def sandy_cards : ℝ := 5.2
def jessica_cards : ℝ := 12.8

theorem total_pokemon_cards :
  (melanie_cards * 12 + benny_cards * 12 + sandy_cards * 12 + jessica_cards * 12) = 414 := by
  sorry

end total_pokemon_cards_l602_60263


namespace money_left_l602_60231

/-- The amount of money Mrs. Hilt had initially, in cents. -/
def initial_amount : ℕ := 15

/-- The cost of the pencil, in cents. -/
def pencil_cost : ℕ := 11

/-- The theorem stating how much money Mrs. Hilt had left after buying the pencil. -/
theorem money_left : initial_amount - pencil_cost = 4 := by
  sorry

end money_left_l602_60231


namespace algebraic_expression_simplification_l602_60253

theorem algebraic_expression_simplification (x : ℝ) :
  x = 2 * Real.cos (45 * π / 180) - 1 →
  (x / (x^2 + 2*x + 1) - 1 / (2*x + 2)) / ((x - 1) / (4*x + 4)) = Real.sqrt 2 := by
  sorry

end algebraic_expression_simplification_l602_60253


namespace age_difference_is_nine_l602_60295

/-- Represents a year in the 19th or 20th century -/
structure Year where
  century : Nat
  tens : Nat
  ones : Nat
  h : century ∈ [18, 19] ∧ tens < 10 ∧ ones < 10

/-- The age of a person at a given meeting year -/
def age (birth : Year) (meetingYear : Nat) : Nat :=
  meetingYear - (birth.century * 100 + birth.tens * 10 + birth.ones)

theorem age_difference_is_nine :
  ∀ (peterBirth : Year) (paulBirth : Year) (meetingYear : Nat),
    peterBirth.century = 18 →
    paulBirth.century = 19 →
    age peterBirth meetingYear = peterBirth.century + peterBirth.tens + peterBirth.ones + 9 →
    age paulBirth meetingYear = paulBirth.century + paulBirth.tens + paulBirth.ones + 10 →
    age peterBirth meetingYear - age paulBirth meetingYear = 9 := by
  sorry

#check age_difference_is_nine

end age_difference_is_nine_l602_60295


namespace oz_words_lost_l602_60288

/-- The number of letters in the Oz alphabet -/
def alphabet_size : ℕ := 68

/-- The maximum number of letters allowed in a word -/
def max_word_length : ℕ := 2

/-- The position of the forbidden letter in the alphabet -/
def forbidden_letter_position : ℕ := 7

/-- Calculates the number of words lost due to prohibiting a letter -/
def words_lost (alphabet_size : ℕ) (max_word_length : ℕ) (forbidden_letter_position : ℕ) : ℕ :=
  let one_letter_words_lost := 1
  let two_letter_words_lost := 2 * (alphabet_size - 1)
  one_letter_words_lost + two_letter_words_lost

/-- The theorem stating the number of words lost in Oz -/
theorem oz_words_lost :
  words_lost alphabet_size max_word_length forbidden_letter_position = 135 := by
  sorry

end oz_words_lost_l602_60288


namespace trig_expression_equality_l602_60201

theorem trig_expression_equality : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 1 - Real.sqrt 3 := by
  sorry

end trig_expression_equality_l602_60201


namespace rectangle_area_difference_l602_60239

theorem rectangle_area_difference (l w : ℕ) : 
  (l + w = 25) →  -- Perimeter condition: 2l + 2w = 50 simplified
  (∃ (l' w' : ℕ), l' + w' = 25 ∧ l' * w' = 156) ∧  -- Existence of max area
  (∃ (l'' w'' : ℕ), l'' + w'' = 25 ∧ l'' * w'' = 24) ∧  -- Existence of min area
  (∀ (l''' w''' : ℕ), l''' + w''' = 25 → l''' * w''' ≤ 156) ∧  -- Max area condition
  (∀ (l'''' w'''' : ℕ), l'''' + w'''' = 25 → l'''' * w'''' ≥ 24) →  -- Min area condition
  156 - 24 = 132  -- Difference between max and min areas
  := by sorry

end rectangle_area_difference_l602_60239


namespace infinite_geometric_series_ratio_l602_60278

/-- For an infinite geometric series with first term 512 and sum 2048, the common ratio is 3/4 -/
theorem infinite_geometric_series_ratio (a : ℝ) (S : ℝ) (r : ℝ) : 
  a = 512 → S = 2048 → S = a / (1 - r) → r = 3/4 := by
  sorry

end infinite_geometric_series_ratio_l602_60278


namespace point_P_coordinates_l602_60289

/-- The mapping f that transforms a point (x, y) to (x+y, 2x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, 2 * p.1 - p.2)

/-- Theorem stating that if P is mapped to (5, 1) under f, then P has coordinates (2, 3) -/
theorem point_P_coordinates (P : ℝ × ℝ) (h : f P = (5, 1)) : P = (2, 3) := by
  sorry


end point_P_coordinates_l602_60289


namespace quadratic_function_unique_l602_60240

/-- A quadratic function is a function of the form f(x) = ax² + bx + c where a ≠ 0 -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_unique
  (f : ℝ → ℝ)
  (hf : QuadraticFunction f)
  (h1 : f 0 = 1)
  (h2 : ∀ x, f (x + 1) - f x = 2 * x) :
  ∀ x, f x = x^2 - x + 1 := by
  sorry

end quadratic_function_unique_l602_60240


namespace cubic_equation_value_l602_60274

theorem cubic_equation_value (x : ℝ) (h : x^2 + x - 2 = 0) :
  x^3 + 2*x^2 - x + 2007 = 2008 := by
  sorry

end cubic_equation_value_l602_60274


namespace cubic_equation_roots_l602_60249

theorem cubic_equation_roots :
  ∃ (r₁ r₂ r₃ : ℝ), r₁ < 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧
  ∀ x : ℝ, x^3 - 2*x^2 - 5*x + 6 = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃) :=
by sorry

end cubic_equation_roots_l602_60249


namespace tangent_line_at_point_one_two_l602_60200

-- Define the curve
def f (x : ℝ) : ℝ := -x^3 + 3*x^2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x

-- Theorem statement
theorem tangent_line_at_point_one_two :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 3*x - 1 :=
by sorry

end tangent_line_at_point_one_two_l602_60200


namespace clothing_prices_correct_l602_60264

/-- Represents the price of a clothing item -/
structure ClothingPrice where
  purchase : ℝ
  marked : ℝ

/-- Solves the clothing price problem given the conditions -/
def solve_clothing_prices (total_paid markup_percent discount_a discount_b total_marked : ℝ) : 
  (ClothingPrice × ClothingPrice) :=
  let a : ClothingPrice := { 
    purchase := 50,
    marked := 70
  }
  let b : ClothingPrice := {
    purchase := 100,
    marked := 140
  }
  (a, b)

/-- Theorem stating the correctness of the solution -/
theorem clothing_prices_correct 
  (total_paid : ℝ)
  (markup_percent : ℝ)
  (discount_a : ℝ)
  (discount_b : ℝ)
  (total_marked : ℝ)
  (h1 : total_paid = 182)
  (h2 : markup_percent = 40)
  (h3 : discount_a = 80)
  (h4 : discount_b = 90)
  (h5 : total_marked = 210) :
  let (a, b) := solve_clothing_prices total_paid markup_percent discount_a discount_b total_marked
  (a.purchase = 50 ∧ 
   a.marked = 70 ∧ 
   b.purchase = 100 ∧ 
   b.marked = 140 ∧
   a.marked + b.marked = total_marked ∧
   (discount_a / 100) * a.marked + (discount_b / 100) * b.marked = total_paid ∧
   a.marked = a.purchase * (1 + markup_percent / 100) ∧
   b.marked = b.purchase * (1 + markup_percent / 100)) := by
  sorry


end clothing_prices_correct_l602_60264


namespace only_eq1_has_zero_constant_term_l602_60261

-- Define the equations
def eq1 (x : ℝ) := x^2 + x = 0
def eq2 (x : ℝ) := 2*x^2 - x - 12 = 0
def eq3 (x : ℝ) := 2*(x^2 - 1) = 3*(x - 1)
def eq4 (x : ℝ) := 2*(x^2 + 1) = x + 4

-- Define a function to check if an equation has zero constant term
def has_zero_constant_term (eq : ℝ → Prop) : Prop :=
  ∃ a b : ℝ, ∀ x, eq x ↔ a*x^2 + b*x = 0

-- Theorem statement
theorem only_eq1_has_zero_constant_term :
  has_zero_constant_term eq1 ∧
  ¬has_zero_constant_term eq2 ∧
  ¬has_zero_constant_term eq3 ∧
  ¬has_zero_constant_term eq4 :=
sorry

end only_eq1_has_zero_constant_term_l602_60261


namespace cocoa_powder_calculation_l602_60248

theorem cocoa_powder_calculation (already_has : ℕ) (still_needs : ℕ) 
  (h1 : already_has = 259)
  (h2 : still_needs = 47) :
  already_has + still_needs = 306 := by
sorry

end cocoa_powder_calculation_l602_60248


namespace multiply_special_form_l602_60217

theorem multiply_special_form (x : ℝ) : 
  (x^4 + 18*x^2 + 324) * (x^2 - 18) = x^6 - 5832 := by
  sorry

end multiply_special_form_l602_60217


namespace repeating_decimal_to_fraction_l602_60214

theorem repeating_decimal_to_fraction : 
  ∃ (x : ℚ), x = 3 + 45 / 99 ∧ x = 38 / 11 := by
  sorry

end repeating_decimal_to_fraction_l602_60214


namespace extreme_value_implies_a_minus_b_l602_60292

/-- A function f(x) with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

/-- The derivative of f(x) with respect to x -/
def f_deriv (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + b

theorem extreme_value_implies_a_minus_b (a b : ℝ) :
  (f a b (-1) = 0) →  -- f(x) has value 0 at x = -1
  (f_deriv a b (-1) = 0) →  -- f'(x) = 0 at x = -1 (condition for extreme value)
  (a - b = -7) :=
sorry

end extreme_value_implies_a_minus_b_l602_60292


namespace bob_has_62_pennies_l602_60219

/-- The number of pennies Alex currently has -/
def a : ℕ := sorry

/-- The number of pennies Bob currently has -/
def b : ℕ := sorry

/-- Condition 1: If Alex gives Bob two pennies, Bob will have four times as many pennies as Alex has left -/
axiom condition1 : b + 2 = 4 * (a - 2)

/-- Condition 2: If Bob gives Alex two pennies, Bob will have three times as many pennies as Alex has -/
axiom condition2 : b - 2 = 3 * (a + 2)

/-- Theorem: Bob currently has 62 pennies -/
theorem bob_has_62_pennies : b = 62 := by sorry

end bob_has_62_pennies_l602_60219


namespace happy_tail_dog_count_l602_60207

theorem happy_tail_dog_count :
  let jump : ℕ := 65
  let fetch : ℕ := 40
  let bark : ℕ := 45
  let jump_fetch : ℕ := 25
  let fetch_bark : ℕ := 20
  let jump_bark : ℕ := 23
  let all_three : ℕ := 15
  let none : ℕ := 12
  
  -- Dogs that can jump and fetch, but not bark
  let jump_fetch_only : ℕ := jump_fetch - all_three
  -- Dogs that can fetch and bark, but not jump
  let fetch_bark_only : ℕ := fetch_bark - all_three
  -- Dogs that can jump and bark, but not fetch
  let jump_bark_only : ℕ := jump_bark - all_three
  
  -- Dogs that can only jump
  let jump_only : ℕ := jump - (jump_fetch_only + jump_bark_only + all_three)
  -- Dogs that can only fetch
  let fetch_only : ℕ := fetch - (jump_fetch_only + fetch_bark_only + all_three)
  -- Dogs that can only bark
  let bark_only : ℕ := bark - (jump_bark_only + fetch_bark_only + all_three)
  
  -- Total number of dogs
  jump_only + fetch_only + bark_only + jump_fetch_only + fetch_bark_only + jump_bark_only + all_three + none = 109 :=
by
  sorry

end happy_tail_dog_count_l602_60207


namespace union_of_M_and_N_l602_60273

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | -3 ≤ x ∧ x ≤ 3} := by sorry

end union_of_M_and_N_l602_60273


namespace strip_overlap_area_l602_60238

/-- The area of overlap for three strips of width 2 intersecting at angle θ -/
theorem strip_overlap_area (θ : Real) (h1 : θ ≠ 0) (h2 : θ ≠ π / 2) : Real :=
  let strip_width : Real := 2
  let overlap_area := 8 * Real.sin θ
  overlap_area

#check strip_overlap_area

end strip_overlap_area_l602_60238


namespace real_roots_condition_l602_60204

theorem real_roots_condition (a : ℝ) :
  (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ a ≤ 2 := by
  sorry

end real_roots_condition_l602_60204


namespace subset_implies_a_equals_one_l602_60228

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

-- State the theorem
theorem subset_implies_a_equals_one (a : ℝ) :
  A a ⊆ B a → a = 1 := by
  sorry

end subset_implies_a_equals_one_l602_60228


namespace consecutive_composites_l602_60220

theorem consecutive_composites (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, (∀ i : ℕ, i < n → ¬ Nat.Prime (k + i + 2)) ∧
           (k + n + 1 < 4^(n + 1)) := by
  sorry

end consecutive_composites_l602_60220


namespace function_inequality_implies_m_bound_l602_60216

/-- Given two functions f and g, where f(x) = |x-3| and g(x) = -|x-7| + m,
    if f(x) > g(x) for all real x, then m < 4 -/
theorem function_inequality_implies_m_bound
  (f g : ℝ → ℝ)
  (hf : ∀ x, f x = |x - 3|)
  (hg : ∀ x, g x = -|x - 7| + m)
  (h_above : ∀ x, f x > g x) :
  m < 4 := by
  sorry

end function_inequality_implies_m_bound_l602_60216


namespace angle_complement_l602_60290

theorem angle_complement (given_angle : ℝ) (straight_angle : ℝ) :
  given_angle = 13 →
  straight_angle = 180 →
  (straight_angle - (13 * given_angle)) = 11 := by
  sorry

end angle_complement_l602_60290


namespace arrange_athletes_eq_144_l602_60223

/-- The number of ways to arrange 6 athletes on 6 tracks with restrictions -/
def arrange_athletes : ℕ :=
  let total_tracks : ℕ := 6
  let total_athletes : ℕ := 6
  let restricted_tracks_for_A : ℕ := 2
  let possible_tracks_for_B : ℕ := 2
  let remaining_athletes : ℕ := total_athletes - 2

  possible_tracks_for_B *
  (total_tracks - restricted_tracks_for_A - 1) *
  Nat.factorial remaining_athletes

theorem arrange_athletes_eq_144 : arrange_athletes = 144 := by
  sorry

end arrange_athletes_eq_144_l602_60223


namespace monday_visitors_l602_60282

/-- Represents the number of visitors to a library in a week -/
structure LibraryVisitors where
  monday : ℕ
  tuesday : ℕ
  remainingDays : ℕ

/-- Theorem: Given the conditions of the library visitors problem, prove that there were 50 visitors on Monday -/
theorem monday_visitors (v : LibraryVisitors) : v.monday = 50 :=
  by
  have h1 : v.tuesday = 2 * v.monday := by sorry
  have h2 : v.remainingDays = 5 * 20 := by sorry
  have h3 : v.monday + v.tuesday + v.remainingDays = 250 := by sorry
  sorry


end monday_visitors_l602_60282


namespace candy_distribution_l602_60297

theorem candy_distribution (total_candy : ℕ) (total_bags : ℕ) (chocolate_heart_bags : ℕ) (chocolate_kiss_bags : ℕ)
  (h1 : total_candy = 63)
  (h2 : total_bags = 9)
  (h3 : chocolate_heart_bags = 2)
  (h4 : chocolate_kiss_bags = 3)
  (h5 : total_candy % total_bags = 0) :
  let candy_per_bag := total_candy / total_bags
  let chocolate_bags := chocolate_heart_bags + chocolate_kiss_bags
  let non_chocolate_bags := total_bags - chocolate_bags
  non_chocolate_bags * candy_per_bag = 28 := by
  sorry

end candy_distribution_l602_60297
