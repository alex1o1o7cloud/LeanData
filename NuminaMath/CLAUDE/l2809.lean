import Mathlib

namespace contractor_absence_l2809_280925

/-- Proves that given the specified contract conditions, the contractor was absent for 10 days -/
theorem contractor_absence (total_days : ℕ) (daily_pay : ℚ) (daily_fine : ℚ) (total_received : ℚ)
  (h_total_days : total_days = 30)
  (h_daily_pay : daily_pay = 25)
  (h_daily_fine : daily_fine = 7.5)
  (h_total_received : total_received = 425) :
  ∃ (days_worked days_absent : ℕ),
    days_worked + days_absent = total_days ∧
    days_worked * daily_pay - days_absent * daily_fine = total_received ∧
    days_absent = 10 :=
by sorry

end contractor_absence_l2809_280925


namespace largest_angle_in_special_triangle_l2809_280937

-- Define the triangle DEF
def triangle_DEF : Set (ℝ × ℝ) := sorry

-- Define that the triangle is isosceles
def is_isosceles (t : Set (ℝ × ℝ)) : Prop := sorry

-- Define that the triangle is acute
def is_acute (t : Set (ℝ × ℝ)) : Prop := sorry

-- Define the measure of an angle
def angle_measure (t : Set (ℝ × ℝ)) (v : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem largest_angle_in_special_triangle :
  ∀ (DEF : Set (ℝ × ℝ)),
    is_isosceles DEF →
    is_acute DEF →
    angle_measure DEF (0, 0) = 30 →
    (∃ (v : ℝ × ℝ), v ∈ DEF ∧ angle_measure DEF v = 75 ∧
      ∀ (w : ℝ × ℝ), w ∈ DEF → angle_measure DEF w ≤ 75) :=
by sorry

end largest_angle_in_special_triangle_l2809_280937


namespace regression_line_equation_l2809_280975

/-- Proves that the regression line equation is y = -x + 3 given the conditions -/
theorem regression_line_equation (b : ℝ) :
  (∃ (x y : ℝ), y = b * x + 3 ∧ (1, 2) = (x, y)) →
  (∀ (x y : ℝ), y = -x + 3) :=
by sorry

end regression_line_equation_l2809_280975


namespace zeros_of_composition_l2809_280945

/-- Given functions f and g, prove that the zeros of their composition h are ±√2 -/
theorem zeros_of_composition (f g h : ℝ → ℝ) :
  (∀ x, f x = 2 * x - 4) →
  (∀ x, g x = x^2) →
  (∀ x, h x = f (g x)) →
  {x : ℝ | h x = 0} = {-Real.sqrt 2, Real.sqrt 2} := by
  sorry

end zeros_of_composition_l2809_280945


namespace max_triangle_area_l2809_280962

/-- The maximum area of a triangle ABC with side length constraints -/
theorem max_triangle_area (AB BC CA : ℝ) 
  (hAB : 0 ≤ AB ∧ AB ≤ 1)
  (hBC : 1 ≤ BC ∧ BC ≤ 2)
  (hCA : 2 ≤ CA ∧ CA ≤ 3) :
  ∃ (area : ℝ), area ≤ 1 ∧ 
  ∀ (a : ℝ), (∃ (x y z : ℝ), 
    0 ≤ x ∧ x ≤ 1 ∧
    1 ≤ y ∧ y ≤ 2 ∧
    2 ≤ z ∧ z ≤ 3 ∧
    a = (x + y + z) / 2 * ((x + y + z) / 2 - x) * ((x + y + z) / 2 - y) * ((x + y + z) / 2 - z)) →
  a ≤ area :=
sorry

end max_triangle_area_l2809_280962


namespace trigonometric_inequality_l2809_280967

theorem trigonometric_inequality (x : ℝ) : 
  0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 
  2 * Real.cos x ≤ Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)) ∧
  Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)) ≤ Real.sqrt 2 →
  Real.pi / 4 ≤ x ∧ x ≤ 7 * Real.pi / 4 := by
sorry

end trigonometric_inequality_l2809_280967


namespace perpendicular_lines_a_value_l2809_280989

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem perpendicular_lines_a_value (a : ℝ) :
  let l1 : ℝ → ℝ → Prop := λ x y => a * x + (3 - a) * y + 1 = 0
  let l2 : ℝ → ℝ → Prop := λ x y => x - 2 * y = 0
  let m1 : ℝ := a / (a - 3)  -- slope of l1
  let m2 : ℝ := 1 / 2        -- slope of l2
  perpendicular m1 m2 → a = 2 := by
sorry

end perpendicular_lines_a_value_l2809_280989


namespace equation_solution_l2809_280959

def f (x : ℝ) (b : ℝ) : ℝ := 2 * x - b

theorem equation_solution :
  let b : ℝ := 3
  let x : ℝ := 5
  2 * (f x b) - 11 = f (x - 2) b :=
by sorry

end equation_solution_l2809_280959


namespace new_rectangle_area_l2809_280986

/-- Given a rectangle with sides a and b (a < b), prove that the area of a new rectangle
    with base (b + 2a) and height (b - a) is b^2 + ab - 2a^2 -/
theorem new_rectangle_area (a b : ℝ) (h : a < b) :
  (b + 2*a) * (b - a) = b^2 + a*b - 2*a^2 := by
  sorry

end new_rectangle_area_l2809_280986


namespace solution_set_for_a_5_range_of_a_l2809_280947

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 2|

-- Part I
theorem solution_set_for_a_5 :
  {x : ℝ | f 5 x > 9} = {x : ℝ | x < -6 ∨ x > 3} := by sorry

-- Part II
def A (a : ℝ) : Set ℝ := {x : ℝ | f a x ≤ |x - 4|}
def B : Set ℝ := {x : ℝ | |2*x - 1| ≤ 3}

theorem range_of_a :
  ∀ a : ℝ, (A a ∪ B = A a) → a ∈ Set.Icc (-1) 0 := by sorry

end solution_set_for_a_5_range_of_a_l2809_280947


namespace hyperbola_equation_l2809_280940

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1, focal length 10, and point P(2, 1) on its asymptote, prove that the equation of C is x²/20 - y²/5 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∃ c : ℝ, c > 0 ∧ c^2 = a^2 + b^2 ∧ 2*c = 10) →
  (∃ x y : ℝ, x = 2 ∧ y = 1 ∧ y = (b/a) * x) →
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ x^2/20 - y^2/5 = 1) :=
by sorry

end hyperbola_equation_l2809_280940


namespace function_non_negative_implies_a_value_l2809_280987

/-- Given a function f and a real number a, proves that if f satisfies certain conditions, then a = 2/3 -/
theorem function_non_negative_implies_a_value (a : ℝ) :
  (∀ x > 1 - 2*a, (Real.exp (x - a) - 1) * Real.log (x + 2*a - 1) ≥ 0) →
  a = 2/3 := by
  sorry

end function_non_negative_implies_a_value_l2809_280987


namespace weekend_rain_probability_l2809_280972

theorem weekend_rain_probability (p_saturday p_sunday : ℝ) 
  (h1 : p_saturday = 0.6)
  (h2 : p_sunday = 0.7)
  (h3 : 0 ≤ p_saturday ∧ p_saturday ≤ 1)
  (h4 : 0 ≤ p_sunday ∧ p_sunday ≤ 1) :
  1 - (1 - p_saturday) * (1 - p_sunday) = 0.88 := by
sorry

end weekend_rain_probability_l2809_280972


namespace average_height_four_people_l2809_280963

/-- The average height of four individuals given their relative heights -/
theorem average_height_four_people (G : ℝ) : 
  G + 2 = 64 →  -- Giselle is 2 inches shorter than Parker
  (G + 2) + 4 = 68 →  -- Parker is 4 inches shorter than Daisy
  68 - 8 = 60 →  -- Daisy is 8 inches taller than Reese
  (G + 64 + 68 + 60) / 4 = (192 + G) / 4 := by sorry

end average_height_four_people_l2809_280963


namespace smallest_five_digit_mod_9_l2809_280957

theorem smallest_five_digit_mod_9 :
  ∀ n : ℕ, 10000 ≤ n ∧ n ≡ 4 [MOD 9] → 10003 ≤ n :=
by sorry

end smallest_five_digit_mod_9_l2809_280957


namespace students_with_glasses_and_watches_l2809_280997

theorem students_with_glasses_and_watches (n : ℕ) 
  (glasses : ℚ) (watches : ℚ) (neither : ℚ) (both : ℕ) :
  glasses = 3/5 →
  watches = 5/6 →
  neither = 1/10 →
  (n : ℚ) * glasses + (n : ℚ) * watches - (n : ℚ) + (n : ℚ) * neither = (n : ℚ) →
  both = 16 :=
by
  sorry

#check students_with_glasses_and_watches

end students_with_glasses_and_watches_l2809_280997


namespace complement_of_M_in_U_l2809_280951

def U : Finset Nat := {1, 3, 5, 7}
def M : Finset Nat := {1, 5}

theorem complement_of_M_in_U :
  (U \ M) = {3, 7} := by sorry

end complement_of_M_in_U_l2809_280951


namespace next_occurrence_is_august_first_l2809_280923

def initial_date : Nat × Nat × Nat := (5, 1, 1994)
def initial_time : Nat × Nat := (7, 32)

def digits : List Nat := [0, 5, 1, 1, 9, 9, 4, 0, 7, 3]

def is_valid_date (d : Nat) (m : Nat) (y : Nat) : Bool :=
  d > 0 && d ≤ 31 && m > 0 && m ≤ 12 && y == 1994

def is_valid_time (h : Nat) (m : Nat) : Bool :=
  h ≥ 0 && h < 24 && m ≥ 0 && m < 60

def date_time_to_digits (d : Nat) (m : Nat) (y : Nat) (h : Nat) (min : Nat) : List Nat :=
  let date_digits := (d / 10) :: (d % 10) :: (m / 10) :: (m % 10) :: (y / 1000) :: ((y / 100) % 10) :: ((y / 10) % 10) :: (y % 10) :: []
  let time_digits := (h / 10) :: (h % 10) :: (min / 10) :: (min % 10) :: []
  date_digits ++ time_digits

def is_next_occurrence (d : Nat) (m : Nat) (h : Nat) (min : Nat) : Prop :=
  is_valid_date d m 1994 ∧
  is_valid_time h min ∧
  date_time_to_digits d m 1994 h min == digits ∧
  (d, m) > (5, 1) ∧
  ∀ (d' : Nat) (m' : Nat) (h' : Nat) (min' : Nat),
    is_valid_date d' m' 1994 →
    is_valid_time h' min' →
    date_time_to_digits d' m' 1994 h' min' == digits →
    (d', m') > (5, 1) →
    (d', m') ≤ (d, m)

theorem next_occurrence_is_august_first :
  is_next_occurrence 1 8 2 45 := by sorry

end next_occurrence_is_august_first_l2809_280923


namespace largest_angle_bound_l2809_280926

/-- Triangle DEF with sides e, f, and d -/
structure Triangle where
  e : ℝ
  f : ℝ
  d : ℝ

/-- The angle opposite to side d in degrees -/
def angle_opposite_d (t : Triangle) : ℝ := sorry

theorem largest_angle_bound (t : Triangle) (y : ℝ) :
  t.e = 2 →
  t.f = 2 →
  t.d > 2 * Real.sqrt 2 →
  (∀ z, z > y → angle_opposite_d t > z) →
  y = 120 := by sorry

end largest_angle_bound_l2809_280926


namespace picnic_gender_difference_l2809_280980

/-- Given a group of people at a picnic, prove the difference between men and women -/
theorem picnic_gender_difference 
  (total : ℕ) 
  (men : ℕ) 
  (women : ℕ) 
  (children : ℕ) 
  (h1 : total = 200)
  (h2 : men + women = children + 20)
  (h3 : men = 65)
  (h4 : total = men + women + children) :
  men - women = 20 := by
sorry

end picnic_gender_difference_l2809_280980


namespace water_polo_team_selection_l2809_280902

def total_team_members : ℕ := 20
def starting_lineup : ℕ := 7
def regular_players : ℕ := 5

def choose_team : ℕ := 
  total_team_members * (total_team_members - 1) * (Nat.choose (total_team_members - 2) regular_players)

theorem water_polo_team_selection :
  choose_team = 3268880 :=
sorry

end water_polo_team_selection_l2809_280902


namespace unique_fraction_property_l2809_280915

theorem unique_fraction_property : ∃! (a b : ℕ), 
  b ≠ 0 ∧ 
  (a : ℚ) / b = (a + 4 : ℚ) / (b + 10) ∧ 
  (a : ℚ) / b = 2 / 5 := by
  sorry

end unique_fraction_property_l2809_280915


namespace max_value_trig_expression_l2809_280918

theorem max_value_trig_expression (x y z : ℝ) :
  (Real.sin (2 * x) + Real.sin y + Real.sin (3 * z)) *
  (Real.cos (2 * x) + Real.cos y + Real.cos (3 * z)) ≤ 9 / 2 := by
sorry

end max_value_trig_expression_l2809_280918


namespace gcd_of_180_210_588_l2809_280996

theorem gcd_of_180_210_588 : Nat.gcd 180 (Nat.gcd 210 588) = 6 := by
  sorry

end gcd_of_180_210_588_l2809_280996


namespace inscribed_triangles_area_relation_l2809_280922

/-- Triangle type -/
structure Triangle where
  area : ℝ

/-- Inscribed triangle relation -/
def inscribed (outer inner : Triangle) : Prop :=
  inner.area < outer.area

/-- Parallel sides relation -/
def parallel_sides (t1 t2 : Triangle) : Prop :=
  true  -- We don't need to define this precisely for the theorem

/-- Theorem statement -/
theorem inscribed_triangles_area_relation (a b c : Triangle)
  (h1 : inscribed a b)
  (h2 : inscribed b c)
  (h3 : parallel_sides a c) :
  b.area = Real.sqrt (a.area * c.area) := by
  sorry

end inscribed_triangles_area_relation_l2809_280922


namespace expenditure_ratio_proof_l2809_280958

/-- Given two persons P1 and P2 with incomes and expenditures, prove their expenditure ratio --/
theorem expenditure_ratio_proof 
  (income_ratio : ℚ) -- Ratio of incomes P1:P2
  (savings : ℕ) -- Amount saved by each person
  (income_p1 : ℕ) -- Income of P1
  (h1 : income_ratio = 5 / 4) -- Income ratio condition
  (h2 : savings = 1600) -- Savings condition
  (h3 : income_p1 = 4000) -- P1's income condition
  : (income_p1 - savings) / ((income_p1 * 4 / 5) - savings) = 3 / 2 := by
  sorry


end expenditure_ratio_proof_l2809_280958


namespace profit_percentage_calculation_l2809_280907

theorem profit_percentage_calculation 
  (cost original_profit selling_price : ℝ)
  (h1 : selling_price = cost + original_profit)
  (h2 : selling_price = 1.12 * cost + 0.53333333333333333 * selling_price) :
  original_profit / cost = 1.4 := by
sorry

end profit_percentage_calculation_l2809_280907


namespace house_selling_price_l2809_280968

theorem house_selling_price (commission_rate : ℝ) (commission : ℝ) (selling_price : ℝ) :
  commission_rate = 0.06 →
  commission = 8880 →
  commission = commission_rate * selling_price →
  selling_price = 148000 := by
  sorry

end house_selling_price_l2809_280968


namespace equation_proof_l2809_280960

theorem equation_proof : 3889 + 12.952 - 47.95 = 3854.002 := by
  sorry

end equation_proof_l2809_280960


namespace basketball_score_l2809_280932

theorem basketball_score (total_shots : ℕ) (three_point_shots : ℕ) : 
  total_shots = 11 → three_point_shots = 4 → 
  3 * three_point_shots + 2 * (total_shots - three_point_shots) = 26 := by
  sorry

end basketball_score_l2809_280932


namespace number_divided_and_subtracted_l2809_280903

theorem number_divided_and_subtracted (x : ℝ) : x = 4.5 → x / 3 = x - 3 := by
  sorry

end number_divided_and_subtracted_l2809_280903


namespace similar_triangles_height_l2809_280982

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 → area_ratio = 9 →
  ∃ h_large : ℝ, h_large = h_small * Real.sqrt area_ratio :=
by
  sorry

end similar_triangles_height_l2809_280982


namespace library_book_difference_prove_book_difference_l2809_280955

theorem library_book_difference (initial_books : ℕ) (bought_two_years_ago : ℕ) 
  (donated_this_year : ℕ) (current_total : ℕ) : ℕ :=
  let books_before_last_year := initial_books + bought_two_years_ago
  let books_bought_last_year := current_total - books_before_last_year + donated_this_year
  books_bought_last_year - bought_two_years_ago

theorem prove_book_difference :
  library_book_difference 500 300 200 1000 = 100 := by
  sorry

end library_book_difference_prove_book_difference_l2809_280955


namespace jacket_cost_ratio_l2809_280928

theorem jacket_cost_ratio (marked_price : ℝ) (h1 : marked_price > 0) : 
  let discount_rate : ℝ := 1/4
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_rate : ℝ := 5/8
  let cost : ℝ := selling_price * cost_rate
  cost / marked_price = 15/32 := by
sorry

end jacket_cost_ratio_l2809_280928


namespace bananas_in_basket_e_l2809_280916

/-- Given 5 baskets of fruits with an average of 25 fruits per basket, 
    where basket A contains 15 apples, B has 30 mangoes, C has 20 peaches, 
    D has 25 pears, and E has an unknown number of bananas, 
    prove that basket E contains 35 bananas. -/
theorem bananas_in_basket_e :
  let num_baskets : ℕ := 5
  let avg_fruits_per_basket : ℕ := 25
  let fruits_a : ℕ := 15
  let fruits_b : ℕ := 30
  let fruits_c : ℕ := 20
  let fruits_d : ℕ := 25
  let total_fruits : ℕ := num_baskets * avg_fruits_per_basket
  let fruits_abcd : ℕ := fruits_a + fruits_b + fruits_c + fruits_d
  let fruits_e : ℕ := total_fruits - fruits_abcd
  fruits_e = 35 := by
  sorry

end bananas_in_basket_e_l2809_280916


namespace absolute_value_equation_solution_difference_l2809_280946

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ - 3| = 25 ∧ |x₂ - 3| = 25 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 50 := by
  sorry

end absolute_value_equation_solution_difference_l2809_280946


namespace diet_soda_sales_l2809_280954

theorem diet_soda_sales (total_sodas : ℕ) (regular_ratio diet_ratio : ℕ) (diet_sodas : ℕ) : 
  total_sodas = 64 →
  regular_ratio = 9 →
  diet_ratio = 7 →
  regular_ratio * diet_sodas = diet_ratio * (total_sodas - diet_sodas) →
  diet_sodas = 28 := by
sorry

end diet_soda_sales_l2809_280954


namespace ylona_initial_count_l2809_280995

/-- The number of rubber bands each person has initially and after Bailey gives some away. -/
structure RubberBands :=
  (bailey_initial : ℕ)
  (justine_initial : ℕ)
  (ylona_initial : ℕ)
  (bailey_final : ℕ)

/-- The conditions of the rubber band problem. -/
def rubber_band_problem (rb : RubberBands) : Prop :=
  rb.justine_initial = rb.bailey_initial + 10 ∧
  rb.ylona_initial = rb.justine_initial + 2 ∧
  rb.bailey_final = rb.bailey_initial - 4 ∧
  rb.bailey_final = 8

/-- Theorem stating that Ylona initially had 24 rubber bands. -/
theorem ylona_initial_count (rb : RubberBands) 
  (h : rubber_band_problem rb) : rb.ylona_initial = 24 := by
  sorry

end ylona_initial_count_l2809_280995


namespace ana_charging_time_proof_l2809_280944

def smartphone_full_charge : ℕ := 26
def tablet_full_charge : ℕ := 53

def ana_charging_time : ℕ :=
  tablet_full_charge + (smartphone_full_charge / 2)

theorem ana_charging_time_proof :
  ana_charging_time = 66 := by
  sorry

end ana_charging_time_proof_l2809_280944


namespace rulers_in_drawer_l2809_280990

/-- The total number of rulers in a drawer after an addition. -/
def total_rulers (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: Given 11 initial rulers and 14 added rulers, the total is 25. -/
theorem rulers_in_drawer : total_rulers 11 14 = 25 := by
  sorry

end rulers_in_drawer_l2809_280990


namespace fair_coin_probability_difference_l2809_280977

def probability_exactly_3_heads (n : ℕ) (p : ℚ) : ℚ :=
  ↑(Nat.choose n 3) * p^3 * (1 - p)^(n - 3)

def probability_all_heads (n : ℕ) (p : ℚ) : ℚ :=
  p^n

theorem fair_coin_probability_difference :
  let n : ℕ := 4
  let p : ℚ := 1/2
  (probability_exactly_3_heads n p) - (probability_all_heads n p) = 7/16 := by
  sorry

end fair_coin_probability_difference_l2809_280977


namespace slide_problem_l2809_280910

/-- The number of additional boys who went down the slide -/
def additional_boys (initial : ℕ) (total : ℕ) : ℕ :=
  total - initial

theorem slide_problem (initial : ℕ) (total : ℕ) 
  (h1 : initial = 22) 
  (h2 : total = 35) : 
  additional_boys initial total = 13 :=
by
  sorry

end slide_problem_l2809_280910


namespace logarithm_equality_implies_golden_ratio_l2809_280929

theorem logarithm_equality_implies_golden_ratio (p q : ℝ) 
  (hp : p > 0) (hq : q > 0) 
  (h : Real.log p / Real.log 9 = Real.log q / Real.log 12 ∧ 
       Real.log p / Real.log 9 = Real.log (p + q) / Real.log 16) : 
  q / p = (1 + Real.sqrt 5) / 2 := by
  sorry

end logarithm_equality_implies_golden_ratio_l2809_280929


namespace ten_faucets_fill_time_l2809_280942

/-- The time (in seconds) it takes for a given number of faucets to fill a tub of a given capacity. -/
def fill_time (num_faucets : ℕ) (capacity : ℝ) : ℝ :=
  sorry

/-- The rate at which one faucet fills a tub (in gallons per minute). -/
def faucet_rate : ℝ :=
  sorry

theorem ten_faucets_fill_time :
  -- Condition 1: Five faucets fill a 150-gallon tub in 10 minutes
  fill_time 5 150 = 10 * 60 →
  -- Condition 2: All faucets dispense water at the same rate (implicit in the definition of faucet_rate)
  -- Prove: Ten faucets will fill a 50-gallon tub in 100 seconds
  fill_time 10 50 = 100 :=
by
  sorry

end ten_faucets_fill_time_l2809_280942


namespace daniels_initial_noodles_l2809_280985

/-- Represents the number of noodles Daniel had initially -/
def initial_noodles : ℕ := sorry

/-- Represents the number of noodles Daniel gave away -/
def noodles_given_away : ℕ := 12

/-- Represents the number of noodles Daniel has now -/
def remaining_noodles : ℕ := 54

/-- Theorem stating that Daniel's initial number of noodles was 66 -/
theorem daniels_initial_noodles : 
  initial_noodles = noodles_given_away + remaining_noodles := by
  sorry

end daniels_initial_noodles_l2809_280985


namespace train_speed_l2809_280927

/-- Proves that the speed of a train is 72 km/hr given its length and time to pass a pole -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 100) (h2 : time = 5) :
  (length / time) * 3.6 = 72 := by
  sorry

end train_speed_l2809_280927


namespace find_A_in_terms_of_B_l2809_280938

/-- Given functions f and g, prove the value of A in terms of B -/
theorem find_A_in_terms_of_B (B : ℝ) (hB : B ≠ 0) :
  let f := fun x => A * x - 3 * B^2 + B * x^2
  let g := fun x => B * x^2
  let A := (3 - 16 * B^2) / 4
  f (g 2) = 0 := by
  sorry

end find_A_in_terms_of_B_l2809_280938


namespace complex_equation_solution_l2809_280900

theorem complex_equation_solution (c d : ℂ) (x : ℝ) :
  Complex.abs c = 3 →
  Complex.abs d = 5 →
  c * d = x - 3 * Complex.I →
  x > 0 →
  x = 6 * Real.sqrt 6 :=
by
  sorry

end complex_equation_solution_l2809_280900


namespace songs_to_learn_l2809_280979

/-- Given that Billy can play 24 songs and his music book contains 52 songs,
    prove that the number of songs he still needs to learn is 28. -/
theorem songs_to_learn (songs_can_play : ℕ) (total_songs : ℕ) 
  (h1 : songs_can_play = 24) (h2 : total_songs = 52) : 
  total_songs - songs_can_play = 28 := by
  sorry

end songs_to_learn_l2809_280979


namespace quarter_circle_roll_path_length_l2809_280950

/-- The path length of the center of a quarter-circle when rolled along a straight line -/
theorem quarter_circle_roll_path_length (r : ℝ) (h : r = 3 / π) :
  let path_length := 3 * (π * r / 2)
  path_length = 9 / 2 := by sorry

end quarter_circle_roll_path_length_l2809_280950


namespace geometric_sequence_inequality_l2809_280901

theorem geometric_sequence_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_geometric : b^2 = a * c) : 
  a^2 + b^2 + c^2 > (a - b + c)^2 := by
sorry

end geometric_sequence_inequality_l2809_280901


namespace trivia_contest_probability_l2809_280953

/-- The number of questions in the trivia contest -/
def num_questions : ℕ := 4

/-- The number of choices for each question -/
def num_choices : ℕ := 4

/-- The minimum number of correct answers needed to win -/
def min_correct : ℕ := 3

/-- The probability of guessing one question correctly -/
def prob_correct : ℚ := 1 / num_choices

/-- The probability of guessing one question incorrectly -/
def prob_incorrect : ℚ := 1 - prob_correct

/-- The probability of winning the trivia contest -/
def prob_winning : ℚ := 13 / 256

theorem trivia_contest_probability :
  (prob_correct ^ num_questions) +
  (num_questions.choose min_correct) * (prob_correct ^ min_correct) * (prob_incorrect ^ (num_questions - min_correct)) =
  prob_winning := by sorry

end trivia_contest_probability_l2809_280953


namespace quadratic_roots_relation_l2809_280976

theorem quadratic_roots_relation (a b c : ℚ) : 
  (∃ r s : ℚ, (5 * r^2 + 2 * r - 4 = 0) ∧ (5 * s^2 + 2 * s - 4 = 0) ∧ 
   (a * (r - 3)^2 + b * (r - 3) + c = 0) ∧ (a * (s - 3)^2 + b * (s - 3) + c = 0)) →
  c = 47/5 := by
sorry

end quadratic_roots_relation_l2809_280976


namespace april_savings_l2809_280949

def savings_pattern (month : Nat) : Nat :=
  2^month

theorem april_savings : savings_pattern 3 = 16 := by
  sorry

end april_savings_l2809_280949


namespace t_cube_surface_area_l2809_280971

/-- Represents a T-shaped structure made of unit cubes -/
structure TCube where
  vertical_cubes : ℕ
  horizontal_cubes : ℕ
  intersection_position : ℕ

/-- Calculates the surface area of a T-shaped structure -/
def surface_area (t : TCube) : ℕ :=
  sorry

/-- The specific T-shaped structure described in the problem -/
def problem_t_cube : TCube :=
  { vertical_cubes := 5
  , horizontal_cubes := 5
  , intersection_position := 3 }

theorem t_cube_surface_area :
  surface_area problem_t_cube = 33 :=
sorry

end t_cube_surface_area_l2809_280971


namespace find_x_l2809_280906

def A : Set ℝ := {0, 1, 2}
def B (x : ℝ) : Set ℝ := {1, 1/x}

theorem find_x : ∃ x : ℝ, B x ⊆ A ∧ x = 1/2 := by
  sorry

end find_x_l2809_280906


namespace harry_apples_l2809_280943

/-- The number of apples each person has -/
structure Apples where
  martha : ℕ
  tim : ℕ
  harry : ℕ
  jane : ℕ

/-- The conditions of the problem -/
def apple_conditions (a : Apples) : Prop :=
  a.martha = 68 ∧
  a.tim = a.martha - 30 ∧
  a.harry = a.tim / 2 ∧
  a.jane = (a.tim + a.martha) / 4

/-- The theorem stating that Harry has 19 apples -/
theorem harry_apples (a : Apples) (h : apple_conditions a) : a.harry = 19 := by
  sorry

end harry_apples_l2809_280943


namespace one_plus_sqrt3i_in_M_l2809_280999

/-- The set M of complex numbers with magnitude 2 -/
def M : Set ℂ := {z : ℂ | Complex.abs z = 2}

/-- Proof that 1 + √3i belongs to M -/
theorem one_plus_sqrt3i_in_M : (1 : ℂ) + Complex.I * Real.sqrt 3 ∈ M := by
  sorry

end one_plus_sqrt3i_in_M_l2809_280999


namespace loss_percentage_tables_l2809_280921

theorem loss_percentage_tables (C S : ℝ) (h : 15 * C = 20 * S) : 
  (C - S) / C * 100 = 25 := by
sorry

end loss_percentage_tables_l2809_280921


namespace complement_A_in_U_equals_union_l2809_280930

-- Define the universal set U
def U : Set ℝ := {x | -3 < x ∧ x < 3}

-- Define set A
def A : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define the complement of A in U
def complement_A_in_U : Set ℝ := {x | x ∈ U ∧ x ∉ A}

-- Theorem statement
theorem complement_A_in_U_equals_union : 
  complement_A_in_U = {x | (-3 < x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x < 3)} :=
by sorry

end complement_A_in_U_equals_union_l2809_280930


namespace power_of_power_l2809_280904

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l2809_280904


namespace no_natural_numbers_satisfying_condition_l2809_280934

theorem no_natural_numbers_satisfying_condition : 
  ¬ ∃ (a b : ℕ), a < b ∧ ∃ (k : ℕ), b^2 + 4*a = k^2 := by
  sorry

end no_natural_numbers_satisfying_condition_l2809_280934


namespace geometric_sequence_property_l2809_280973

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

/-- The product of the first n terms of a sequence -/
def SequenceProduct (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (λ acc i => acc * b (i + 1)) 1

theorem geometric_sequence_property (b : ℕ → ℝ) (h : GeometricSequence b) (h7 : b 7 = 1) :
  ∀ n : ℕ+, n < 13 → SequenceProduct b n = SequenceProduct b (13 - n) :=
sorry

end geometric_sequence_property_l2809_280973


namespace hotel_towels_l2809_280913

/-- A hotel with a fixed number of rooms, people per room, and towels per person. -/
structure Hotel where
  rooms : ℕ
  peoplePerRoom : ℕ
  towelsPerPerson : ℕ

/-- Calculate the total number of towels handed out in a full hotel. -/
def totalTowels (h : Hotel) : ℕ :=
  h.rooms * h.peoplePerRoom * h.towelsPerPerson

/-- Theorem stating that a specific hotel configuration hands out 60 towels. -/
theorem hotel_towels :
  ∃ (h : Hotel), h.rooms = 10 ∧ h.peoplePerRoom = 3 ∧ h.towelsPerPerson = 2 ∧ totalTowels h = 60 :=
by
  sorry


end hotel_towels_l2809_280913


namespace divisors_of_square_of_four_divisor_number_l2809_280992

/-- A natural number has exactly 4 divisors -/
def has_four_divisors (m : ℕ) : Prop :=
  (Finset.filter (· ∣ m) (Finset.range (m + 1))).card = 4

/-- The number of divisors of the square of a number with 4 divisors -/
theorem divisors_of_square_of_four_divisor_number (m : ℕ) :
  has_four_divisors m →
  (Finset.filter (· ∣ m^2) (Finset.range (m^2 + 1))).card = 7 ∨
  (Finset.filter (· ∣ m^2) (Finset.range (m^2 + 1))).card = 9 :=
by
  sorry

end divisors_of_square_of_four_divisor_number_l2809_280992


namespace equal_cost_guests_correct_l2809_280939

/-- The number of guests for which the costs of renting Caesar's and Venus Hall are equal -/
def equal_cost_guests : ℕ :=
  let caesar_rental : ℕ := 800
  let caesar_per_meal : ℕ := 30
  let venus_rental : ℕ := 500
  let venus_per_meal : ℕ := 35
  60

theorem equal_cost_guests_correct :
  let caesar_rental : ℕ := 800
  let caesar_per_meal : ℕ := 30
  let venus_rental : ℕ := 500
  let venus_per_meal : ℕ := 35
  caesar_rental + caesar_per_meal * equal_cost_guests = venus_rental + venus_per_meal * equal_cost_guests :=
by sorry

end equal_cost_guests_correct_l2809_280939


namespace erasers_left_in_box_l2809_280912

/-- The number of erasers left in the box after Doris, Mark, and Ellie take some out. -/
def erasers_left (initial : ℕ) (doris_takes : ℕ) (mark_takes : ℕ) (ellie_takes : ℕ) : ℕ :=
  initial - doris_takes - mark_takes - ellie_takes

/-- Theorem stating that 105 erasers are left in the box -/
theorem erasers_left_in_box :
  erasers_left 250 75 40 30 = 105 := by
  sorry

end erasers_left_in_box_l2809_280912


namespace students_neither_sport_l2809_280952

theorem students_neither_sport (total : ℕ) (football : ℕ) (cricket : ℕ) (both : ℕ)
  (h_total : total = 460)
  (h_football : football = 325)
  (h_cricket : cricket = 175)
  (h_both : both = 90) :
  total - (football + cricket - both) = 50 := by
  sorry

end students_neither_sport_l2809_280952


namespace neil_final_three_prob_l2809_280948

/-- A 3-sided die with numbers 1, 2, and 3 -/
inductive Die : Type
| one : Die
| two : Die
| three : Die

/-- The probability of rolling each number on the die -/
def prob_roll (d : Die) : ℚ := 1/3

/-- The event of Neil's final number being 3 -/
def neil_final_three : Set (Die × Die) := {(j, n) | n = Die.three}

/-- The probability space of all possible outcomes (Jerry's roll, Neil's final roll) -/
def prob_space : Set (Die × Die) := Set.univ

/-- The theorem stating the probability of Neil's final number being 3 -/
theorem neil_final_three_prob :
  ∃ (P : Set (Die × Die) → ℚ),
    P prob_space = 1 ∧
    P neil_final_three = 11/18 :=
sorry

end neil_final_three_prob_l2809_280948


namespace square_minus_one_factorization_l2809_280991

theorem square_minus_one_factorization (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end square_minus_one_factorization_l2809_280991


namespace modified_system_solution_l2809_280983

theorem modified_system_solution 
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) 
  (h : a₁ * 8 + b₁ * 3 = c₁ ∧ a₂ * 8 + b₂ * 3 = c₂) :
  4 * a₁ * 10 + 3 * b₁ * 5 = 5 * c₁ ∧ 
  4 * a₂ * 10 + 3 * b₂ * 5 = 5 * c₂ :=
by sorry

end modified_system_solution_l2809_280983


namespace first_bouquet_carnations_l2809_280935

/-- The number of carnations in the first bouquet -/
def carnations_in_first_bouquet (total_bouquets : ℕ) 
  (carnations_in_second : ℕ) (carnations_in_third : ℕ) (average : ℕ) : ℕ :=
  total_bouquets * average - carnations_in_second - carnations_in_third

/-- Theorem stating the number of carnations in the first bouquet -/
theorem first_bouquet_carnations :
  carnations_in_first_bouquet 3 14 13 12 = 9 := by
  sorry

#eval carnations_in_first_bouquet 3 14 13 12

end first_bouquet_carnations_l2809_280935


namespace monica_has_27_peaches_l2809_280911

/-- The number of peaches each person has -/
structure Peaches where
  steven : ℕ
  jake : ℕ
  jill : ℕ
  monica : ℕ

/-- The conditions given in the problem -/
def peach_conditions (p : Peaches) : Prop :=
  p.steven = 16 ∧
  p.jake = p.steven - 7 ∧
  p.jake = p.jill + 9 ∧
  p.monica = 3 * p.jake

/-- Theorem: Given the conditions, Monica has 27 peaches -/
theorem monica_has_27_peaches (p : Peaches) (h : peach_conditions p) : p.monica = 27 := by
  sorry

end monica_has_27_peaches_l2809_280911


namespace abc_order_l2809_280905

theorem abc_order : 
  let a : ℝ := 1/2
  let b : ℝ := Real.log (3/2)
  let c : ℝ := (π/2) * Real.sin (1/2)
  b < a ∧ a < c :=
by sorry

end abc_order_l2809_280905


namespace solution_set_when_m_equals_3_range_of_m_for_inequality_l2809_280908

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 6| - |m - x|

-- Theorem for part I
theorem solution_set_when_m_equals_3 :
  {x : ℝ | f x 3 ≥ 5} = {x : ℝ | x ≥ 1} := by sorry

-- Theorem for part II
theorem range_of_m_for_inequality :
  {m : ℝ | ∀ x, f x m ≤ 7} = {m : ℝ | -13 ≤ m ∧ m ≤ 1} := by sorry

end solution_set_when_m_equals_3_range_of_m_for_inequality_l2809_280908


namespace inequality_proof_l2809_280993

theorem inequality_proof (x y z : ℝ) 
  (h1 : x^2 + y*z ≠ 0) (h2 : y^2 + z*x ≠ 0) (h3 : z^2 + x*y ≠ 0) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) ≥ 6 := by
  sorry

end inequality_proof_l2809_280993


namespace peg_arrangement_count_l2809_280969

/-- The number of ways to distribute colored pegs on a square board. -/
def peg_arrangements : ℕ :=
  let red_pegs := 6
  let green_pegs := 5
  let blue_pegs := 4
  let yellow_pegs := 3
  let orange_pegs := 2
  let board_size := 6
  Nat.factorial red_pegs * Nat.factorial green_pegs * Nat.factorial blue_pegs *
  Nat.factorial yellow_pegs * Nat.factorial orange_pegs

/-- Theorem stating the number of valid peg arrangements. -/
theorem peg_arrangement_count :
  peg_arrangements = 12441600 := by
  sorry

end peg_arrangement_count_l2809_280969


namespace infinite_solutions_iff_a_eq_neg_twelve_l2809_280909

theorem infinite_solutions_iff_a_eq_neg_twelve (a : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - a) = 3 * (4 * x + 16)) ↔ a = -12 := by
  sorry

end infinite_solutions_iff_a_eq_neg_twelve_l2809_280909


namespace distance_to_y_axis_l2809_280956

/-- The distance from a point P(-2, 3) to the y-axis is 2. -/
theorem distance_to_y_axis :
  let P : ℝ × ℝ := (-2, 3)
  abs P.1 = 2 :=
by sorry

end distance_to_y_axis_l2809_280956


namespace overall_gain_percent_l2809_280920

/-- Calculates the overall gain percent after applying two discounts -/
theorem overall_gain_percent (M : ℝ) (M_pos : M > 0) : 
  let cost_price := 0.64 * M
  let price_after_first_discount := 0.86 * M
  let final_price := 0.9 * price_after_first_discount
  let gain := final_price - cost_price
  let gain_percent := (gain / cost_price) * 100
  ∃ ε > 0, |gain_percent - 20.94| < ε :=
by sorry

end overall_gain_percent_l2809_280920


namespace largest_fraction_l2809_280961

theorem largest_fraction :
  let a := 5 / 13
  let b := 7 / 16
  let c := 23 / 46
  let d := 51 / 101
  let e := 203 / 405
  d > a ∧ d > b ∧ d > c ∧ d > e := by
sorry

end largest_fraction_l2809_280961


namespace absolute_value_simplification_l2809_280998

theorem absolute_value_simplification : |(-4^2 + 7)| = 9 := by sorry

end absolute_value_simplification_l2809_280998


namespace karl_drove_420_miles_l2809_280917

/-- Represents Karl's car and trip details --/
structure KarlsTrip where
  miles_per_gallon : ℝ
  tank_capacity : ℝ
  initial_distance : ℝ
  gas_bought : ℝ
  final_tank_fraction : ℝ

/-- Calculates the total distance driven given the trip details --/
def total_distance (trip : KarlsTrip) : ℝ :=
  trip.initial_distance

/-- Theorem stating that Karl drove 420 miles --/
theorem karl_drove_420_miles :
  let trip := KarlsTrip.mk 30 16 420 10 (3/4)
  total_distance trip = 420 := by sorry

end karl_drove_420_miles_l2809_280917


namespace bill_vote_difference_l2809_280994

theorem bill_vote_difference (total : ℕ) (initial_for initial_against revote_for revote_against : ℕ) :
  total = 400 →
  initial_for + initial_against = total →
  revote_for + revote_against = total →
  (revote_for : ℚ) - revote_against = 3 * (initial_against - initial_for) →
  (revote_for : ℚ) = 13 / 12 * initial_against →
  revote_for - initial_for = 36 :=
by sorry

end bill_vote_difference_l2809_280994


namespace fourth_grade_final_count_l2809_280981

/-- Calculates the final number of students in a class given the initial count and changes throughout the year. -/
def final_student_count (initial : ℕ) (left_first : ℕ) (joined_first : ℕ) (left_second : ℕ) (joined_second : ℕ) : ℕ :=
  initial - left_first + joined_first - left_second + joined_second

/-- Theorem stating that the final number of students in the fourth grade class is 37. -/
theorem fourth_grade_final_count : 
  final_student_count 35 6 4 3 7 = 37 := by
  sorry

end fourth_grade_final_count_l2809_280981


namespace hoseok_took_fewest_l2809_280919

/-- Represents the number of cards taken by each person -/
structure CardCount where
  jungkook : ℕ
  hoseok : ℕ
  seokjin : ℕ

/-- Defines the conditions of the problem -/
def problemConditions (cc : CardCount) : Prop :=
  cc.jungkook = 10 ∧
  cc.hoseok = 7 ∧
  cc.seokjin = cc.jungkook - 2

/-- Theorem stating that Hoseok took the fewest cards -/
theorem hoseok_took_fewest (cc : CardCount) 
  (h : problemConditions cc) : 
  cc.hoseok < cc.jungkook ∧ cc.hoseok < cc.seokjin :=
by
  sorry

#check hoseok_took_fewest

end hoseok_took_fewest_l2809_280919


namespace initial_cats_proof_l2809_280941

def total_cats : ℕ := 7
def female_kittens : ℕ := 3
def male_kittens : ℕ := 2

def initial_cats : ℕ := total_cats - (female_kittens + male_kittens)

theorem initial_cats_proof : initial_cats = 2 := by
  sorry

end initial_cats_proof_l2809_280941


namespace eighty_percent_of_forty_l2809_280914

theorem eighty_percent_of_forty (x : ℚ) : x * 20 + 16 = 32 → x = 4/5 := by sorry

end eighty_percent_of_forty_l2809_280914


namespace yellow_crayon_count_l2809_280966

theorem yellow_crayon_count (red : ℕ) (blue : ℕ) (yellow : ℕ) : 
  red = 14 → 
  blue = red + 5 → 
  yellow = 2 * blue - 6 → 
  yellow = 32 := by
sorry

end yellow_crayon_count_l2809_280966


namespace arithmetic_sequence_51st_term_l2809_280984

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_51st_term 
  (a : ℕ → ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + 2) : 
  a 51 = 101 := by
sorry

end arithmetic_sequence_51st_term_l2809_280984


namespace sin_2alpha_plus_2pi_3_l2809_280924

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ) + 1

theorem sin_2alpha_plus_2pi_3 (ω φ α : ℝ) :
  ω > 0 →
  0 ≤ φ ∧ φ ≤ Real.pi / 2 →
  (∀ x : ℝ, f ω φ (x + Real.pi / ω) = f ω φ x) →
  f ω φ (Real.pi / 6) = 2 →
  f ω φ α = 9 / 5 →
  Real.pi / 6 < α ∧ α < 2 * Real.pi / 3 →
  Real.sin (2 * α + 2 * Real.pi / 3) = -24 / 25 := by
sorry

end sin_2alpha_plus_2pi_3_l2809_280924


namespace largest_prime_factor_of_1729_l2809_280965

theorem largest_prime_factor_of_1729 : ∃ (p : Nat), p.Prime ∧ p ∣ 1729 ∧ ∀ (q : Nat), q.Prime → q ∣ 1729 → q ≤ p := by
  sorry

end largest_prime_factor_of_1729_l2809_280965


namespace star_not_associative_l2809_280936

-- Define the set T of non-zero real numbers
def T : Set ℝ := {x : ℝ | x ≠ 0}

-- Define the binary operation *
def star (a b : ℝ) : ℝ := 3 * a * b

-- Theorem: * is not associative over T
theorem star_not_associative :
  ∃ (a b c : T), star (star a b) c ≠ star a (star b c) := by
  sorry

end star_not_associative_l2809_280936


namespace group_a_better_performance_l2809_280964

/-- Represents a group of students with their quiz scores -/
structure StudentGroup where
  scores : List Nat
  mean : Nat
  median : Nat
  mode : Nat
  variance : Nat
  excellent_rate : Rat

/-- Defines what score is considered excellent -/
def excellent_score : Nat := 8

/-- Group A data -/
def group_a : StudentGroup := {
  scores := [5, 7, 8, 8, 8, 8, 8, 9, 9, 10],
  mean := 8,
  median := 8,
  mode := 8,
  variance := 16,
  excellent_rate := 8 / 10
}

/-- Group B data -/
def group_b : StudentGroup := {
  scores := [7, 7, 7, 7, 8, 8, 8, 9, 9, 10],
  mean := 8,
  median := 8,
  mode := 7,
  variance := 1,
  excellent_rate := 6 / 10
}

/-- Theorem stating that Group A has a higher excellent rate than Group B -/
theorem group_a_better_performance (ga : StudentGroup) (gb : StudentGroup) 
  (h1 : ga = group_a) (h2 : gb = group_b) : 
  ga.excellent_rate > gb.excellent_rate := by
  sorry

end group_a_better_performance_l2809_280964


namespace line_segment_proportions_l2809_280933

/-- Given line segments a and b, prove the fourth proportional and mean proportional -/
theorem line_segment_proportions (a b : ℝ) (ha : a = 5) (hb : b = 3) :
  let fourth_prop := b * (a - b) / a
  let mean_prop := Real.sqrt ((a + b) * (a - b))
  fourth_prop = 1.2 ∧ mean_prop = 4 := by
  sorry

end line_segment_proportions_l2809_280933


namespace rectangular_field_dimensions_l2809_280974

theorem rectangular_field_dimensions (m : ℝ) : 
  m > 3 ∧ (2 * m + 9) * (m - 3) = 55 →
  m = (-3 + Real.sqrt 665) / 4 :=
by sorry

end rectangular_field_dimensions_l2809_280974


namespace first_file_size_is_80_l2809_280931

/-- Calculates the size of the first file given the internet speed, total download time, and sizes of two other files. -/
def first_file_size (speed : ℝ) (time : ℝ) (file2 : ℝ) (file3 : ℝ) : ℝ :=
  speed * time * 60 - file2 - file3

/-- Proves that given the specified conditions, the size of the first file is 80 megabits. -/
theorem first_file_size_is_80 :
  first_file_size 2 2 90 70 = 80 := by
  sorry

end first_file_size_is_80_l2809_280931


namespace parallel_line_through_point_l2809_280970

/-- A line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal --/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- A point lies on a line if it satisfies the line's equation --/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem parallel_line_through_point :
  let l1 : Line := { a := 3, b := 4, c := 1 }
  let l2 : Line := { a := 3, b := 4, c := -11 }
  parallel l1 l2 ∧ point_on_line 1 2 l2 := by
  sorry

end parallel_line_through_point_l2809_280970


namespace least_positive_integer_to_multiple_of_three_l2809_280988

theorem least_positive_integer_to_multiple_of_three :
  ∃ (n : ℕ), n > 0 ∧ (527 + n) % 3 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (527 + m) % 3 = 0 → n ≤ m :=
by sorry

end least_positive_integer_to_multiple_of_three_l2809_280988


namespace wage_increase_constant_wage_increase_l2809_280978

/-- Represents the regression equation for worker's wage based on labor productivity -/
def wage_equation (x : ℝ) : ℝ := 10 + 70 * x

/-- Proves that an increase of 1 in labor productivity results in an increase of 70 in wage -/
theorem wage_increase (x : ℝ) : wage_equation (x + 1) - wage_equation x = 70 := by
  sorry

/-- Proves that the wage increase is constant for any labor productivity value -/
theorem constant_wage_increase (x y : ℝ) : 
  wage_equation (x + 1) - wage_equation x = wage_equation (y + 1) - wage_equation y := by
  sorry

end wage_increase_constant_wage_increase_l2809_280978
