import Mathlib

namespace sum_of_squared_pairs_l612_61214

theorem sum_of_squared_pairs (a b c d : ℝ) : 
  (a^4 - 24*a^3 + 50*a^2 - 35*a + 10 = 0) →
  (b^4 - 24*b^3 + 50*b^2 - 35*b + 10 = 0) →
  (c^4 - 24*c^3 + 50*c^2 - 35*c + 10 = 0) →
  (d^4 - 24*d^3 + 50*d^2 - 35*d + 10 = 0) →
  (a+b)^2 + (b+c)^2 + (c+d)^2 + (d+a)^2 = 541 := by sorry

end sum_of_squared_pairs_l612_61214


namespace quadratic_no_real_roots_l612_61269

theorem quadratic_no_real_roots (k : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x - k ≠ 0) → k < -1 :=
by sorry

end quadratic_no_real_roots_l612_61269


namespace tank_plastering_cost_l612_61291

/-- Calculate the cost of plastering a tank's walls and bottom -/
theorem tank_plastering_cost 
  (length : ℝ) 
  (width : ℝ) 
  (depth : ℝ) 
  (cost_per_sq_m : ℝ) : 
  length = 25 → 
  width = 12 → 
  depth = 6 → 
  cost_per_sq_m = 0.75 → 
  2 * (length * depth + width * depth) + length * width = 744 ∧ 
  (2 * (length * depth + width * depth) + length * width) * cost_per_sq_m = 558 := by
  sorry

end tank_plastering_cost_l612_61291


namespace nineteen_eleven_div_eight_l612_61225

theorem nineteen_eleven_div_eight (x : ℕ) : 19^11 / 19^8 = 6859 := by
  sorry

end nineteen_eleven_div_eight_l612_61225


namespace geometric_sequence_product_l612_61266

def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r ∧ e = d * r

theorem geometric_sequence_product (a b c : ℝ) :
  is_geometric_sequence (-1) a b c (-2) →
  a * b * c = -2 * Real.sqrt 2 := by
sorry

end geometric_sequence_product_l612_61266


namespace circumcenter_on_side_implies_right_angled_l612_61279

/-- A triangle is represented by its three vertices in a 2D plane. -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The circumcenter of a triangle is the point where the perpendicular bisectors of the sides intersect. -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- A predicate to check if a point lies on a side of a triangle. -/
def point_on_side (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- A predicate to check if a triangle is right-angled. -/
def is_right_angled (t : Triangle) : Prop := sorry

/-- Theorem: If the circumcenter of a triangle lies on one of its sides, then the triangle is right-angled. -/
theorem circumcenter_on_side_implies_right_angled (t : Triangle) :
  point_on_side (circumcenter t) t → is_right_angled t := by
  sorry

end circumcenter_on_side_implies_right_angled_l612_61279


namespace condition1_arrangements_condition2_arrangements_condition3_arrangements_l612_61244

def num_boys : ℕ := 5
def num_girls : ℕ := 3
def num_subjects : ℕ := 5

def arrangements_condition1 : ℕ := 5520
def arrangements_condition2 : ℕ := 3360
def arrangements_condition3 : ℕ := 360

/-- The number of ways to select representatives under condition 1 -/
theorem condition1_arrangements :
  (Nat.choose num_boys num_boys +
   Nat.choose num_boys (num_boys - 1) * Nat.choose num_girls 1 +
   Nat.choose num_boys (num_boys - 2) * Nat.choose num_girls 2) *
  Nat.factorial num_subjects = arrangements_condition1 := by sorry

/-- The number of ways to select representatives under condition 2 -/
theorem condition2_arrangements :
  Nat.choose (num_boys + num_girls - 1) (num_subjects - 1) *
  (num_subjects - 1) * Nat.factorial (num_subjects - 1) = arrangements_condition2 := by sorry

/-- The number of ways to select representatives under condition 3 -/
theorem condition3_arrangements :
  Nat.choose (num_boys + num_girls - 2) (num_subjects - 2) *
  (num_subjects - 2) * Nat.factorial (num_subjects - 2) = arrangements_condition3 := by sorry

end condition1_arrangements_condition2_arrangements_condition3_arrangements_l612_61244


namespace infinite_sum_evaluation_l612_61247

theorem infinite_sum_evaluation : 
  (∑' n : ℕ, (n : ℝ) / ((n : ℝ)^4 + 4)) = 3/8 := by sorry

end infinite_sum_evaluation_l612_61247


namespace ball_ratio_problem_l612_61220

theorem ball_ratio_problem (white_balls red_balls : ℕ) : 
  (white_balls : ℚ) / red_balls = 4 / 3 →
  white_balls = 12 →
  red_balls = 9 := by
sorry

end ball_ratio_problem_l612_61220


namespace squares_in_35x2_grid_l612_61278

/-- The number of squares in a rectangular grid --/
def count_squares (length width : ℕ) : ℕ :=
  -- Count 1x1 squares
  length * width +
  -- Count 2x2 squares
  (length - 1) * (width - 1)

/-- Theorem: The number of squares in a 35x2 grid is 104 --/
theorem squares_in_35x2_grid :
  count_squares 35 2 = 104 := by
  sorry

end squares_in_35x2_grid_l612_61278


namespace ricks_ironing_total_l612_61258

/-- Rick's ironing problem -/
theorem ricks_ironing_total (shirts_per_hour pants_per_hour shirt_hours pant_hours : ℕ) 
  (h1 : shirts_per_hour = 4)
  (h2 : pants_per_hour = 3)
  (h3 : shirt_hours = 3)
  (h4 : pant_hours = 5) :
  shirts_per_hour * shirt_hours + pants_per_hour * pant_hours = 27 := by
  sorry

#check ricks_ironing_total

end ricks_ironing_total_l612_61258


namespace vanessa_recycled_20_pounds_l612_61209

/-- The number of pounds that earn one point -/
def pounds_per_point : ℕ := 9

/-- The number of pounds Vanessa's friends recycled -/
def friends_pounds : ℕ := 16

/-- The total number of points earned -/
def total_points : ℕ := 4

/-- Vanessa's recycled pounds -/
def vanessa_pounds : ℕ := total_points * pounds_per_point - friends_pounds

theorem vanessa_recycled_20_pounds : vanessa_pounds = 20 := by
  sorry

end vanessa_recycled_20_pounds_l612_61209


namespace initial_segment_theorem_l612_61282

theorem initial_segment_theorem (m : ℕ) : ∃ (n k : ℕ), (10^k * m : ℕ) ≤ 2^n ∧ 2^n < 10^k * (m + 1) := by
  sorry

end initial_segment_theorem_l612_61282


namespace pie_eating_contest_l612_61224

theorem pie_eating_contest (student1 student2 student3 : ℚ) 
  (h1 : student1 = 5/6)
  (h2 : student2 = 7/8)
  (h3 : student3 = 2/3) :
  max student1 (max student2 student3) - min student1 (min student2 student3) = 5/24 := by
sorry

end pie_eating_contest_l612_61224


namespace min_value_sqrt_sum_l612_61212

theorem min_value_sqrt_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  Real.sqrt (x + 1/x) + Real.sqrt (y + 1/y) ≥ Real.sqrt 10 := by
  sorry

end min_value_sqrt_sum_l612_61212


namespace halfway_between_one_eighth_and_one_third_l612_61210

theorem halfway_between_one_eighth_and_one_third :
  (1/8 : ℚ) + ((1/3 : ℚ) - (1/8 : ℚ)) / 2 = 11/48 := by sorry

end halfway_between_one_eighth_and_one_third_l612_61210


namespace jellybean_count_l612_61228

/-- The number of jellybeans needed to fill a large drinking glass -/
def large_glass : ℕ := sorry

/-- The number of jellybeans needed to fill a small drinking glass -/
def small_glass : ℕ := sorry

/-- The total number of large glasses -/
def num_large_glasses : ℕ := 5

/-- The total number of small glasses -/
def num_small_glasses : ℕ := 3

/-- The total number of jellybeans needed to fill all glasses -/
def total_jellybeans : ℕ := 325

theorem jellybean_count :
  (small_glass = large_glass / 2) →
  (num_large_glasses * large_glass + num_small_glasses * small_glass = total_jellybeans) →
  large_glass = 50 := by
  sorry

end jellybean_count_l612_61228


namespace max_dip_amount_l612_61284

/-- Given the following conditions:
  * total_money: The total amount of money available to spend on artichokes
  * cost_per_artichoke: The cost of each artichoke
  * artichokes_per_batch: The number of artichokes needed to make one batch of dip
  * ounces_per_batch: The number of ounces of dip produced from one batch

  Prove that the maximum amount of dip that can be made is 20 ounces.
-/
theorem max_dip_amount (total_money : ℚ) (cost_per_artichoke : ℚ) 
  (artichokes_per_batch : ℕ) (ounces_per_batch : ℚ) 
  (h1 : total_money = 15)
  (h2 : cost_per_artichoke = 5/4)
  (h3 : artichokes_per_batch = 3)
  (h4 : ounces_per_batch = 5) :
  (total_money / cost_per_artichoke) * (ounces_per_batch / artichokes_per_batch) = 20 :=
by sorry

end max_dip_amount_l612_61284


namespace reporter_earnings_l612_61264

/-- A reporter's earnings calculation --/
theorem reporter_earnings 
  (earnings_per_article : ℝ) 
  (articles : ℕ) 
  (total_hours : ℝ) 
  (words_per_minute : ℝ) 
  (earnings_per_hour : ℝ) 
  (h1 : earnings_per_article = 60) 
  (h2 : articles = 3) 
  (h3 : total_hours = 4) 
  (h4 : words_per_minute = 10) 
  (h5 : earnings_per_hour = 105) : 
  let total_earnings := earnings_per_hour * total_hours
  let total_words := words_per_minute * (total_hours * 60)
  let article_earnings := earnings_per_article * articles
  let word_earnings := total_earnings - article_earnings
  word_earnings / total_words = 0.1 := by
sorry

end reporter_earnings_l612_61264


namespace sum_of_i_powers_l612_61287

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^23 + i^28 + i^33 + i^38 + i^43 = -i := by
  sorry

end sum_of_i_powers_l612_61287


namespace system_solution_approximation_l612_61230

/-- The system of equations has a unique solution close to (0.4571, 0.1048) -/
theorem system_solution_approximation : ∃! (x y : ℝ), 
  (4 * x - 6 * y = -2) ∧ 
  (5 * x + 3 * y = 2.6) ∧ 
  (abs (x - 0.4571) < 0.0001) ∧ 
  (abs (y - 0.1048) < 0.0001) := by
  sorry

end system_solution_approximation_l612_61230


namespace area_of_sine_curve_l612_61237

theorem area_of_sine_curve (f : ℝ → ℝ) (a b : ℝ) : 
  (f = λ x => Real.sin x) →
  (a = -π/2) →
  (b = 5*π/4) →
  (∫ x in a..b, |f x| ) = 4 - Real.sqrt 2 / 2 := by
sorry

end area_of_sine_curve_l612_61237


namespace cylinder_minus_cones_volume_l612_61202

/-- The volume of a cylinder minus two congruent cones -/
theorem cylinder_minus_cones_volume (r h_cylinder h_cone : ℝ) 
  (hr : r = 10)
  (hh_cylinder : h_cylinder = 20)
  (hh_cone : h_cone = 9) :
  π * r^2 * h_cylinder - 2 * (1/3 * π * r^2 * h_cone) = 1400 * π := by
sorry

end cylinder_minus_cones_volume_l612_61202


namespace picnic_attendance_l612_61226

/-- Represents the number of people at a picnic -/
structure PicnicAttendance where
  men : ℕ
  women : ℕ
  adults : ℕ
  children : ℕ

/-- Theorem: Given the conditions of the picnic, the total number of attendees is 240 -/
theorem picnic_attendance (p : PicnicAttendance) 
  (h1 : p.men = p.women + 40)
  (h2 : p.adults = p.children + 40)
  (h3 : p.men = 90)
  : p.men + p.women + p.children = 240 := by
  sorry

#check picnic_attendance

end picnic_attendance_l612_61226


namespace circle_symmetry_l612_61236

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 5 = 0

-- Define circle D
def circle_D (x y : ℝ) : Prop := (x + 2)^2 + (y - 6)^2 = 1

-- Define circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 1

-- Define symmetry with respect to a line
def symmetric_wrt_line (c₁ c₂ : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ (p : ℝ × ℝ), l p.1 p.2 ∧ 
    (c₁.1 + c₂.1) / 2 = p.1 ∧ 
    (c₁.2 + c₂.2) / 2 = p.2 ∧
    (c₂.1 - c₁.1) * (p.2 - c₁.2) = (c₂.2 - c₁.2) * (p.1 - c₁.1)

theorem circle_symmetry :
  symmetric_wrt_line (-2, 6) (1, 3) line_l →
  (∀ x y : ℝ, circle_D x y ↔ circle_C (x + 3) (y - 3)) :=
sorry

end circle_symmetry_l612_61236


namespace trailing_zeros_bound_l612_61254

theorem trailing_zeros_bound (n : ℕ) : ∃ (k : ℕ), k ≤ 2 ∧ (1^n + 2^n + 3^n + 4^n) % 10^(k+1) ≠ 0 := by
  sorry

end trailing_zeros_bound_l612_61254


namespace kelly_wendy_ratio_l612_61243

def scholarship_problem (kelly wendy nina : ℕ) : Prop :=
  let total := 92000
  wendy = 20000 ∧
  ∃ n : ℕ, kelly = n * wendy ∧
  nina = kelly - 8000 ∧
  kelly + nina + wendy = total

theorem kelly_wendy_ratio :
  ∀ kelly wendy nina : ℕ,
  scholarship_problem kelly wendy nina →
  kelly / wendy = 2 :=
sorry

end kelly_wendy_ratio_l612_61243


namespace school_survey_is_stratified_sampling_l612_61280

/-- Represents a sampling method -/
inductive SamplingMethod
  | DrawingLots
  | RandomNumber
  | Systematic
  | Stratified

/-- Represents a population divided into groups -/
structure Population where
  totalSize : ℕ
  groups : List (ℕ × ℕ)  -- (group size, sample size) pairs

/-- Checks if a sampling method is stratified -/
def isStratifiedSampling (pop : Population) (method : SamplingMethod) : Prop :=
  method = SamplingMethod.Stratified ∧
  pop.groups.length ≥ 2 ∧
  (∀ (g₁ g₂ : ℕ × ℕ), g₁ ∈ pop.groups → g₂ ∈ pop.groups →
    (g₁.1 : ℚ) / (g₂.1 : ℚ) = (g₁.2 : ℚ) / (g₂.2 : ℚ))

/-- The main theorem to prove -/
theorem school_survey_is_stratified_sampling
  (totalStudents : ℕ)
  (maleStudents femaleStudents : ℕ)
  (maleSample femaleSample : ℕ)
  (h_total : totalStudents = maleStudents + femaleStudents)
  (h_male_ratio : (maleStudents : ℚ) / (totalStudents : ℚ) = 2 / 5)
  (h_female_ratio : (femaleStudents : ℚ) / (totalStudents : ℚ) = 3 / 5)
  (h_sample_ratio : (maleSample : ℚ) / (femaleSample : ℚ) = 2 / 3)
  : isStratifiedSampling
      { totalSize := totalStudents,
        groups := [(maleStudents, maleSample), (femaleStudents, femaleSample)] }
      SamplingMethod.Stratified :=
by sorry

end school_survey_is_stratified_sampling_l612_61280


namespace induction_sum_terms_l612_61233

theorem induction_sum_terms (k : ℕ) (h : k > 1) :
  (Finset.range (2^(k+1) - 1)).card - (Finset.range (2^k - 1)).card = 2^k := by
  sorry

end induction_sum_terms_l612_61233


namespace min_sum_unpainted_cells_l612_61272

/-- Represents a 10x10 table of natural numbers -/
def Table := Fin 10 → Fin 10 → ℕ

/-- Checks if a cell is a corner cell -/
def is_corner (i j : Fin 10) : Prop :=
  (i = 0 ∨ i = 9) ∧ (j = 0 ∨ j = 9)

/-- Checks if two cells are neighbors -/
def are_neighbors (i1 j1 i2 j2 : Fin 10) : Prop :=
  (i1 = i2 ∧ (j1 = j2 + 1 ∨ j1 + 1 = j2)) ∨
  (j1 = j2 ∧ (i1 = i2 + 1 ∨ i1 + 1 = i2))

/-- Checks if a cell should be painted based on its neighbors -/
def should_be_painted (t : Table) (i j : Fin 10) : Prop :=
  ∃ (i1 j1 i2 j2 : Fin 10), 
    are_neighbors i j i1 j1 ∧ 
    are_neighbors i j i2 j2 ∧ 
    t i j < t i1 j1 ∧ 
    t i j > t i2 j2

/-- The main theorem -/
theorem min_sum_unpainted_cells (t : Table) :
  (∃! (i1 j1 i2 j2 : Fin 10), 
    ¬is_corner i1 j1 ∧ 
    ¬is_corner i2 j2 ∧ 
    ¬should_be_painted t i1 j1 ∧ 
    ¬should_be_painted t i2 j2 ∧ 
    (∀ (i j : Fin 10), (i ≠ i1 ∨ j ≠ j1) ∧ (i ≠ i2 ∨ j ≠ j2) → should_be_painted t i j)) →
  (∃ (i1 j1 i2 j2 : Fin 10), 
    ¬is_corner i1 j1 ∧ 
    ¬is_corner i2 j2 ∧ 
    ¬should_be_painted t i1 j1 ∧ 
    ¬should_be_painted t i2 j2 ∧ 
    t i1 j1 + t i2 j2 = 3 ∧
    (∀ (k1 l1 k2 l2 : Fin 10), 
      ¬is_corner k1 l1 ∧ 
      ¬is_corner k2 l2 ∧ 
      ¬should_be_painted t k1 l1 ∧ 
      ¬should_be_painted t k2 l2 → 
      t k1 l1 + t k2 l2 ≥ 3)) :=
by sorry

end min_sum_unpainted_cells_l612_61272


namespace water_truck_capacity_l612_61270

/-- The maximum capacity of the water truck in tons -/
def truck_capacity : ℝ := 12

/-- The amount of water (in tons) injected by pipe A when used with pipe C -/
def water_A_with_C : ℝ := 4

/-- The amount of water (in tons) injected by pipe B when used with pipe C -/
def water_B_with_C : ℝ := 6

/-- The ratio of pipe B's injection rate to pipe A's injection rate -/
def rate_ratio_B_to_A : ℝ := 2

theorem water_truck_capacity :
  truck_capacity = water_A_with_C * rate_ratio_B_to_A ∧
  truck_capacity = water_B_with_C + water_A_with_C :=
by sorry

end water_truck_capacity_l612_61270


namespace right_triangle_area_l612_61201

theorem right_triangle_area (a b c : ℝ) (h1 : a + b = 4) (h2 : a^2 + b^2 = c^2) (h3 : c = 3) :
  (1/2) * a * b = 7/4 := by
  sorry

end right_triangle_area_l612_61201


namespace total_worksheets_l612_61299

/-- Given a teacher grading worksheets, this theorem proves the total number of worksheets. -/
theorem total_worksheets (problems_per_worksheet : ℕ) (graded_worksheets : ℕ) (remaining_problems : ℕ) : 
  problems_per_worksheet = 4 →
  graded_worksheets = 5 →
  remaining_problems = 16 →
  graded_worksheets + (remaining_problems / problems_per_worksheet) = 9 := by
  sorry

end total_worksheets_l612_61299


namespace correct_seasons_before_announcement_l612_61262

/-- The number of seasons before the announcement of a TV show. -/
def seasons_before_announcement : ℕ := 9

/-- The number of episodes in a regular season. -/
def regular_season_episodes : ℕ := 22

/-- The number of episodes in the last season. -/
def last_season_episodes : ℕ := 26

/-- The duration of each episode in hours. -/
def episode_duration : ℚ := 1/2

/-- The total watch time for all episodes in hours. -/
def total_watch_time : ℕ := 112

theorem correct_seasons_before_announcement :
  seasons_before_announcement * regular_season_episodes + last_season_episodes =
  total_watch_time / (episode_duration : ℚ) := by sorry

end correct_seasons_before_announcement_l612_61262


namespace second_player_eats_53_seeds_l612_61241

/-- The number of seeds eaten by the first player -/
def first_player_seeds : ℕ := 78

/-- The number of seeds eaten by the second player -/
def second_player_seeds : ℕ := 53

/-- The number of seeds eaten by the third player -/
def third_player_seeds : ℕ := second_player_seeds + 30

/-- The total number of seeds eaten by all players -/
def total_seeds : ℕ := 214

/-- Theorem stating that the given conditions result in the second player eating 53 seeds -/
theorem second_player_eats_53_seeds :
  first_player_seeds + second_player_seeds + third_player_seeds = total_seeds :=
by sorry

end second_player_eats_53_seeds_l612_61241


namespace absolute_value_inequality_l612_61211

theorem absolute_value_inequality (x y : ℝ) (h : x < y ∧ y < 0) :
  abs x > (abs (x + y)) / 2 ∧ (abs (x + y)) / 2 > abs y := by
  sorry

end absolute_value_inequality_l612_61211


namespace largest_multiple_of_seven_solution_is_correct_l612_61268

theorem largest_multiple_of_seven (n : ℤ) : 
  (n % 7 = 0 ∧ -n > -150) → n ≤ 147 :=
by sorry

theorem solution_is_correct : 
  147 % 7 = 0 ∧ -147 > -150 ∧ 
  ∀ m : ℤ, (m % 7 = 0 ∧ -m > -150) → m ≤ 147 :=
by sorry

end largest_multiple_of_seven_solution_is_correct_l612_61268


namespace total_soda_bottles_l612_61294

/-- The number of regular soda bottles -/
def regular_soda : ℕ := 49

/-- The number of diet soda bottles -/
def diet_soda : ℕ := 40

/-- Theorem: The total number of regular and diet soda bottles is 89 -/
theorem total_soda_bottles : regular_soda + diet_soda = 89 := by
  sorry

end total_soda_bottles_l612_61294


namespace two_power_and_factorial_l612_61239

theorem two_power_and_factorial (n : ℕ) :
  (¬ (2^n ∣ n!)) ∧ (∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, 2^(n-1) ∣ n!) := by
  sorry

end two_power_and_factorial_l612_61239


namespace f_derivative_at_2_l612_61290

noncomputable def f (x : ℝ) := 2 * x^3 - 2 * x^2 + 3

theorem f_derivative_at_2 : 
  (deriv f) 2 = 16 := by sorry

end f_derivative_at_2_l612_61290


namespace mrs_wilsborough_tickets_l612_61288

def prove_regular_tickets_bought : Prop :=
  let initial_savings : ℕ := 500
  let vip_ticket_cost : ℕ := 100
  let vip_tickets_bought : ℕ := 2
  let regular_ticket_cost : ℕ := 50
  let money_left : ℕ := 150
  let total_spent : ℕ := initial_savings - money_left
  let vip_tickets_total_cost : ℕ := vip_ticket_cost * vip_tickets_bought
  let regular_tickets_total_cost : ℕ := total_spent - vip_tickets_total_cost
  let regular_tickets_bought : ℕ := regular_tickets_total_cost / regular_ticket_cost
  regular_tickets_bought = 3

theorem mrs_wilsborough_tickets : prove_regular_tickets_bought := by
  sorry

end mrs_wilsborough_tickets_l612_61288


namespace movie_watching_time_l612_61293

/-- The duration of Bret's train ride to Boston -/
def total_duration : ℕ := 9

/-- The time Bret spends reading a book -/
def reading_time : ℕ := 2

/-- The time Bret spends eating dinner -/
def eating_time : ℕ := 1

/-- The time Bret has left for a nap -/
def nap_time : ℕ := 3

/-- Theorem stating that the time spent watching movies is 3 hours -/
theorem movie_watching_time :
  total_duration - (reading_time + eating_time + nap_time) = 3 := by
  sorry

end movie_watching_time_l612_61293


namespace single_weighing_correctness_check_l612_61271

/-- Represents a weight with its mass and marking -/
structure Weight where
  mass : ℝ
  marking : ℝ

/-- Represents the position of a weight on the scale -/
structure Position where
  weight : Weight
  distance : ℝ

/-- Calculates the moment of a weight at a given position -/
def moment (p : Position) : ℝ := p.weight.mass * p.distance

/-- Theorem: It's always possible to check if all markings are correct in a single weighing -/
theorem single_weighing_correctness_check 
  (weights : Finset Weight) 
  (hweights : weights.Nonempty) 
  (hmasses : ∀ w ∈ weights, ∃ w' ∈ weights, w.mass = w'.marking) 
  (hmarkings : ∀ w ∈ weights, ∃ w' ∈ weights, w.marking = w'.mass) :
  ∃ (left right : Finset Position),
    (∀ p ∈ left, p.weight ∈ weights) ∧
    (∀ p ∈ right, p.weight ∈ weights) ∧
    (left.sum moment = right.sum moment ↔ 
      ∀ w ∈ weights, w.mass = w.marking) :=
sorry

end single_weighing_correctness_check_l612_61271


namespace complex_square_root_l612_61250

theorem complex_square_root (z : ℂ) (h : z ^ 2 = 3 + 4 * I) :
  (z.im = 1 ∨ z.im = -1) ∧ Complex.abs z = Real.sqrt 5 := by sorry

end complex_square_root_l612_61250


namespace f_strictly_increasing_l612_61267

open Real

/-- The function f(x) = √3 sin x - cos x is strictly increasing in the intervals [-π/3 + 2kπ, 2π/3 + 2kπ], where k ∈ ℤ -/
theorem f_strictly_increasing (x : ℝ) :
  ∃ (k : ℤ), x ∈ Set.Icc (-π/3 + 2*π*k) (2*π/3 + 2*π*k) →
  StrictMonoOn (λ x => Real.sqrt 3 * sin x - cos x) (Set.Icc (-π/3 + 2*π*k) (2*π/3 + 2*π*k)) :=
by sorry

end f_strictly_increasing_l612_61267


namespace cistern_length_is_8_l612_61218

/-- Represents a cistern with water --/
structure Cistern where
  length : ℝ
  width : ℝ
  depth : ℝ
  wetSurfaceArea : ℝ

/-- Calculates the total wet surface area of a cistern --/
def totalWetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem: The length of the cistern is 8 meters --/
theorem cistern_length_is_8 (c : Cistern) 
    (h_width : c.width = 4)
    (h_depth : c.depth = 1.25)
    (h_area : c.wetSurfaceArea = 62) :
    c.length = 8 := by
  sorry

#check cistern_length_is_8

end cistern_length_is_8_l612_61218


namespace product_cost_l612_61246

/-- The cost of a product given its selling price and profit margin -/
theorem product_cost (x a : ℝ) (h : a > 0) :
  let selling_price := x
  let profit_margin := a / 100
  selling_price = (1 + profit_margin) * (selling_price / (1 + profit_margin)) :=
by sorry

end product_cost_l612_61246


namespace intersection_M_N_l612_61297

def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {-1, 0, 1}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end intersection_M_N_l612_61297


namespace wheel_distance_l612_61238

/-- Proves that a wheel rotating 10 times per minute and moving 20 cm per rotation will move 12000 cm in 1 hour -/
theorem wheel_distance (rotations_per_minute : ℕ) (cm_per_rotation : ℕ) (minutes_per_hour : ℕ) :
  rotations_per_minute = 10 →
  cm_per_rotation = 20 →
  minutes_per_hour = 60 →
  rotations_per_minute * minutes_per_hour * cm_per_rotation = 12000 := by
  sorry

#check wheel_distance

end wheel_distance_l612_61238


namespace optimal_solution_l612_61275

/-- Represents a vehicle type with its capacity, quantity, and fuel efficiency. -/
structure VehicleType where
  capacity : Nat
  quantity : Nat
  fuelEfficiency : Nat

/-- Represents the problem setup for the field trip. -/
structure FieldTripProblem where
  cars : VehicleType
  minivans : VehicleType
  buses : VehicleType
  totalPeople : Nat
  tripDistance : Nat

/-- Represents a solution to the field trip problem. -/
structure FieldTripSolution where
  numCars : Nat
  numMinivans : Nat
  numBuses : Nat

def fuelUsage (problem : FieldTripProblem) (solution : FieldTripSolution) : Rat :=
  (problem.tripDistance * solution.numCars / problem.cars.fuelEfficiency : Rat) +
  (problem.tripDistance * solution.numMinivans / problem.minivans.fuelEfficiency : Rat) +
  (problem.tripDistance * solution.numBuses / problem.buses.fuelEfficiency : Rat)

def totalCapacity (problem : FieldTripProblem) (solution : FieldTripSolution) : Nat :=
  solution.numCars * problem.cars.capacity +
  solution.numMinivans * problem.minivans.capacity +
  solution.numBuses * problem.buses.capacity

def isValidSolution (problem : FieldTripProblem) (solution : FieldTripSolution) : Prop :=
  solution.numCars ≤ problem.cars.quantity ∧
  solution.numMinivans ≤ problem.minivans.quantity ∧
  solution.numBuses ≤ problem.buses.quantity ∧
  totalCapacity problem solution ≥ problem.totalPeople

theorem optimal_solution (problem : FieldTripProblem) (solution : FieldTripSolution) :
  problem.cars = { capacity := 4, quantity := 3, fuelEfficiency := 30 } ∧
  problem.minivans = { capacity := 6, quantity := 2, fuelEfficiency := 20 } ∧
  problem.buses = { capacity := 20, quantity := 1, fuelEfficiency := 10 } ∧
  problem.totalPeople = 33 ∧
  problem.tripDistance = 50 ∧
  solution = { numCars := 1, numMinivans := 1, numBuses := 1 } ∧
  isValidSolution problem solution →
  ∀ (altSolution : FieldTripSolution),
    isValidSolution problem altSolution →
    fuelUsage problem solution ≤ fuelUsage problem altSolution :=
by sorry

end optimal_solution_l612_61275


namespace arithmetic_mean_of_fractions_l612_61208

theorem arithmetic_mean_of_fractions (x b : ℝ) (hx : x ≠ 0) :
  (1 / 2) * ((2*x + b) / x + (2*x - b) / x) = 2 := by
  sorry

end arithmetic_mean_of_fractions_l612_61208


namespace matrix_is_own_inverse_l612_61205

def A (c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![2, -2; c, d]

theorem matrix_is_own_inverse (c d : ℝ) :
  A c d * A c d = 1 ↔ c = 3/2 ∧ d = -2 := by sorry

end matrix_is_own_inverse_l612_61205


namespace circle_line_distance_l612_61207

theorem circle_line_distance (m : ℝ) : 
  (∃ (A B C : ℝ × ℝ), 
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    (A.1^2 + A.2^2 = 4) ∧ (B.1^2 + B.2^2 = 4) ∧ (C.1^2 + C.2^2 = 4) ∧
    (|A.2 + A.1 - m| / Real.sqrt 2 = 1) ∧
    (|B.2 + B.1 - m| / Real.sqrt 2 = 1) ∧
    (|C.2 + C.1 - m| / Real.sqrt 2 = 1)) →
  -Real.sqrt 2 ≤ m ∧ m ≤ Real.sqrt 2 :=
by sorry

end circle_line_distance_l612_61207


namespace range_of_m_when_p_true_range_of_m_when_p_or_q_true_and_p_and_q_false_l612_61257

-- Define proposition p
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 1 ≥ m

-- Define proposition q
def q (m : ℝ) : Prop := ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ (m - 2) * (m + 2) < 0 ∧
  ∀ x y : ℝ, x^2 / (m - 2) + y^2 / (m + 2) = 1 ↔ (x / a)^2 - (y / b)^2 = 1

-- Theorem 1
theorem range_of_m_when_p_true (m : ℝ) : p m → m ≤ 1 := by sorry

-- Theorem 2
theorem range_of_m_when_p_or_q_true_and_p_and_q_false (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m ≤ -2 ∨ (1 < m ∧ m < 2) := by sorry

end range_of_m_when_p_true_range_of_m_when_p_or_q_true_and_p_and_q_false_l612_61257


namespace sin_3theta_l612_61235

theorem sin_3theta (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (1 + Complex.I * Real.sqrt 2) / 2) : 
  Real.sin (3 * θ) = Real.sqrt 2 / 8 := by
  sorry

end sin_3theta_l612_61235


namespace certain_to_draw_black_ball_l612_61292

/-- Represents the number of black balls in the bag -/
def black_balls : ℕ := 6

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 3

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := black_balls + white_balls

/-- Represents the number of balls drawn -/
def drawn_balls : ℕ := 4

/-- Theorem stating that drawing at least one black ball is certain -/
theorem certain_to_draw_black_ball : 
  drawn_balls > white_balls → drawn_balls ≤ total_balls → true := by sorry

end certain_to_draw_black_ball_l612_61292


namespace history_or_geography_not_both_count_l612_61260

/-- The number of students taking both history and geography -/
def both : ℕ := 15

/-- The number of students taking history -/
def history : ℕ := 30

/-- The number of students taking geography only -/
def geography_only : ℕ := 12

/-- The number of students taking history or geography but not both -/
def history_or_geography_not_both : ℕ := (history - both) + geography_only

theorem history_or_geography_not_both_count : history_or_geography_not_both = 27 := by
  sorry

end history_or_geography_not_both_count_l612_61260


namespace nancys_apples_calculation_l612_61219

/-- The number of apples Nancy ate -/
def nancys_apples : ℝ := 3.0

/-- The number of apples Mike picked -/
def mikes_apples : ℝ := 7.0

/-- The number of apples Keith picked -/
def keiths_apples : ℝ := 6.0

/-- The number of apples left -/
def apples_left : ℝ := 10.0

/-- Theorem: Nancy's apples equals the total picked by Mike and Keith minus the apples left -/
theorem nancys_apples_calculation : 
  nancys_apples = mikes_apples + keiths_apples - apples_left := by
  sorry

end nancys_apples_calculation_l612_61219


namespace sum_100_from_neg_49_l612_61231

/-- Sum of consecutive integers -/
def sum_consecutive_integers (start : Int) (count : Nat) : Int :=
  count * (2 * start + count.pred) / 2

/-- Theorem: Sum of 100 consecutive integers from -49 is 50 -/
theorem sum_100_from_neg_49 : sum_consecutive_integers (-49) 100 = 50 := by
  sorry

end sum_100_from_neg_49_l612_61231


namespace product_of_fractions_is_zero_l612_61259

def fraction (n : ℕ) : ℚ := (n^3 - 1) / (n^3 + 1)

theorem product_of_fractions_is_zero :
  (fraction 1) * (fraction 2) * (fraction 3) * (fraction 4) = 0 := by
sorry

end product_of_fractions_is_zero_l612_61259


namespace plane_division_l612_61223

/-- Given m parallel lines and n non-parallel lines on a plane,
    where no more than two lines pass through any single point,
    the number of regions into which these lines divide the plane
    is 1 + (n(n+1))/2 + m(n+1). -/
theorem plane_division (m n : ℕ) : ℕ := by
  sorry

#check plane_division

end plane_division_l612_61223


namespace finite_decimal_consecutive_denominators_l612_61263

def is_finite_decimal (q : ℚ) : Prop :=
  ∃ (a b : ℤ) (k : ℕ), q = a / (b * 10^k) ∧ b ≠ 0

theorem finite_decimal_consecutive_denominators :
  ∀ n : ℕ, (is_finite_decimal (1 / n) ∧ is_finite_decimal (1 / (n + 1))) ↔ (n = 1 ∨ n = 4) :=
sorry

end finite_decimal_consecutive_denominators_l612_61263


namespace chef_manager_wage_difference_l612_61285

/-- Represents the hourly wages at Joe's Steakhouse -/
structure SteakhouseWages where
  manager : ℝ
  dishwasher : ℝ
  chef : ℝ

/-- The conditions for wages at Joe's Steakhouse -/
def wage_conditions (w : SteakhouseWages) : Prop :=
  w.manager = 8.50 ∧
  w.dishwasher = w.manager / 2 ∧
  w.chef = w.dishwasher * 1.22

theorem chef_manager_wage_difference (w : SteakhouseWages) 
  (h : wage_conditions w) : w.manager - w.chef = 3.315 := by
  sorry

#check chef_manager_wage_difference

end chef_manager_wage_difference_l612_61285


namespace books_per_shelf_l612_61265

theorem books_per_shelf 
  (total_shelves : ℕ) 
  (total_books : ℕ) 
  (h1 : total_shelves = 14240)
  (h2 : total_books = 113920) :
  total_books / total_shelves = 8 :=
by sorry

end books_per_shelf_l612_61265


namespace final_sum_after_fillings_l612_61203

/-- Represents the state of the blackboard after each filling -/
structure BoardState :=
  (numbers : List Int)
  (sum : Int)

/-- Perform one filling operation on the board -/
def fill (state : BoardState) : BoardState :=
  sorry

/-- The initial state of the board -/
def initial_state : BoardState :=
  { numbers := [2, 0, 2, 3], sum := 7 }

/-- Theorem stating the final sum after 2023 fillings -/
theorem final_sum_after_fillings :
  (Nat.iterate fill 2023 initial_state).sum = 2030 :=
sorry

end final_sum_after_fillings_l612_61203


namespace hawks_score_l612_61251

/-- Represents a basketball game between two teams -/
structure BasketballGame where
  total_score : ℕ
  winning_margin : ℕ

/-- Calculates the score of the losing team in a basketball game -/
def losing_team_score (game : BasketballGame) : ℕ :=
  (game.total_score - game.winning_margin) / 2

theorem hawks_score (game : BasketballGame) 
  (h1 : game.total_score = 82) 
  (h2 : game.winning_margin = 6) : 
  losing_team_score game = 38 := by
  sorry

end hawks_score_l612_61251


namespace value_of_a_l612_61221

theorem value_of_a (a b d : ℤ) 
  (h1 : a + b = d) 
  (h2 : b + d = 8) 
  (h3 : d = 4) : 
  a = 0 := by
sorry

end value_of_a_l612_61221


namespace cone_base_radius_l612_61276

/-- Represents a cone with given properties -/
structure Cone where
  surface_area : ℝ
  lateral_unfolds_semicircle : Prop

/-- Theorem: For a cone with surface area 12π and lateral surface that unfolds into a semicircle, 
    the radius of the base is 2 -/
theorem cone_base_radius 
  (cone : Cone) 
  (h1 : cone.surface_area = 12 * Real.pi) 
  (h2 : cone.lateral_unfolds_semicircle) : 
  ∃ (r : ℝ), r = 2 ∧ r > 0 ∧ 
  cone.surface_area = Real.pi * r^2 + Real.pi * r * (2 * r) := by
  sorry


end cone_base_radius_l612_61276


namespace milk_mixture_price_l612_61229

/-- Calculate the selling price of a milk-water mixture per litre -/
theorem milk_mixture_price (pure_milk_cost : ℝ) (pure_milk_volume : ℝ) (water_volume : ℝ) :
  pure_milk_cost = 3.60 →
  pure_milk_volume = 25 →
  water_volume = 5 →
  (pure_milk_cost * pure_milk_volume) / (pure_milk_volume + water_volume) = 3 := by
sorry


end milk_mixture_price_l612_61229


namespace number_value_relationship_l612_61295

theorem number_value_relationship (n v : ℝ) : 
  n > 0 → n = 7 → n - 4 = 21 * v → v = 1 / 7 := by sorry

end number_value_relationship_l612_61295


namespace sum_of_fractions_l612_61234

theorem sum_of_fractions : 
  (3 : ℚ) / 100 + (2 : ℚ) / 1000 + (8 : ℚ) / 10000 + (5 : ℚ) / 100000 = 0.03285 := by
  sorry

end sum_of_fractions_l612_61234


namespace factorization_theorem_1_l612_61222

theorem factorization_theorem_1 (x : ℝ) : 
  4 * (x - 2)^2 - 1 = (2*x - 3) * (2*x - 5) := by
  sorry

end factorization_theorem_1_l612_61222


namespace at_least_two_thirds_covered_l612_61213

/-- Represents a chessboard with dominoes -/
structure ChessboardWithDominoes where
  m : Nat
  n : Nat
  dominoes : Finset (Nat × Nat)
  m_ge_two : m ≥ 2
  n_ge_two : n ≥ 2
  valid_placement : ∀ (i j : Nat), (i, j) ∈ dominoes → 
    (i < m ∧ j < n) ∧ (
      ((i + 1, j) ∈ dominoes ∧ (i + 1) < m) ∨
      ((i, j + 1) ∈ dominoes ∧ (j + 1) < n)
    )
  no_overlap : ∀ (i j k l : Nat), (i, j) ∈ dominoes → (k, l) ∈ dominoes → 
    (i = k ∧ j = l) ∨ (i + 1 = k ∧ j = l) ∨ (i = k ∧ j + 1 = l) ∨
    (k + 1 = i ∧ j = l) ∨ (k = i ∧ l + 1 = j)
  no_more_addable : ∀ (i j : Nat), i < m → j < n → 
    (i, j) ∉ dominoes → (i + 1 < m → (i + 1, j) ∈ dominoes) ∧
    (j + 1 < n → (i, j + 1) ∈ dominoes)

/-- The main theorem stating that at least 2/3 of the chessboard is covered by dominoes -/
theorem at_least_two_thirds_covered (board : ChessboardWithDominoes) : 
  (2 : ℚ) / 3 * (board.m * board.n : ℚ) ≤ (board.dominoes.card * 2 : ℚ) := by
  sorry

end at_least_two_thirds_covered_l612_61213


namespace pet_store_theorem_l612_61274

/-- The number of ways to choose and assign different pets to four people -/
def pet_store_combinations : ℕ :=
  let puppies : ℕ := 12
  let kittens : ℕ := 10
  let hamsters : ℕ := 9
  let parrots : ℕ := 7
  let people : ℕ := 4
  puppies * kittens * hamsters * parrots * Nat.factorial people

/-- Theorem stating the number of combinations for the pet store problem -/
theorem pet_store_theorem : pet_store_combinations = 181440 := by
  sorry

end pet_store_theorem_l612_61274


namespace max_common_tangents_shared_focus_l612_61296

/-- Represents an ellipse in 2D space -/
structure Ellipse where
  foci : Fin 2 → ℝ × ℝ
  majorAxis : ℝ

/-- Represents a tangent line to an ellipse -/
structure Tangent where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Returns the number of common tangents between two ellipses -/
def commonTangents (e1 e2 : Ellipse) : ℕ := sorry

/-- Theorem: The maximum number of common tangents for two ellipses sharing one focus is 2 -/
theorem max_common_tangents_shared_focus (e1 e2 : Ellipse) 
  (h : e1.foci 1 = e2.foci 1) : 
  commonTangents e1 e2 ≤ 2 := by sorry

end max_common_tangents_shared_focus_l612_61296


namespace polynomial_symmetry_l612_61227

/-- Given a polynomial function f(x) = ax^5 + bx^3 + cx + 1 where f(2012) = 3,
    prove that f(-2012) = -1 -/
theorem polynomial_symmetry (a b c : ℝ) :
  let f := fun x => a * x^5 + b * x^3 + c * x + 1
  (f 2012 = 3) → (f (-2012) = -1) := by
  sorry

end polynomial_symmetry_l612_61227


namespace solve_system_l612_61242

theorem solve_system (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 5*t + 6) 
  (h3 : x = 1) : 
  y = 11 := by
  sorry

end solve_system_l612_61242


namespace total_age_is_47_l612_61256

/-- Given three people A, B, and C, where A is two years older than B, B is twice as old as C, 
    and B is 18 years old, prove that the total of their ages is 47 years. -/
theorem total_age_is_47 (A B C : ℕ) : 
  B = 18 → A = B + 2 → B = 2 * C → A + B + C = 47 := by sorry

end total_age_is_47_l612_61256


namespace min_values_xy_l612_61248

/-- Given positive real numbers x and y satisfying lg x + lg y = lg(x + y + 3),
    prove that the minimum value of xy is 9 and the minimum value of x + y is 6. -/
theorem min_values_xy (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : Real.log x + Real.log y = Real.log (x + y + 3)) :
  (∀ a b, a > 0 → b > 0 → Real.log a + Real.log b = Real.log (a + b + 3) → x * y ≤ a * b) ∧
  (∀ a b, a > 0 → b > 0 → Real.log a + Real.log b = Real.log (a + b + 3) → x + y ≤ a + b) ∧
  x * y = 9 ∧ x + y = 6 :=
sorry

end min_values_xy_l612_61248


namespace ship_placement_theorem_l612_61253

/-- Represents a ship on the grid -/
structure Ship :=
  (length : Nat)
  (width : Nat)

/-- Represents the grid -/
def Grid := Fin 10 → Fin 10 → Bool

/-- Checks if a ship placement is valid -/
def isValidPlacement (grid : Grid) (ship : Ship) (x y : Fin 10) : Bool :=
  sorry

/-- Places a ship on the grid -/
def placeShip (grid : Grid) (ship : Ship) (x y : Fin 10) : Grid :=
  sorry

/-- List of ships to be placed -/
def ships : List Ship :=
  [⟨4, 1⟩, ⟨3, 1⟩, ⟨3, 1⟩, ⟨2, 1⟩, ⟨2, 1⟩, ⟨2, 1⟩, ⟨1, 1⟩, ⟨1, 1⟩, ⟨1, 1⟩, ⟨1, 1⟩]

/-- Attempts to place all ships on the grid -/
def placeAllShips (grid : Grid) (ships : List Ship) : Option Grid :=
  sorry

theorem ship_placement_theorem :
  (∃ (grid : Grid), placeAllShips grid ships = some grid) ∧
  (∃ (grid : Grid), placeAllShips grid (ships.reverse) = none) :=
by sorry

end ship_placement_theorem_l612_61253


namespace farmer_cows_l612_61281

theorem farmer_cows (initial_cows : ℕ) (added_cows : ℕ) (sold_fraction : ℚ) 
  (h1 : initial_cows = 51)
  (h2 : added_cows = 5)
  (h3 : sold_fraction = 1/4) :
  initial_cows + added_cows - ⌊(initial_cows + added_cows : ℚ) * sold_fraction⌋ = 42 := by
  sorry

end farmer_cows_l612_61281


namespace total_cantaloupes_l612_61286

def fred_cantaloupes : ℕ := 38
def tim_cantaloupes : ℕ := 44

theorem total_cantaloupes : fred_cantaloupes + tim_cantaloupes = 82 := by
  sorry

end total_cantaloupes_l612_61286


namespace min_workers_proof_l612_61216

/-- The minimum number of workers in team A that satisfies the given conditions -/
def min_workers_A : ℕ := 153

/-- The number of workers team B transfers to team A -/
def workers_transferred : ℕ := (11 * min_workers_A - 1620) / 7

theorem min_workers_proof :
  (∀ a b : ℕ,
    (a ≥ min_workers_A) →
    (b + 90 = 2 * (a - 90)) →
    (a + workers_transferred = 6 * (b - workers_transferred)) →
    (workers_transferred > 0) →
    (∃ k : ℕ, a + 1 = 7 * k)) →
  (∀ a : ℕ,
    (a < min_workers_A) →
    (¬∃ b : ℕ,
      (b + 90 = 2 * (a - 90)) ∧
      (a + workers_transferred = 6 * (b - workers_transferred)) ∧
      (workers_transferred > 0))) :=
by sorry

end min_workers_proof_l612_61216


namespace ellipse_fixed_point_l612_61245

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Definition of a point on the ellipse -/
def point_on_ellipse (P : ℝ × ℝ) : Prop :=
  ellipse_C P.1 P.2

/-- Definition of the right focus -/
def right_focus : ℝ × ℝ := (1, 0)

/-- Definition of a line passing through the right focus -/
def line_through_focus (A B : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), A = right_focus + t • (B - right_focus) ∨
             B = right_focus + t • (A - right_focus)

/-- Definition of the dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- The main theorem to be proved -/
theorem ellipse_fixed_point :
  ∃ (M : ℝ × ℝ), M.1 = 5/4 ∧ M.2 = 0 ∧
  ∀ (A B : ℝ × ℝ), point_on_ellipse A → point_on_ellipse B →
  line_through_focus A B →
  dot_product (A.1 - M.1, A.2 - M.2) (B.1 - M.1, B.2 - M.2) = -7/16 :=
sorry

end ellipse_fixed_point_l612_61245


namespace base_seventeen_distinct_digits_l612_61273

/-- The number of three-digit numbers with distinct digits in base b -/
def distinctThreeDigitNumbers (b : ℕ) : ℕ := (b - 1) * (b - 1) * (b - 2)

/-- Theorem stating that there are exactly 256 three-digit numbers with distinct digits in base 17 -/
theorem base_seventeen_distinct_digits : 
  ∃ (b : ℕ), b > 2 ∧ distinctThreeDigitNumbers b = 256 ↔ b = 17 := by sorry

end base_seventeen_distinct_digits_l612_61273


namespace opposite_of_negative_one_third_l612_61277

theorem opposite_of_negative_one_third :
  -(-(1/3 : ℚ)) = 1/3 := by sorry

end opposite_of_negative_one_third_l612_61277


namespace reach_destination_in_time_l612_61204

/-- The distance to the destination in kilometers -/
def destination_distance : ℝ := 62

/-- The walking speed in km/hr -/
def walking_speed : ℝ := 5

/-- The car speed in km/hr -/
def car_speed : ℝ := 50

/-- The maximum time allowed to reach the destination in hours -/
def max_time : ℝ := 3

/-- A strategy represents a plan for A, B, and C to reach the destination -/
structure Strategy where
  -- Add necessary fields to represent the strategy
  dummy : Unit

/-- Calculates the time taken to execute a given strategy -/
def time_taken (s : Strategy) : ℝ :=
  -- Implement the calculation of time taken for the strategy
  sorry

/-- Theorem stating that there exists a strategy to reach the destination in less than the maximum allowed time -/
theorem reach_destination_in_time :
  ∃ (s : Strategy), time_taken s < max_time :=
sorry

end reach_destination_in_time_l612_61204


namespace cookies_per_guest_l612_61217

theorem cookies_per_guest (total_cookies : ℕ) (num_guests : ℕ) (h1 : total_cookies = 38) (h2 : num_guests = 2) :
  total_cookies / num_guests = 19 := by
  sorry

end cookies_per_guest_l612_61217


namespace division_problem_l612_61249

theorem division_problem : (150 : ℚ) / ((6 : ℚ) / 3) = 75 := by sorry

end division_problem_l612_61249


namespace total_blue_balloons_l612_61206

theorem total_blue_balloons (alyssa_balloons sandy_balloons sally_balloons : ℕ)
  (h1 : alyssa_balloons = 37)
  (h2 : sandy_balloons = 28)
  (h3 : sally_balloons = 39) :
  alyssa_balloons + sandy_balloons + sally_balloons = 104 := by
  sorry

end total_blue_balloons_l612_61206


namespace johnny_work_hours_l612_61200

/-- Given Johnny's hourly wage and total earnings, prove the number of hours he worked -/
theorem johnny_work_hours (hourly_wage : ℝ) (total_earnings : ℝ) (h1 : hourly_wage = 6.75) (h2 : total_earnings = 67.5) :
  total_earnings / hourly_wage = 10 := by
  sorry

end johnny_work_hours_l612_61200


namespace cubic_root_cube_relation_l612_61255

/-- Given a cubic polynomial f(x) = x^3 - 2x^2 + 5x - 3 with three distinct roots,
    and another cubic polynomial g(x) = x^3 + bx^2 + cx + d whose roots are
    the cubes of the roots of f(x), prove that b = -2, c = -5, and d = 3. -/
theorem cubic_root_cube_relation :
  let f (x : ℝ) := x^3 - 2*x^2 + 5*x - 3
  let g (x : ℝ) := x^3 + b*x^2 + c*x + d
  ∀ (b c d : ℝ),
  (∀ r : ℝ, f r = 0 → g (r^3) = 0) →
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) →
  b = -2 ∧ c = -5 ∧ d = 3 := by
sorry

end cubic_root_cube_relation_l612_61255


namespace circle_area_with_diameter_10_l612_61283

theorem circle_area_with_diameter_10 :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 25 * π :=
by sorry

end circle_area_with_diameter_10_l612_61283


namespace parentheses_placement_l612_61252

theorem parentheses_placement :
  (7 * (9 + 12 / 3) = 91) ∧
  ((7 * 9 + 12) / 3 = 25) ∧
  (7 * (9 + 12) / 3 = 49) ∧
  ((48 * 6) / (48 * 6) = 1) := by
  sorry

end parentheses_placement_l612_61252


namespace stratified_sample_size_l612_61240

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  stratum_size : ℕ
  stratum_sample : ℕ
  total_sample : ℕ

/-- Checks if the sampling is proportionally correct -/
def is_proportional_sampling (s : StratifiedSample) : Prop :=
  s.stratum_sample * s.total_population = s.total_sample * s.stratum_size

theorem stratified_sample_size 
  (s : StratifiedSample) 
  (h1 : s.total_population = 4320)
  (h2 : s.stratum_size = 1800)
  (h3 : s.stratum_sample = 45)
  (h4 : is_proportional_sampling s) :
  s.total_sample = 108 := by
  sorry

end stratified_sample_size_l612_61240


namespace solutions_count_no_solutions_when_a_less_than_neg_one_one_solution_when_a_eq_neg_one_two_solutions_when_a_between_neg_one_and_zero_one_solution_when_a_greater_than_zero_l612_61289

/-- The number of real solutions to the equation √(3a - 2x) + x = a depends on the value of a -/
theorem solutions_count (a : ℝ) :
  (∀ x, ¬ (Real.sqrt (3 * a - 2 * x) + x = a)) ∨
  (∃! x, Real.sqrt (3 * a - 2 * x) + x = a) ∨
  (∃ x y, x ≠ y ∧ Real.sqrt (3 * a - 2 * x) + x = a ∧ Real.sqrt (3 * a - 2 * y) + y = a) :=
by
  sorry

/-- For a < -1, there are no real solutions -/
theorem no_solutions_when_a_less_than_neg_one (a : ℝ) (h : a < -1) :
  ∀ x, ¬ (Real.sqrt (3 * a - 2 * x) + x = a) :=
by
  sorry

/-- For a = -1, there is exactly one real solution -/
theorem one_solution_when_a_eq_neg_one (a : ℝ) (h : a = -1) :
  ∃! x, Real.sqrt (3 * a - 2 * x) + x = a :=
by
  sorry

/-- For -1 < a ≤ 0, there are exactly two real solutions -/
theorem two_solutions_when_a_between_neg_one_and_zero (a : ℝ) (h1 : -1 < a) (h2 : a ≤ 0) :
  ∃ x y, x ≠ y ∧ Real.sqrt (3 * a - 2 * x) + x = a ∧ Real.sqrt (3 * a - 2 * y) + y = a :=
by
  sorry

/-- For a > 0, there is exactly one real solution -/
theorem one_solution_when_a_greater_than_zero (a : ℝ) (h : a > 0) :
  ∃! x, Real.sqrt (3 * a - 2 * x) + x = a :=
by
  sorry

end solutions_count_no_solutions_when_a_less_than_neg_one_one_solution_when_a_eq_neg_one_two_solutions_when_a_between_neg_one_and_zero_one_solution_when_a_greater_than_zero_l612_61289


namespace palindrome_probability_l612_61298

/-- A function that checks if a number is a palindrome -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- A function that generates all 5-digit palindromes -/
def fiveDigitPalindromes : Finset ℕ := sorry

/-- The total number of 5-digit palindromes -/
def totalPalindromes : ℕ := Finset.card fiveDigitPalindromes

/-- A function that checks if a number is divisible by another number -/
def isDivisibleBy (n m : ℕ) : Prop := n % m = 0

/-- The set of 5-digit palindromes m where m/7 is a palindrome and divisible by 11 -/
def validPalindromes : Finset ℕ := sorry

/-- The number of valid palindromes -/
def validCount : ℕ := Finset.card validPalindromes

theorem palindrome_probability :
  (validCount : ℚ) / totalPalindromes = 1 / 30 := by sorry

end palindrome_probability_l612_61298


namespace charley_beads_problem_l612_61232

theorem charley_beads_problem (white_beads black_beads : ℕ) 
  (black_fraction : ℚ) (total_pulled : ℕ) :
  white_beads = 51 →
  black_beads = 90 →
  black_fraction = 1 / 6 →
  total_pulled = 32 →
  ∃ (white_fraction : ℚ),
    white_fraction * white_beads + black_fraction * black_beads = total_pulled ∧
    white_fraction = 1 / 3 := by
  sorry

end charley_beads_problem_l612_61232


namespace circle_equation_l612_61261

/-- Given a circle C with radius 3 and center symmetric to (1,0) about y=x,
    prove that its standard equation is x^2 + (y-1)^2 = 9 -/
theorem circle_equation (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ C ↔ (x - center.1)^2 + (y - center.2)^2 = 3^2) →
  (center.1, center.2) = (0, 1) →
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + (y - 1)^2 = 9) :=
by sorry

end circle_equation_l612_61261


namespace f_periodic_l612_61215

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.tan (x / 2) + 1

theorem f_periodic (a : ℝ) (h : f (-a) = 11) : f (2 * Real.pi + a) = -9 := by
  sorry

end f_periodic_l612_61215
