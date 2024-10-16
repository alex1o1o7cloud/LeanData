import Mathlib

namespace NUMINAMATH_CALUDE_remaining_cheese_calories_l3468_346861

/-- Calculates the remaining calories in a block of cheese after a portion is removed -/
theorem remaining_cheese_calories (length width height : ℝ) 
  (calorie_density : ℝ) (eaten_side_length : ℝ) : 
  length = 4 → width = 8 → height = 2 → calorie_density = 110 → eaten_side_length = 2 →
  (length * width * height - eaten_side_length ^ 3) * calorie_density = 6160 := by
  sorry

#check remaining_cheese_calories

end NUMINAMATH_CALUDE_remaining_cheese_calories_l3468_346861


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_16_l3468_346852

theorem arithmetic_sqrt_16 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_16_l3468_346852


namespace NUMINAMATH_CALUDE_typing_time_proof_l3468_346838

def typing_time (original_speed : ℕ) (speed_reduction : ℕ) (document_length : ℕ) : ℕ :=
  document_length / (original_speed - speed_reduction)

theorem typing_time_proof (original_speed : ℕ) (speed_reduction : ℕ) (document_length : ℕ) 
  (h1 : original_speed = 65)
  (h2 : speed_reduction = 20)
  (h3 : document_length = 810)
  (h4 : original_speed > speed_reduction) :
  typing_time original_speed speed_reduction document_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_typing_time_proof_l3468_346838


namespace NUMINAMATH_CALUDE_krishans_money_l3468_346804

theorem krishans_money (x y : ℝ) (ram gopal krishan : ℝ) : 
  ram = 1503 →
  ram + gopal + krishan = 15000 →
  ram / (7 * x) = gopal / (17 * x) →
  ram / (7 * x) = krishan / (17 * y) →
  ∃ ε > 0, |krishan - 9845| < ε :=
by sorry

end NUMINAMATH_CALUDE_krishans_money_l3468_346804


namespace NUMINAMATH_CALUDE_donated_area_is_108_45_l3468_346842

/-- Calculates the total area of cloth donated given the areas and percentages of three cloths. -/
def total_donated_area (cloth1_area cloth2_area cloth3_area : ℝ)
  (cloth1_keep_percent cloth2_keep_percent cloth3_keep_percent : ℝ) : ℝ :=
  let cloth1_donate := cloth1_area * (1 - cloth1_keep_percent)
  let cloth2_donate := cloth2_area * (1 - cloth2_keep_percent)
  let cloth3_donate := cloth3_area * (1 - cloth3_keep_percent)
  cloth1_donate + cloth2_donate + cloth3_donate

/-- Theorem stating that the total donated area is 108.45 square inches. -/
theorem donated_area_is_108_45 :
  total_donated_area 100 65 48 0.4 0.55 0.6 = 108.45 := by
  sorry

end NUMINAMATH_CALUDE_donated_area_is_108_45_l3468_346842


namespace NUMINAMATH_CALUDE_open_box_volume_l3468_346825

/-- The volume of an open box formed by cutting squares from a rectangular sheet -/
theorem open_box_volume (sheet_length sheet_width cut_length : ℝ) 
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 36)
  (h3 : cut_length = 5) :
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 9880 :=
by sorry

end NUMINAMATH_CALUDE_open_box_volume_l3468_346825


namespace NUMINAMATH_CALUDE_friend_initial_savings_l3468_346845

/-- Proves that given the conditions of the savings problem, the friend's initial amount is $210 --/
theorem friend_initial_savings (your_initial : ℕ) (your_weekly : ℕ) (friend_weekly : ℕ) (weeks : ℕ) 
  (h1 : your_initial = 160)
  (h2 : your_weekly = 7)
  (h3 : friend_weekly = 5)
  (h4 : weeks = 25)
  (h5 : your_initial + your_weekly * weeks = friend_initial + friend_weekly * weeks) :
  friend_initial = 210 := by
  sorry

#check friend_initial_savings

end NUMINAMATH_CALUDE_friend_initial_savings_l3468_346845


namespace NUMINAMATH_CALUDE_percentage_to_pass_l3468_346848

/-- Given a test with maximum marks, a student's score, and the amount by which they failed,
    calculate the percentage of marks needed to pass the test. -/
theorem percentage_to_pass (max_marks student_score fail_by : ℕ) :
  max_marks = 300 →
  student_score = 80 →
  fail_by = 100 →
  (((student_score + fail_by : ℚ) / max_marks) * 100 : ℚ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_pass_l3468_346848


namespace NUMINAMATH_CALUDE_smallest_sum_is_14_l3468_346843

/-- Represents a pentagon arrangement of numbers 1 through 10 -/
structure PentagonArrangement where
  vertices : Fin 5 → Fin 10
  sides : Fin 5 → Fin 10
  all_used : ∀ n : Fin 10, (n ∈ Set.range vertices) ∨ (n ∈ Set.range sides)
  distinct : Function.Injective vertices ∧ Function.Injective sides

/-- The sum along each side of the pentagon -/
def side_sum (arr : PentagonArrangement) : ℕ → ℕ
| 0 => arr.vertices 0 + arr.sides 0 + arr.vertices 1
| 1 => arr.vertices 1 + arr.sides 1 + arr.vertices 2
| 2 => arr.vertices 2 + arr.sides 2 + arr.vertices 3
| 3 => arr.vertices 3 + arr.sides 3 + arr.vertices 4
| 4 => arr.vertices 4 + arr.sides 4 + arr.vertices 0
| _ => 0

/-- The arrangement is valid if all side sums are equal -/
def is_valid_arrangement (arr : PentagonArrangement) : Prop :=
  ∀ i j : Fin 5, side_sum arr i = side_sum arr j

/-- The main theorem: the smallest possible sum is 14 -/
theorem smallest_sum_is_14 :
  ∃ (arr : PentagonArrangement), is_valid_arrangement arr ∧
  (∀ i : Fin 5, side_sum arr i = 14) ∧
  (∀ arr' : PentagonArrangement, is_valid_arrangement arr' →
    ∀ i : Fin 5, side_sum arr' i ≥ 14) :=
  sorry

end NUMINAMATH_CALUDE_smallest_sum_is_14_l3468_346843


namespace NUMINAMATH_CALUDE_remainder_problem_l3468_346808

theorem remainder_problem (g : ℕ) (h1 : g = 101) (h2 : 4351 % g = 8) :
  5161 % g = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3468_346808


namespace NUMINAMATH_CALUDE_project_work_time_l3468_346873

/-- Calculates the time spent working on a project given the number of days,
    number of naps, hours per nap, and hours per day. -/
def time_spent_working (days : ℕ) (num_naps : ℕ) (hours_per_nap : ℕ) (hours_per_day : ℕ) : ℕ :=
  days * hours_per_day - num_naps * hours_per_nap

/-- Proves that given a 4-day project where 6 seven-hour naps are taken,
    and each day has 24 hours, the time spent working on the project is 54 hours. -/
theorem project_work_time :
  time_spent_working 4 6 7 24 = 54 := by
  sorry

end NUMINAMATH_CALUDE_project_work_time_l3468_346873


namespace NUMINAMATH_CALUDE_function_value_sum_l3468_346869

/-- A quadratic function f(x) with specific properties -/
def f (a : ℝ) (x : ℝ) : ℝ := a * (x + 2)^2 + 4

/-- The theorem stating the value of a + b + 2c for the given function -/
theorem function_value_sum (a : ℝ) :
  f a 0 = 5 ∧ f a 2 = 5 → a + 0 + 2 * 4 = 8.25 := by sorry

end NUMINAMATH_CALUDE_function_value_sum_l3468_346869


namespace NUMINAMATH_CALUDE_silver_dollars_problem_l3468_346866

theorem silver_dollars_problem (chiu phung ha : ℕ) : 
  phung = chiu + 16 →
  ha = phung + 5 →
  chiu + phung + ha = 205 →
  chiu = 56 := by
  sorry

end NUMINAMATH_CALUDE_silver_dollars_problem_l3468_346866


namespace NUMINAMATH_CALUDE_complement_of_intersection_l3468_346892

def U : Set ℕ := {1,2,3,4,5}
def M : Set ℕ := {1,2,4}
def N : Set ℕ := {3,4,5}

theorem complement_of_intersection :
  (M ∩ N)ᶜ = {1,2,3,5} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_intersection_l3468_346892


namespace NUMINAMATH_CALUDE_test_scores_l3468_346820

theorem test_scores (scores : Finset ℕ) (petya_score : ℕ) : 
  scores.card = 7317 →
  (∀ (x y : ℕ), x ∈ scores → y ∈ scores → x ≠ y) →
  (∀ (x y z : ℕ), x ∈ scores → y ∈ scores → z ∈ scores → x < y + z) →
  petya_score ∈ scores →
  petya_score > 15 :=
by sorry

end NUMINAMATH_CALUDE_test_scores_l3468_346820


namespace NUMINAMATH_CALUDE_inequality_proof_l3468_346890

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) :
  a * Real.sqrt b + b * Real.sqrt c + c * Real.sqrt a ≤ 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3468_346890


namespace NUMINAMATH_CALUDE_team_total_score_l3468_346816

def team_score (connor_score : ℕ) (amy_score : ℕ) (jason_score : ℕ) : ℕ :=
  connor_score + amy_score + jason_score

theorem team_total_score :
  ∀ (connor_score amy_score jason_score : ℕ),
    connor_score = 2 →
    amy_score = connor_score + 4 →
    jason_score = 2 * amy_score →
    team_score connor_score amy_score jason_score = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_team_total_score_l3468_346816


namespace NUMINAMATH_CALUDE_one_third_of_one_fourth_implies_three_tenths_l3468_346874

theorem one_third_of_one_fourth_implies_three_tenths (x : ℝ) : 
  (1 / 3) * (1 / 4) * x = 18 → (3 / 10) * x = 64.8 := by
sorry

end NUMINAMATH_CALUDE_one_third_of_one_fourth_implies_three_tenths_l3468_346874


namespace NUMINAMATH_CALUDE_square_of_integer_l3468_346840

theorem square_of_integer (x y : ℤ) (h : x + y = 10^18) :
  (x^2 * y^2) + ((x^2 + y^2) * (x + y)^2) = (x*y + x^2 + y^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_integer_l3468_346840


namespace NUMINAMATH_CALUDE_freds_marbles_l3468_346879

theorem freds_marbles (total : ℕ) (dark_blue red green : ℕ) : 
  dark_blue ≥ total / 3 →
  red = 38 →
  green = 4 →
  total = dark_blue + red + green →
  total ≥ 63 :=
by sorry

end NUMINAMATH_CALUDE_freds_marbles_l3468_346879


namespace NUMINAMATH_CALUDE_music_spending_l3468_346881

theorem music_spending (total_allowance : ℝ) (music_fraction : ℝ) : 
  total_allowance = 50 → music_fraction = 3/10 → music_fraction * total_allowance = 15 := by
  sorry

end NUMINAMATH_CALUDE_music_spending_l3468_346881


namespace NUMINAMATH_CALUDE_inequality_proof_l3468_346886

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_condition : a^2 + b^2 + c^2 + (a+b+c)^2 ≤ 4) :
  (a*b + 1) / (a+b)^2 + (b*c + 1) / (b+c)^2 + (c*a + 1) / (c+a)^2 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3468_346886


namespace NUMINAMATH_CALUDE_distance_foci_to_asymptotes_l3468_346862

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the foci
def foci : Set (ℝ × ℝ) := {(5, 0), (-5, 0)}

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := (3 * x - 4 * y = 0) ∨ (3 * x + 4 * y = 0)

-- Theorem statement
theorem distance_foci_to_asymptotes :
  ∀ (f : ℝ × ℝ) (x y : ℝ),
  f ∈ foci → asymptotes x y →
  ∃ (d : ℝ), d = 3 ∧ d = |3 * f.1 + 4 * f.2| / Real.sqrt 25 :=
sorry

end NUMINAMATH_CALUDE_distance_foci_to_asymptotes_l3468_346862


namespace NUMINAMATH_CALUDE_remaining_wire_length_l3468_346801

-- Define the initial wire length in centimeters
def initial_length_cm : ℝ := 23.3

-- Define the first cut in millimeters
def first_cut_mm : ℝ := 105

-- Define the second cut in centimeters
def second_cut_cm : ℝ := 4.6

-- Define the conversion factor from cm to mm
def cm_to_mm : ℝ := 10

-- Theorem statement
theorem remaining_wire_length :
  (initial_length_cm * cm_to_mm - first_cut_mm - second_cut_cm * cm_to_mm) = 82 := by
  sorry

end NUMINAMATH_CALUDE_remaining_wire_length_l3468_346801


namespace NUMINAMATH_CALUDE_systematic_sampling_eighth_group_l3468_346829

/-- Systematic sampling function -/
def systematicSample (totalEmployees : ℕ) (sampleSize : ℕ) (firstSample : ℕ) : ℕ → ℕ :=
  fun n => firstSample + (n - 1) * (totalEmployees / sampleSize)

theorem systematic_sampling_eighth_group 
  (totalEmployees : ℕ) 
  (sampleSize : ℕ) 
  (firstSample : ℕ) :
  totalEmployees = 200 →
  sampleSize = 40 →
  firstSample = 22 →
  systematicSample totalEmployees sampleSize firstSample 8 = 37 :=
by
  sorry

#eval systematicSample 200 40 22 8

end NUMINAMATH_CALUDE_systematic_sampling_eighth_group_l3468_346829


namespace NUMINAMATH_CALUDE_cone_volume_l3468_346819

/-- The volume of a cone with base diameter 12 cm and slant height 10 cm is 96π cubic centimeters -/
theorem cone_volume (π : ℝ) (diameter : ℝ) (slant_height : ℝ) : 
  diameter = 12 → slant_height = 10 → 
  (1 / 3) * π * ((diameter / 2) ^ 2) * (Real.sqrt (slant_height ^ 2 - (diameter / 2) ^ 2)) = 96 * π := by
  sorry


end NUMINAMATH_CALUDE_cone_volume_l3468_346819


namespace NUMINAMATH_CALUDE_passengers_who_got_off_l3468_346867

theorem passengers_who_got_off (initial : ℕ) (got_on : ℕ) (final : ℕ) : 
  initial = 28 → got_on = 7 → final = 26 → initial + got_on - final = 9 := by
  sorry

end NUMINAMATH_CALUDE_passengers_who_got_off_l3468_346867


namespace NUMINAMATH_CALUDE_digits_of_2_pow_120_l3468_346889

theorem digits_of_2_pow_120 (h : ∃ n : ℕ, 10^60 ≤ 2^200 ∧ 2^200 < 10^61) :
  ∃ n : ℕ, 10^36 ≤ 2^120 ∧ 2^120 < 10^37 :=
sorry

end NUMINAMATH_CALUDE_digits_of_2_pow_120_l3468_346889


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l3468_346812

def original_price : ℝ := 77.95
def sale_price : ℝ := 59.95

theorem price_decrease_percentage :
  let difference := original_price - sale_price
  let percentage_decrease := (difference / original_price) * 100
  ∃ ε > 0, abs (percentage_decrease - 23.08) < ε :=
sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l3468_346812


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l3468_346839

/-- Represents a cricket team with given properties -/
structure CricketTeam where
  size : ℕ
  captainAge : ℕ
  wicketKeeperAge : ℕ
  averageAge : ℚ
  remainingAverageAge : ℚ

/-- The age difference between the wicket keeper and the captain -/
def ageDifference (team : CricketTeam) : ℕ :=
  team.wicketKeeperAge - team.captainAge

/-- Theorem stating the properties of the cricket team and the age difference -/
theorem cricket_team_age_difference (team : CricketTeam) 
  (h1 : team.size = 11)
  (h2 : team.captainAge = 26)
  (h3 : team.wicketKeeperAge > team.captainAge)
  (h4 : team.averageAge = 24)
  (h5 : team.remainingAverageAge = team.averageAge - 1)
  : ageDifference team = 5 := by
  sorry


end NUMINAMATH_CALUDE_cricket_team_age_difference_l3468_346839


namespace NUMINAMATH_CALUDE_cone_from_sector_cone_sector_proof_l3468_346885

theorem cone_from_sector (sector_angle : Real) (circle_radius : Real) 
  (base_radius : Real) (slant_height : Real) : Prop :=
  sector_angle = 252 ∧
  circle_radius = 10 ∧
  base_radius = 7 ∧
  slant_height = 10 ∧
  2 * Real.pi * base_radius = (sector_angle / 360) * 2 * Real.pi * circle_radius ∧
  base_radius ^ 2 + (circle_radius ^ 2 - base_radius ^ 2) = slant_height ^ 2

theorem cone_sector_proof : 
  ∃ (sector_angle circle_radius base_radius slant_height : Real),
    cone_from_sector sector_angle circle_radius base_radius slant_height := by
  sorry

end NUMINAMATH_CALUDE_cone_from_sector_cone_sector_proof_l3468_346885


namespace NUMINAMATH_CALUDE_circle_radius_l3468_346837

theorem circle_radius (M N : ℝ) (h1 : M > 0) (h2 : N > 0) (h3 : M / N = 15) :
  ∃ r : ℝ, r > 0 ∧ M = π * r^2 ∧ N = 2 * π * r ∧ r = 30 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3468_346837


namespace NUMINAMATH_CALUDE_vertices_must_be_even_l3468_346865

-- Define a polyhedron
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

-- Define a property for trihedral angles
def has_trihedral_angles (p : Polyhedron) : Prop :=
  3 * p.vertices = 2 * p.edges

-- Theorem statement
theorem vertices_must_be_even (p : Polyhedron) 
  (h : has_trihedral_angles p) : Even p.vertices := by
  sorry


end NUMINAMATH_CALUDE_vertices_must_be_even_l3468_346865


namespace NUMINAMATH_CALUDE_zacks_countries_l3468_346853

theorem zacks_countries (alex george joseph patrick zack : ℕ) : 
  alex = 24 →
  george = alex / 4 →
  joseph = george / 2 →
  patrick = joseph * 3 →
  zack = patrick * 2 →
  zack = 18 :=
by sorry

end NUMINAMATH_CALUDE_zacks_countries_l3468_346853


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l3468_346809

theorem line_intercepts_sum (c : ℝ) : 
  (∃ (x y : ℝ), 6*x + 9*y + c = 0 ∧ x + y = 30) → c = -108 := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l3468_346809


namespace NUMINAMATH_CALUDE_olivias_race_time_l3468_346854

def total_time : ℕ := 112  -- 1 hour 52 minutes in minutes

theorem olivias_race_time (olivia_time : ℕ) 
  (h1 : olivia_time + (olivia_time - 4) = total_time) : 
  olivia_time = 58 := by
  sorry

end NUMINAMATH_CALUDE_olivias_race_time_l3468_346854


namespace NUMINAMATH_CALUDE_jacket_sale_profit_l3468_346863

/-- Calculates the merchant's gross profit for a jacket sale -/
theorem jacket_sale_profit (purchase_price : ℝ) (markup_percent : ℝ) (discount_percent : ℝ) : 
  purchase_price = 60 ∧ 
  markup_percent = 0.25 ∧ 
  discount_percent = 0.20 → 
  let selling_price := purchase_price / (1 - markup_percent)
  let discounted_price := selling_price * (1 - discount_percent)
  discounted_price - purchase_price = 4 :=
by sorry

end NUMINAMATH_CALUDE_jacket_sale_profit_l3468_346863


namespace NUMINAMATH_CALUDE_representable_set_l3468_346823

def representable (k : ℕ) : Prop :=
  ∃ x y z : ℕ+, k = (x + y + z)^2 / (x * y * z)

theorem representable_set : 
  {k : ℕ | representable k} = {1, 2, 3, 4, 5, 6, 8, 9} :=
by sorry

end NUMINAMATH_CALUDE_representable_set_l3468_346823


namespace NUMINAMATH_CALUDE_sqrt_21_bounds_l3468_346826

theorem sqrt_21_bounds : 4 < Real.sqrt 21 ∧ Real.sqrt 21 < 5 := by sorry

end NUMINAMATH_CALUDE_sqrt_21_bounds_l3468_346826


namespace NUMINAMATH_CALUDE_inequalities_from_sum_positive_l3468_346897

theorem inequalities_from_sum_positive (a b : ℝ) (h : a + b > 0) :
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a+2)*(b+2) > a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_from_sum_positive_l3468_346897


namespace NUMINAMATH_CALUDE_fill_tank_theorem_l3468_346807

/-- Calculates the remaining water needed to fill a tank -/
def remaining_water (tank_capacity : ℕ) (pour_rate : ℚ) (pour_time : ℕ) : ℕ :=
  tank_capacity - (pour_time / 15 : ℕ)

/-- Theorem: Given a 150-gallon tank, pouring water at 1 gallon per 15 seconds for 525 seconds,
    the remaining water needed to fill the tank is 115 gallons -/
theorem fill_tank_theorem :
  remaining_water 150 (1/15 : ℚ) 525 = 115 := by
  sorry

end NUMINAMATH_CALUDE_fill_tank_theorem_l3468_346807


namespace NUMINAMATH_CALUDE_find_n_l3468_346894

theorem find_n : ∃ n : ℤ, (5 : ℝ) ^ (2 * n) = (1 / 5 : ℝ) ^ (n - 12) ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l3468_346894


namespace NUMINAMATH_CALUDE_max_marks_proof_l3468_346846

/-- Given a maximum mark M, calculate the passing mark as 60% of M -/
def passing_mark (M : ℝ) : ℝ := 0.6 * M

/-- The maximum marks for an exam -/
def M : ℝ := 300

/-- The marks obtained by the student -/
def obtained_marks : ℝ := 160

/-- The number of marks by which the student failed -/
def failed_by : ℝ := 20

theorem max_marks_proof :
  passing_mark M = obtained_marks + failed_by :=
sorry

end NUMINAMATH_CALUDE_max_marks_proof_l3468_346846


namespace NUMINAMATH_CALUDE_marble_selection_ways_l3468_346882

theorem marble_selection_ways (total_marbles : ℕ) (special_marbles : ℕ) (selection_size : ℕ) 
  (h1 : total_marbles = 15)
  (h2 : special_marbles = 6)
  (h3 : selection_size = 5) :
  (special_marbles : ℕ) * Nat.choose (total_marbles - special_marbles) (selection_size - 1) = 756 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_ways_l3468_346882


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3468_346810

/-- Two vectors in R² are perpendicular if their dot product is zero -/
def isPerpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem perpendicular_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, -2)
  isPerpendicular a b → x = 4 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3468_346810


namespace NUMINAMATH_CALUDE_hyperbola_n_range_l3468_346868

/-- Represents a hyperbola with parameters m and n -/
structure Hyperbola (m n : ℝ) where
  eq : ∀ x y : ℝ, x^2 / (m^2 + n) - y^2 / (3 * m^2 - n) = 1

/-- The distance between the foci of the hyperbola is 4 -/
def foci_distance (m n : ℝ) : Prop :=
  (m^2 + n) + (3 * m^2 - n) = 4

/-- The theorem stating the range of n for the given hyperbola -/
theorem hyperbola_n_range (m n : ℝ) (h : Hyperbola m n) (d : foci_distance m n) :
  -1 < n ∧ n < 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_n_range_l3468_346868


namespace NUMINAMATH_CALUDE_initial_gasohol_volume_l3468_346833

/-- Represents the composition of a fuel mixture -/
structure FuelMixture where
  ethanol : ℝ
  gasoline : ℝ
  valid : ethanol + gasoline = 1

/-- Represents the state of the fuel tank -/
structure FuelTank where
  volume : ℝ
  mixture : FuelMixture

def initial_mixture : FuelMixture := {
  ethanol := 0.05,
  gasoline := 0.95,
  valid := by norm_num
}

def desired_mixture : FuelMixture := {
  ethanol := 0.1,
  gasoline := 0.9,
  valid := by norm_num
}

def ethanol_added : ℝ := 2

theorem initial_gasohol_volume (initial : FuelTank) :
  initial.mixture = initial_mixture →
  (∃ (final : FuelTank), 
    final.volume = initial.volume + ethanol_added ∧
    final.mixture = desired_mixture) →
  initial.volume = 36 := by
  sorry

end NUMINAMATH_CALUDE_initial_gasohol_volume_l3468_346833


namespace NUMINAMATH_CALUDE_library_visitors_average_l3468_346898

/-- Calculates the average number of visitors per day in a 30-day month starting on a Sunday -/
def averageVisitors (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalSundays : ℕ := 4
  let totalOtherDays : ℕ := 26
  let totalDays : ℕ := 30
  let totalVisitors : ℕ := sundayVisitors * totalSundays + otherDayVisitors * totalOtherDays
  (totalVisitors : ℚ) / totalDays

theorem library_visitors_average (sundayVisitors : ℕ) (otherDayVisitors : ℕ) 
    (h1 : sundayVisitors = 510) (h2 : otherDayVisitors = 240) : 
    averageVisitors sundayVisitors otherDayVisitors = 276 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_average_l3468_346898


namespace NUMINAMATH_CALUDE_sector_properties_l3468_346815

/-- Represents a circular sector --/
structure Sector where
  radius : ℝ
  centralAngle : ℝ

/-- Calculates the perimeter of a sector --/
def sectorPerimeter (s : Sector) : ℝ :=
  2 * s.radius + s.radius * s.centralAngle

/-- Calculates the area of a sector --/
def sectorArea (s : Sector) : ℝ :=
  0.5 * s.radius * s.radius * s.centralAngle

theorem sector_properties :
  ∃ (s : Sector),
    sectorPerimeter s = 8 ∧
    (s.centralAngle = 2 → sectorArea s = 4) ∧
    (∀ (t : Sector), sectorPerimeter t = 8 → sectorArea t ≤ 4) ∧
    (sectorArea s = 4 ∧ s.centralAngle = 2) := by
  sorry

end NUMINAMATH_CALUDE_sector_properties_l3468_346815


namespace NUMINAMATH_CALUDE_simplify_expression_l3468_346875

theorem simplify_expression (x : ℝ) : (3 * x + 20) + (97 * x + 30) = 100 * x + 50 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3468_346875


namespace NUMINAMATH_CALUDE_flight_duration_sum_l3468_346860

/-- Represents a time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes, accounting for time zone difference -/
def timeDifference (departure : Time) (arrival : Time) (timeZoneDiff : ℤ) : ℕ :=
  let totalMinutes := (arrival.hours - departure.hours) * 60 + arrival.minutes - departure.minutes
  (totalMinutes + timeZoneDiff * 60).toNat

/-- Theorem stating the flight duration property -/
theorem flight_duration_sum (departureTime : Time) (arrivalTime : Time) 
    (h : ℕ) (m : ℕ) (mPos : 0 < m) (mLt60 : m < 60) :
    departureTime.hours = 15 ∧ departureTime.minutes = 15 →
    arrivalTime.hours = 16 ∧ arrivalTime.minutes = 50 →
    timeDifference departureTime arrivalTime (-1) = h * 60 + m →
    h + m = 36 := by
  sorry

end NUMINAMATH_CALUDE_flight_duration_sum_l3468_346860


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3468_346870

open Set

def U : Set Nat := {1,2,3,4,5,6,7,8}
def A : Set Nat := {1,2,3}
def B : Set Nat := {2,3,4,5}

theorem complement_intersection_theorem : 
  (Aᶜ ∪ Bᶜ) ∩ U = {1,4,5,6,7,8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3468_346870


namespace NUMINAMATH_CALUDE_two_digit_number_theorem_l3468_346821

/-- Represents a two-digit number with specific properties -/
structure TwoDigitNumber where
  x : ℕ  -- tens digit
  -- Ensure x is a single digit
  h1 : x ≥ 1 ∧ x ≤ 9
  -- Ensure the units digit is non-negative
  h2 : 2 * x ≥ 3

/-- The value of the two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : ℕ :=
  10 * n.x + (2 * n.x - 3)

theorem two_digit_number_theorem (n : TwoDigitNumber) :
  n.value = 12 * n.x - 3 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_theorem_l3468_346821


namespace NUMINAMATH_CALUDE_nth_equation_l3468_346858

theorem nth_equation (n : ℕ) (h : n > 0) :
  (n + 2 : ℚ) / n - 2 / (n + 2) = ((n + 2)^2 + n^2 : ℚ) / (n * (n + 2)) - 1 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_l3468_346858


namespace NUMINAMATH_CALUDE_baker_remaining_pastries_l3468_346800

-- Define the initial number of pastries and the number of pastries sold
def initial_pastries : ℕ := 56
def sold_pastries : ℕ := 29

-- Define the function to calculate remaining pastries
def remaining_pastries : ℕ := initial_pastries - sold_pastries

-- Theorem statement
theorem baker_remaining_pastries : remaining_pastries = 27 := by
  sorry

end NUMINAMATH_CALUDE_baker_remaining_pastries_l3468_346800


namespace NUMINAMATH_CALUDE_square_root_problem_l3468_346832

theorem square_root_problem (a b : ℝ) 
  (h1 : Real.sqrt (a + 3) = 3) 
  (h2 : (3 * b - 2) ^ (1/3 : ℝ) = 2) : 
  Real.sqrt (a + 3*b) = 6 := by
sorry

end NUMINAMATH_CALUDE_square_root_problem_l3468_346832


namespace NUMINAMATH_CALUDE_car_start_time_difference_l3468_346864

/-- Two cars traveling at the same speed with specific distance ratios at different times --/
theorem car_start_time_difference
  (speed : ℝ)
  (distance_ratio_10am : ℝ)
  (distance_ratio_12pm : ℝ)
  (h1 : speed > 0)
  (h2 : distance_ratio_10am = 5)
  (h3 : distance_ratio_12pm = 3)
  : ∃ (start_time_diff : ℝ),
    start_time_diff = 8 ∧
    distance_ratio_10am * (10 - start_time_diff) = 10 ∧
    distance_ratio_12pm * (12 - start_time_diff) = 12 :=
by sorry

end NUMINAMATH_CALUDE_car_start_time_difference_l3468_346864


namespace NUMINAMATH_CALUDE_third_quadrant_angle_tangent_l3468_346805

theorem third_quadrant_angle_tangent (α β : Real) : 
  (2 * Real.pi - Real.pi < α) ∧ (α < 2 * Real.pi - Real.pi/2) →
  (Real.sin (α + β) * Real.cos β - Real.sin β * Real.cos (α + β) = -12/13) →
  Real.tan (α/2) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_third_quadrant_angle_tangent_l3468_346805


namespace NUMINAMATH_CALUDE_shoe_pair_difference_l3468_346828

theorem shoe_pair_difference (ellie_shoes riley_shoes : ℕ) : 
  ellie_shoes = 8 →
  riley_shoes < ellie_shoes →
  ellie_shoes + riley_shoes = 13 →
  ellie_shoes - riley_shoes = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_shoe_pair_difference_l3468_346828


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3468_346887

theorem equal_roots_quadratic (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 4 = 0 ∧ 
   ∀ y : ℝ, y^2 + a*y + 4 = 0 → y = x) → 
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3468_346887


namespace NUMINAMATH_CALUDE_two_digit_number_puzzle_l3468_346895

theorem two_digit_number_puzzle :
  ∀ x y : ℕ,
  x < 10 ∧ y < 10 ∧  -- Ensuring x and y are single digits
  x + y = 7 ∧  -- Sum of digits is 7
  (x + 2) + 10 * (y + 2) = 2 * (10 * y + x) - 3  -- Condition after adding 2 to each digit
  → 10 * y + x = 25 :=  -- The original number is 25
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_puzzle_l3468_346895


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l3468_346841

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2/9 - y^2 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := y = (1/3)*(x + 1)

/-- The number of intersection points -/
def intersection_count : ℕ := 1

/-- Theorem stating that the number of intersection points between the hyperbola and the line is 1 -/
theorem hyperbola_line_intersection :
  ∃! n : ℕ, n = intersection_count ∧ 
  (∃ (x y : ℝ), hyperbola x y ∧ line x y) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), hyperbola x₁ y₁ ∧ line x₁ y₁ ∧ hyperbola x₂ y₂ ∧ line x₂ y₂ → x₁ = x₂ ∧ y₁ = y₂) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l3468_346841


namespace NUMINAMATH_CALUDE_book_length_l3468_346831

theorem book_length (pages_read : ℚ) (pages_remaining : ℚ) (total_pages : ℚ) : 
  pages_read = (2 : ℚ) / 3 * total_pages →
  pages_remaining = (1 : ℚ) / 3 * total_pages →
  pages_read = pages_remaining + 30 →
  total_pages = 90 := by
sorry

end NUMINAMATH_CALUDE_book_length_l3468_346831


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3468_346855

def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3468_346855


namespace NUMINAMATH_CALUDE_initial_geese_count_l3468_346814

theorem initial_geese_count (initial_count : ℕ) : 
  (initial_count / 2 + 4 = 12) → initial_count = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_geese_count_l3468_346814


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l3468_346883

theorem divisibility_equivalence (n : ℕ) : 
  7 ∣ (3^n + n^3) ↔ 7 ∣ (3^n * n^3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l3468_346883


namespace NUMINAMATH_CALUDE_cone_slant_height_l3468_346851

/-- The slant height of a cone given its base circumference and lateral surface sector angle -/
theorem cone_slant_height (base_circumference : ℝ) (sector_angle : ℝ) : 
  base_circumference = 2 * Real.pi → sector_angle = 120 → 3 = 
    (base_circumference * 180) / (sector_angle * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cone_slant_height_l3468_346851


namespace NUMINAMATH_CALUDE_gcd_102_238_l3468_346835

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l3468_346835


namespace NUMINAMATH_CALUDE_fisherman_catch_l3468_346878

/-- The number of fish caught by a fisherman -/
def total_fish (num_boxes : ℕ) (fish_per_box : ℕ) (fish_outside : ℕ) : ℕ :=
  num_boxes * fish_per_box + fish_outside

/-- Theorem stating the total number of fish caught by the fisherman -/
theorem fisherman_catch :
  total_fish 15 20 6 = 306 := by
  sorry

end NUMINAMATH_CALUDE_fisherman_catch_l3468_346878


namespace NUMINAMATH_CALUDE_sunday_price_calculation_l3468_346856

def original_price : ℝ := 250
def regular_discount : ℝ := 0.4
def sunday_discount : ℝ := 0.25

theorem sunday_price_calculation : 
  original_price * (1 - regular_discount) * (1 - sunday_discount) = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_sunday_price_calculation_l3468_346856


namespace NUMINAMATH_CALUDE_prob_two_odd_dice_l3468_346850

/-- The number of faces on a standard die -/
def num_faces : ℕ := 6

/-- The number of odd faces on a standard die -/
def num_odd_faces : ℕ := 3

/-- The total number of possible outcomes when throwing two dice -/
def total_outcomes : ℕ := num_faces * num_faces

/-- The number of outcomes where both dice show odd numbers -/
def favorable_outcomes : ℕ := num_odd_faces * num_odd_faces

/-- The probability of rolling two odd numbers when throwing two dice simultaneously -/
theorem prob_two_odd_dice : 
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_odd_dice_l3468_346850


namespace NUMINAMATH_CALUDE_survey_result_l3468_346813

theorem survey_result (U : Finset Int) (A B : Finset Int) 
  (h1 : Finset.card U = 70)
  (h2 : Finset.card A = 37)
  (h3 : Finset.card B = 49)
  (h4 : Finset.card (A ∩ B) = 20) :
  Finset.card (U \ (A ∪ B)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_survey_result_l3468_346813


namespace NUMINAMATH_CALUDE_fraction_division_addition_l3468_346884

theorem fraction_division_addition : (3 / 7 : ℚ) / 4 + 1 / 2 = 17 / 28 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_addition_l3468_346884


namespace NUMINAMATH_CALUDE_division_by_fraction_not_always_larger_l3468_346818

theorem division_by_fraction_not_always_larger : ∃ (a b c : ℚ), b ≠ 0 ∧ c ≠ 0 ∧ (a / (b / c)) ≤ a := by
  sorry

end NUMINAMATH_CALUDE_division_by_fraction_not_always_larger_l3468_346818


namespace NUMINAMATH_CALUDE_empty_board_prob_2013_l3468_346824

/-- Represents the state of the blackboard -/
inductive BoardState
| Empty : BoardState
| NonEmpty : Nat → BoardState

/-- The rules for updating the blackboard based on a coin flip -/
def updateBoard (state : BoardState) (n : Nat) (isHeads : Bool) : BoardState :=
  match state, isHeads with
  | BoardState.Empty, true => BoardState.NonEmpty n
  | BoardState.NonEmpty m, true => 
      if (m^2 + 2*n^2) % 3 = 0 then BoardState.Empty else BoardState.NonEmpty n
  | _, false => state

/-- The probability of an empty blackboard after n flips -/
def emptyBoardProb (n : Nat) : ℚ :=
  sorry  -- Definition omitted for brevity

theorem empty_board_prob_2013 :
  ∃ (u v : ℕ), emptyBoardProb 2013 = (2 * u + 1) / (2^1336 * (2 * v + 1)) :=
sorry

#check empty_board_prob_2013

end NUMINAMATH_CALUDE_empty_board_prob_2013_l3468_346824


namespace NUMINAMATH_CALUDE_arcsin_one_half_l3468_346802

theorem arcsin_one_half : Real.arcsin (1/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_half_l3468_346802


namespace NUMINAMATH_CALUDE_no_solution_exists_l3468_346806

theorem no_solution_exists : ¬ ∃ (a : ℝ), 
  ({0, 1} : Set ℝ) ∩ ({11 - a, Real.log a, 2^a, a} : Set ℝ) = {1} := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3468_346806


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l3468_346877

theorem largest_integer_with_remainder : ∃ n : ℕ, n = 95 ∧ 
  n < 100 ∧ 
  n % 7 = 4 ∧ 
  ∀ m : ℕ, m < 100 → m % 7 = 4 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l3468_346877


namespace NUMINAMATH_CALUDE_probability_factor_less_than_8_of_90_l3468_346849

def positive_factors (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ x => x > 0 ∧ n % x = 0)

def factors_less_than (n k : ℕ) : Finset ℕ :=
  (positive_factors n).filter (λ x => x < k)

theorem probability_factor_less_than_8_of_90 :
  (factors_less_than 90 8).card / (positive_factors 90).card = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_factor_less_than_8_of_90_l3468_346849


namespace NUMINAMATH_CALUDE_right_triangle_rotation_forms_cone_l3468_346811

/-- A right-angled triangle -/
structure RightTriangle where
  /-- One of the right-angled edges of the triangle -/
  edge : ℝ
  /-- The other right-angled edge of the triangle -/
  base : ℝ
  /-- The hypotenuse of the triangle -/
  hypotenuse : ℝ
  /-- Condition for a right-angled triangle -/
  right_angle : edge^2 + base^2 = hypotenuse^2

/-- A solid formed by rotating a plane figure -/
inductive RotatedSolid
  | Cone
  | Cylinder
  | Sphere

/-- Function to determine the solid formed by rotating a right-angled triangle -/
def solidFormedByRotation (triangle : RightTriangle) (rotationAxis : ℝ) : RotatedSolid :=
  sorry

/-- Theorem stating that rotating a right-angled triangle about one of its right-angled edges forms a cone -/
theorem right_triangle_rotation_forms_cone (triangle : RightTriangle) :
  solidFormedByRotation triangle triangle.edge = RotatedSolid.Cone := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_rotation_forms_cone_l3468_346811


namespace NUMINAMATH_CALUDE_f_satisfies_conditions_l3468_346891

def f (x : ℝ) := x^2

theorem f_satisfies_conditions :
  (∀ x y, x < y ∧ y < -1 → f x > f y) ∧
  (∀ x, f x = f (-x)) ∧
  (∃ m, ∀ x, f m ≤ f x) := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_conditions_l3468_346891


namespace NUMINAMATH_CALUDE_sum_of_odd_numbers_sum_of_cubes_l3468_346880

def u (n : ℕ) : ℕ := (2 * n - 1) + if n = 0 then 0 else u (n - 1)

def S (n : ℕ) : ℕ := n^3 + if n = 0 then 0 else S (n - 1)

theorem sum_of_odd_numbers (n : ℕ) : u n = n^2 := by
  sorry

theorem sum_of_cubes (n : ℕ) : S n = (n * (n + 1) / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_odd_numbers_sum_of_cubes_l3468_346880


namespace NUMINAMATH_CALUDE_not_square_of_two_pow_minus_one_l3468_346896

theorem not_square_of_two_pow_minus_one (n : ℕ) (h : n > 1) :
  ¬ ∃ k : ℕ, 2^n - 1 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_square_of_two_pow_minus_one_l3468_346896


namespace NUMINAMATH_CALUDE_largest_multiple_of_11_less_than_neg_150_l3468_346834

theorem largest_multiple_of_11_less_than_neg_150 :
  ∀ n : ℤ, n * 11 < -150 → n * 11 ≤ -154 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_11_less_than_neg_150_l3468_346834


namespace NUMINAMATH_CALUDE_point_on_x_axis_l3468_346876

theorem point_on_x_axis (m : ℝ) : 
  (∃ x : ℝ, (x = m - 1 ∧ 0 = 2 * m + 3)) → m = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l3468_346876


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_sq_l3468_346859

theorem abs_eq_sqrt_sq (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_sq_l3468_346859


namespace NUMINAMATH_CALUDE_polynomial_value_constraint_l3468_346827

theorem polynomial_value_constraint (P : ℤ → ℤ) (a b c d : ℤ) :
  (∃ (n : ℤ → ℤ), P = n) →
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
  P a = 1979 →
  P b = 1979 →
  P c = 1979 →
  P d = 1979 →
  ∀ (x : ℤ), P x ≠ 3958 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_value_constraint_l3468_346827


namespace NUMINAMATH_CALUDE_sock_selection_l3468_346872

theorem sock_selection (n m k : ℕ) : 
  n = 8 → m = 4 → k = 1 →
  (Nat.choose n m) - (Nat.choose (n - k) m) = 35 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_l3468_346872


namespace NUMINAMATH_CALUDE_function_composition_l3468_346822

theorem function_composition (f : ℝ → ℝ) (h : ∀ x > 0, f (3 * x) = 3 / (3 + x)) :
  ∀ x > 0, 2 * f x = 18 / (9 + x) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l3468_346822


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l3468_346871

-- Define a function f with the given property
def f : ℝ → ℝ := sorry

-- State the property of f
axiom f_property : ∀ x : ℝ, f x = f (3 - x)

-- Define what it means for a line to be an axis of symmetry
def is_axis_of_symmetry (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

-- Theorem statement
theorem axis_of_symmetry :
  is_axis_of_symmetry (3/2) f :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l3468_346871


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3468_346803

/-- Proposition p -/
def p (x : ℝ) : Prop := x^2 - x - 20 > 0

/-- Proposition q -/
def q (x : ℝ) : Prop := |x| - 2 > 0

theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3468_346803


namespace NUMINAMATH_CALUDE_least_number_of_pennies_l3468_346857

theorem least_number_of_pennies : ∃ (a : ℕ), a > 0 ∧ 
  a % 7 = 3 ∧ 
  a % 5 = 4 ∧ 
  a % 3 = 2 ∧ 
  ∀ (b : ℕ), b > 0 → b % 7 = 3 → b % 5 = 4 → b % 3 = 2 → a ≤ b :=
by sorry

end NUMINAMATH_CALUDE_least_number_of_pennies_l3468_346857


namespace NUMINAMATH_CALUDE_circle_tangent_to_y_axis_l3468_346830

/-- A circle in the Euclidean plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- A point is on the y-axis if its x-coordinate is 0 -/
def on_y_axis (p : ℝ × ℝ) : Prop := p.1 = 0

/-- A circle is tangent to the y-axis if there exists exactly one point that is both on the circle and on the y-axis -/
def tangent_to_y_axis (c : Circle) : Prop :=
  ∃! p : ℝ × ℝ, c.equation p.1 p.2 ∧ on_y_axis p

/-- The main theorem -/
theorem circle_tangent_to_y_axis :
  let c := Circle.mk (-2, 3) 2
  c.equation x y ↔ (x + 2)^2 + (y - 3)^2 = 4 ∧ tangent_to_y_axis c :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_y_axis_l3468_346830


namespace NUMINAMATH_CALUDE_quadratic_symmetry_axis_l3468_346817

/-- A quadratic function passing through points (-4,m) and (2,m) has its axis of symmetry at x = -1 -/
theorem quadratic_symmetry_axis (f : ℝ → ℝ) (m : ℝ) : 
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →  -- f is a quadratic function
  f (-4) = m →                                    -- f passes through (-4,m)
  f 2 = m →                                       -- f passes through (2,m)
  (∀ x, f (x - 1) = f (-x - 1)) :=                -- axis of symmetry is x = -1
by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_axis_l3468_346817


namespace NUMINAMATH_CALUDE_base7_246_equals_132_l3468_346847

/-- Converts a base 7 number to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

theorem base7_246_equals_132 :
  base7ToBase10 [6, 4, 2] = 132 := by
  sorry

end NUMINAMATH_CALUDE_base7_246_equals_132_l3468_346847


namespace NUMINAMATH_CALUDE_alex_score_l3468_346899

/-- Represents the number of shots attempted for each type --/
structure ShotAttempts where
  free_throws : ℕ
  three_points : ℕ
  two_points : ℕ

/-- Calculates the total points scored given the shot attempts --/
def calculate_score (attempts : ShotAttempts) : ℕ :=
  (attempts.free_throws * 8 / 10) +
  (attempts.three_points * 3 * 1 / 10) +
  (attempts.two_points * 2 * 5 / 10)

theorem alex_score :
  ∃ (attempts : ShotAttempts),
    attempts.free_throws + attempts.three_points + attempts.two_points = 40 ∧
    calculate_score attempts = 28 := by
  sorry

end NUMINAMATH_CALUDE_alex_score_l3468_346899


namespace NUMINAMATH_CALUDE_total_tips_calculation_l3468_346893

def lawn_price : ℕ := 33
def lawns_mowed : ℕ := 16
def total_earned : ℕ := 558

theorem total_tips_calculation : 
  total_earned - (lawn_price * lawns_mowed) = 30 := by sorry

end NUMINAMATH_CALUDE_total_tips_calculation_l3468_346893


namespace NUMINAMATH_CALUDE_unique_intersection_point_l3468_346844

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point satisfies the equation 3x + 2y - 9 = 0 -/
def satisfiesLine1 (p : Point) : Prop :=
  3 * p.x + 2 * p.y - 9 = 0

/-- Checks if a point satisfies the equation 5x - 2y - 10 = 0 -/
def satisfiesLine2 (p : Point) : Prop :=
  5 * p.x - 2 * p.y - 10 = 0

/-- Checks if a point satisfies the equation x = 3 -/
def satisfiesLine3 (p : Point) : Prop :=
  p.x = 3

/-- Checks if a point satisfies the equation y = 1 -/
def satisfiesLine4 (p : Point) : Prop :=
  p.y = 1

/-- Checks if a point satisfies the equation x + y = 4 -/
def satisfiesLine5 (p : Point) : Prop :=
  p.x + p.y = 4

/-- Checks if a point satisfies all five line equations -/
def satisfiesAllLines (p : Point) : Prop :=
  satisfiesLine1 p ∧ satisfiesLine2 p ∧ satisfiesLine3 p ∧ satisfiesLine4 p ∧ satisfiesLine5 p

/-- Theorem stating that there is exactly one point satisfying all five line equations -/
theorem unique_intersection_point : ∃! p : Point, satisfiesAllLines p := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l3468_346844


namespace NUMINAMATH_CALUDE_mistaken_division_l3468_346836

theorem mistaken_division (n : ℕ) (h : n = 172) :
  ∃! x : ℕ, x > 0 ∧ n % x = 7 ∧ n / x = n / 4 - 28 := by
  sorry

end NUMINAMATH_CALUDE_mistaken_division_l3468_346836


namespace NUMINAMATH_CALUDE_nancy_bottle_caps_l3468_346888

/-- The number of bottle caps Nancy starts with -/
def initial_caps : ℕ := 91

/-- The number of bottle caps Nancy finds -/
def found_caps : ℕ := 88

/-- The total number of bottle caps Nancy ends with -/
def total_caps : ℕ := initial_caps + found_caps

theorem nancy_bottle_caps : total_caps = 179 := by sorry

end NUMINAMATH_CALUDE_nancy_bottle_caps_l3468_346888
