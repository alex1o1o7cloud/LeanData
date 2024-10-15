import Mathlib

namespace NUMINAMATH_CALUDE_fuwa_selection_theorem_l2551_255167

/-- The number of types of "Chinese Fuwa" mascots -/
def num_types : ℕ := 5

/-- The total number of Fuwa mascots -/
def total_fuwa : ℕ := 10

/-- The number of Fuwa to be selected -/
def select_num : ℕ := 5

/-- The number of ways to select Fuwa mascots -/
def ways_to_select : ℕ := 160

/-- Theorem stating the number of ways to select Fuwa mascots -/
theorem fuwa_selection_theorem :
  (num_types = 5) →
  (total_fuwa = 10) →
  (select_num = 5) →
  (ways_to_select = 
    2 * (Nat.choose num_types 1) * (2^(num_types - 1))) :=
by sorry

end NUMINAMATH_CALUDE_fuwa_selection_theorem_l2551_255167


namespace NUMINAMATH_CALUDE_min_passengers_for_no_loss_l2551_255185

/-- Represents the monthly expenditure of the bus in yuan -/
def monthly_expenditure : ℕ := 6000

/-- Represents the fare per person in yuan -/
def fare_per_person : ℕ := 2

/-- Represents the relationship between the number of passengers (x) and the difference between income and expenditure (y) -/
def income_expenditure_difference (x : ℕ) : ℤ :=
  (fare_per_person * x : ℤ) - monthly_expenditure

/-- Represents the condition for the bus to operate without a loss -/
def no_loss (x : ℕ) : Prop :=
  income_expenditure_difference x ≥ 0

/-- States that the minimum number of passengers required for the bus to operate without a loss is 3000 -/
theorem min_passengers_for_no_loss :
  ∀ x : ℕ, no_loss x ↔ x ≥ 3000 :=
by sorry

end NUMINAMATH_CALUDE_min_passengers_for_no_loss_l2551_255185


namespace NUMINAMATH_CALUDE_theresas_work_hours_l2551_255140

theorem theresas_work_hours : 
  let weekly_hours : List ℕ := [10, 13, 9, 14, 8, 0]
  let total_weeks : ℕ := 7
  let required_average : ℕ := 12
  let final_week_hours : ℕ := 30
  (List.sum weekly_hours + final_week_hours) / total_weeks = required_average :=
by
  sorry

end NUMINAMATH_CALUDE_theresas_work_hours_l2551_255140


namespace NUMINAMATH_CALUDE_sam_and_david_licks_l2551_255130

/-- The number of licks it takes for Dan to reach the center of a lollipop -/
def dan_licks : ℕ := 58

/-- The number of licks it takes for Michael to reach the center of a lollipop -/
def michael_licks : ℕ := 63

/-- The number of licks it takes for Lance to reach the center of a lollipop -/
def lance_licks : ℕ := 39

/-- The total number of people -/
def total_people : ℕ := 5

/-- The average number of licks for all people -/
def average_licks : ℕ := 60

/-- The theorem stating that Sam and David together take 140 licks to reach the center of a lollipop -/
theorem sam_and_david_licks : 
  total_people * average_licks - (dan_licks + michael_licks + lance_licks) = 140 := by
  sorry

end NUMINAMATH_CALUDE_sam_and_david_licks_l2551_255130


namespace NUMINAMATH_CALUDE_inequality_range_l2551_255103

theorem inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x - 1 < 0) ↔ k ∈ Set.Ioc (-4) 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l2551_255103


namespace NUMINAMATH_CALUDE_percentage_increase_proof_l2551_255199

def original_earnings : ℝ := 30
def new_earnings : ℝ := 40

theorem percentage_increase_proof :
  (new_earnings - original_earnings) / original_earnings * 100 =
  (40 - 30) / 30 * 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_increase_proof_l2551_255199


namespace NUMINAMATH_CALUDE_remainder_theorem_l2551_255168

theorem remainder_theorem (P E M S F N T : ℕ) 
  (h1 : P = E * M + S) 
  (h2 : M = N * F + T) : 
  P % (E * F + 1) = E * T + S - N :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2551_255168


namespace NUMINAMATH_CALUDE_unique_root_quadratic_l2551_255127

theorem unique_root_quadratic (p : ℝ) : 
  (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ 
   (∀ x : ℝ, x^2 - 5*p*x + 2*p^3 = 0 ↔ (x = a ∨ x = b)) ∧
   (∃! x : ℝ, x^2 - a*x + b = 0)) →
  p = 3 := by
sorry


end NUMINAMATH_CALUDE_unique_root_quadratic_l2551_255127


namespace NUMINAMATH_CALUDE_five_people_arrangement_l2551_255156

/-- The number of ways to arrange n people in a line -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n-1 people in a line -/
def arrangements_without_youngest (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of valid arrangements for n people where the youngest cannot be first or last -/
def validArrangements (n : ℕ) : ℕ :=
  totalArrangements n - 2 * arrangements_without_youngest n

theorem five_people_arrangement :
  validArrangements 5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_five_people_arrangement_l2551_255156


namespace NUMINAMATH_CALUDE_certain_number_problem_l2551_255102

theorem certain_number_problem (x : ℚ) :
  (2 / 5 : ℚ) * 300 - (3 / 5 : ℚ) * x = 45 → x = 125 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l2551_255102


namespace NUMINAMATH_CALUDE_depression_comparison_l2551_255163

-- Define the prevalence of depression for women and men
def depression_prevalence_women : ℝ := 2
def depression_prevalence_men : ℝ := 1

-- Define the correct comparative phrase
def correct_phrase : String := "twice as...as"

-- Theorem to prove
theorem depression_comparison (w m : ℝ) (phrase : String) :
  w = 2 * m → phrase = correct_phrase → 
  (w = depression_prevalence_women ∧ m = depression_prevalence_men) :=
by sorry

end NUMINAMATH_CALUDE_depression_comparison_l2551_255163


namespace NUMINAMATH_CALUDE_intersection_when_a_is_two_subset_condition_l2551_255134

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

-- Theorem 1: When a = 2, A ∩ B = (4, 5)
theorem intersection_when_a_is_two :
  A 2 ∩ B 2 = Set.Ioo 4 5 := by sorry

-- Theorem 2: B ⊆ A if and only if a ∈ [1, 3] ∪ {-1}
theorem subset_condition (a : ℝ) :
  B a ⊆ A a ↔ a ∈ Set.Icc 1 3 ∪ {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_two_subset_condition_l2551_255134


namespace NUMINAMATH_CALUDE_intersecting_squares_area_difference_l2551_255136

/-- Given four intersecting squares with side lengths 12, 9, 7, and 3 (from left to right),
    the sum of the areas of the black regions minus the sum of the areas of the gray regions equals 103. -/
theorem intersecting_squares_area_difference : 
  let a := 12 -- side length of the largest square
  let b := 9  -- side length of the second largest square
  let c := 7  -- side length of the third largest square
  let d := 3  -- side length of the smallest square
  (a^2 + c^2) - (b^2 + d^2) = 103 := by sorry

end NUMINAMATH_CALUDE_intersecting_squares_area_difference_l2551_255136


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l2551_255161

theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) :
  b₁ = 2 →
  (∃ r : ℝ, b₂ = b₁ * r ∧ b₃ = b₂ * r) →
  ∀ b₂' b₃' : ℝ, (∃ r' : ℝ, b₂' = 2 * r' ∧ b₃' = b₂' * r') →
  3 * b₂ + 4 * b₃ ≥ 3 * b₂' + 4 * b₃' →
  3 * b₂ + 4 * b₃ ≥ -9/8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l2551_255161


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2551_255148

theorem expand_and_simplify (x : ℝ) : (x + 6) * (x - 11) = x^2 - 5*x - 66 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2551_255148


namespace NUMINAMATH_CALUDE_square_on_hypotenuse_l2551_255184

theorem square_on_hypotenuse (a b : ℝ) (ha : a = 9) (hb : b = 12) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a * b * c) / (a^2 + b^2)
  s = 45 / 8 := by sorry

end NUMINAMATH_CALUDE_square_on_hypotenuse_l2551_255184


namespace NUMINAMATH_CALUDE_range_of_a_l2551_255198

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + 2*x + a ≥ 0) ↔ a ∈ Set.Ici (-8) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2551_255198


namespace NUMINAMATH_CALUDE_sum_of_integers_l2551_255195

theorem sum_of_integers (a b : ℕ+) (h1 : a - b = 8) (h2 : a * b = 65) : a + b = 18 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2551_255195


namespace NUMINAMATH_CALUDE_max_friendly_groups_19_20_l2551_255146

/-- A friendly group in a tournament --/
structure FriendlyGroup (α : Type*) :=
  (a b c : α)
  (a_beats_b : a ≠ b)
  (b_beats_c : b ≠ c)
  (c_beats_a : c ≠ a)

/-- Round-robin tournament results --/
def RoundRobinTournament (α : Type*) := α → α → Prop

/-- Maximum number of friendly groups in a tournament --/
def MaxFriendlyGroups (n : ℕ) : ℕ :=
  if n % 2 = 1 then
    n * (n - 1) * (n + 1) / 24
  else
    n * (n - 2) * (n + 2) / 24

/-- Theorem about maximum friendly groups in tournaments with 19 and 20 teams --/
theorem max_friendly_groups_19_20 :
  (MaxFriendlyGroups 19 = 285) ∧ (MaxFriendlyGroups 20 = 330) :=
by sorry

end NUMINAMATH_CALUDE_max_friendly_groups_19_20_l2551_255146


namespace NUMINAMATH_CALUDE_product_remainder_l2551_255188

theorem product_remainder (a b c : ℕ) (h : a = 1125 ∧ b = 1127 ∧ c = 1129) : 
  (a * b * c) % 12 = 3 := by
sorry

end NUMINAMATH_CALUDE_product_remainder_l2551_255188


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l2551_255186

theorem probability_at_least_one_woman (total_people : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) :
  total_people = men + women →
  men = 10 →
  women = 5 →
  selected = 4 →
  1 - (Nat.choose men selected / Nat.choose total_people selected) = 11 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l2551_255186


namespace NUMINAMATH_CALUDE_net_marble_change_l2551_255147

def marble_transactions (initial : Int) (lost : Int) (found : Int) (traded_out : Int) (traded_in : Int) (gave_away : Int) (received : Int) : Int :=
  initial - lost + found - traded_out + traded_in - gave_away + received

theorem net_marble_change : 
  marble_transactions 20 16 8 5 9 3 4 = -3 := by
  sorry

end NUMINAMATH_CALUDE_net_marble_change_l2551_255147


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_tangents_l2551_255197

/-- The circle equation: x^2 - 2x + y^2 - 2y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 - 2*x + y^2 - 2*y + 1 = 0

/-- The external point P -/
def P : ℝ × ℝ := (3, 2)

/-- The cosine of the angle between two tangent lines -/
noncomputable def cos_angle_between_tangents : ℝ := 3/5

theorem cosine_of_angle_between_tangents :
  let (px, py) := P
  ∀ x y : ℝ, circle_equation x y →
  cos_angle_between_tangents = 3/5 := by sorry

end NUMINAMATH_CALUDE_cosine_of_angle_between_tangents_l2551_255197


namespace NUMINAMATH_CALUDE_quadratic_equivalent_forms_l2551_255183

theorem quadratic_equivalent_forms : ∀ x : ℝ, x^2 - 2*x - 1 = (x - 1)^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equivalent_forms_l2551_255183


namespace NUMINAMATH_CALUDE_charlottes_phone_usage_l2551_255118

/-- Charlotte's daily phone usage problem -/
theorem charlottes_phone_usage 
  (social_media_time : ℝ) 
  (weekly_social_media : ℝ) 
  (h1 : social_media_time = weekly_social_media / 7)
  (h2 : weekly_social_media = 56)
  (h3 : social_media_time = daily_phone_time / 2) : 
  daily_phone_time = 16 :=
sorry

end NUMINAMATH_CALUDE_charlottes_phone_usage_l2551_255118


namespace NUMINAMATH_CALUDE_train_length_l2551_255155

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 3 → ∃ (length : ℝ), abs (length - 50.01) < 0.01 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2551_255155


namespace NUMINAMATH_CALUDE_hexagon_diagonal_intersection_probability_l2551_255172

/-- A regular hexagon -/
structure RegularHexagon where
  -- Add any necessary properties here

/-- A diagonal of a regular hexagon -/
structure Diagonal (h : RegularHexagon) where
  -- Add any necessary properties here

/-- Predicate to check if two diagonals intersect inside the hexagon -/
def intersect_inside (h : RegularHexagon) (d1 d2 : Diagonal h) : Prop :=
  sorry

/-- The set of all diagonals in a regular hexagon -/
def all_diagonals (h : RegularHexagon) : Set (Diagonal h) :=
  sorry

/-- The probability that two randomly chosen diagonals intersect inside the hexagon -/
def intersection_probability (h : RegularHexagon) : ℚ :=
  sorry

theorem hexagon_diagonal_intersection_probability (h : RegularHexagon) :
  intersection_probability h = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonal_intersection_probability_l2551_255172


namespace NUMINAMATH_CALUDE_total_tickets_bought_l2551_255165

theorem total_tickets_bought (adult_price children_price total_spent adult_count : ℚ)
  (h1 : adult_price = 5.5)
  (h2 : children_price = 3.5)
  (h3 : total_spent = 83.5)
  (h4 : adult_count = 5) :
  ∃ (children_count : ℚ), adult_count + children_count = 21 ∧
    adult_price * adult_count + children_price * children_count = total_spent :=
by sorry

end NUMINAMATH_CALUDE_total_tickets_bought_l2551_255165


namespace NUMINAMATH_CALUDE_bankers_discount_l2551_255157

/-- Banker's discount calculation --/
theorem bankers_discount (bankers_gain : ℚ) (interest_rate : ℚ) (time : ℕ) : 
  bankers_gain = 270 → interest_rate = 12 / 100 → time = 3 → 
  let present_value := (bankers_gain * 100) / (interest_rate * time)
  let face_value := present_value + bankers_gain
  let bankers_discount := (face_value * interest_rate * time)
  bankers_discount = 36720 / 100 := by
  sorry

end NUMINAMATH_CALUDE_bankers_discount_l2551_255157


namespace NUMINAMATH_CALUDE_distance_between_cities_l2551_255117

/-- The distance between two cities A and B, where two trains traveling towards each other meet. -/
theorem distance_between_cities (v1 v2 t1 t2 : ℝ) (h1 : v1 = 60) (h2 : v2 = 75) (h3 : t1 = 4) (h4 : t2 = 3) : v1 * t1 + v2 * t2 = 465 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l2551_255117


namespace NUMINAMATH_CALUDE_hiker_distance_l2551_255164

/-- Calculates the final straight-line distance of a hiker from their starting point
    given their movements in cardinal directions. -/
theorem hiker_distance (north south west east : ℝ) :
  north = 20 ∧ south = 8 ∧ west = 15 ∧ east = 10 →
  Real.sqrt ((north - south)^2 + (west - east)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_hiker_distance_l2551_255164


namespace NUMINAMATH_CALUDE_monthly_average_production_l2551_255152

/-- Calculates the daily average production for a month given the production data for two periods. -/
theorem monthly_average_production 
  (days_first_period : ℕ) 
  (days_second_period : ℕ) 
  (avg_first_period : ℚ) 
  (avg_second_period : ℚ) : 
  days_first_period = 25 →
  days_second_period = 5 →
  avg_first_period = 70 →
  avg_second_period = 58 →
  (days_first_period * avg_first_period + days_second_period * avg_second_period) / (days_first_period + days_second_period) = 68 := by
  sorry

#check monthly_average_production

end NUMINAMATH_CALUDE_monthly_average_production_l2551_255152


namespace NUMINAMATH_CALUDE_three_quantities_change_l2551_255109

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

-- Define the line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the triangle type
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the problem setup
def setup (A B P Q : Point) (l1 l2 lPQ : Line) : Prop :=
  (P.x = (A.x + B.x) / 2) ∧ 
  (P.y = (A.y + B.y) / 2) ∧
  (l1.a = lPQ.a ∧ l1.b = lPQ.b) ∧
  (l2.a = lPQ.a ∧ l2.b = lPQ.b)

-- Define the four quantities
def lengthAB (A B : Point) : ℝ := sorry
def perimeterAPB (A B P : Point) : ℝ := sorry
def areaAPB (A B P : Point) : ℝ := sorry
def distancePtoAB (A B P : Point) : ℝ := sorry

-- Define a function that counts how many quantities change
def countChangingQuantities (A B P Q : Point) (l1 l2 lPQ : Line) : ℕ := sorry

-- The main theorem
theorem three_quantities_change 
  (A B P Q : Point) (l1 l2 lPQ : Line) 
  (h : setup A B P Q l1 l2 lPQ) : 
  countChangingQuantities A B P Q l1 l2 lPQ = 3 := sorry

end NUMINAMATH_CALUDE_three_quantities_change_l2551_255109


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l2551_255133

theorem sphere_surface_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 / 3 * Real.pi * r₁^3) / (4 / 3 * Real.pi * r₂^3) = 8 / 27 →
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 4 / 9 :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l2551_255133


namespace NUMINAMATH_CALUDE_product_negative_implies_one_less_than_one_l2551_255128

theorem product_negative_implies_one_less_than_one (a b c : ℝ) :
  (a - 1) * (b - 1) * (c - 1) < 0 →
  (a < 1) ∨ (b < 1) ∨ (c < 1) :=
by
  sorry

end NUMINAMATH_CALUDE_product_negative_implies_one_less_than_one_l2551_255128


namespace NUMINAMATH_CALUDE_both_knaves_lied_yesterday_on_friday_l2551_255129

-- Define the days of the week
inductive Day : Type
  | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define the knaves
inductive Knave : Type
  | Hearts | Diamonds

-- Define the truth-telling function for each knave
def tells_truth (k : Knave) (d : Day) : Prop :=
  match k with
  | Knave.Hearts => d = Day.Monday ∨ d = Day.Tuesday ∨ d = Day.Wednesday ∨ d = Day.Thursday
  | Knave.Diamonds => d = Day.Friday ∨ d = Day.Saturday ∨ d = Day.Sunday ∨ d = Day.Monday

-- Define the function to check if a knave lied yesterday
def lied_yesterday (k : Knave) (d : Day) : Prop :=
  ¬(tells_truth k (match d with
    | Day.Monday => Day.Sunday
    | Day.Tuesday => Day.Monday
    | Day.Wednesday => Day.Tuesday
    | Day.Thursday => Day.Wednesday
    | Day.Friday => Day.Thursday
    | Day.Saturday => Day.Friday
    | Day.Sunday => Day.Saturday))

-- Theorem: The only day when both knaves can truthfully say "Yesterday I told lies" is Friday
theorem both_knaves_lied_yesterday_on_friday :
  ∀ d : Day, (tells_truth Knave.Hearts d ∧ lied_yesterday Knave.Hearts d ∧
              tells_truth Knave.Diamonds d ∧ lied_yesterday Knave.Diamonds d) 
             ↔ d = Day.Friday :=
sorry

end NUMINAMATH_CALUDE_both_knaves_lied_yesterday_on_friday_l2551_255129


namespace NUMINAMATH_CALUDE_solve_for_n_l2551_255194

/-- Given an equation x + 1315 + n - 1569 = 11901 where x = 88320,
    prove that the value of n is -75165. -/
theorem solve_for_n (x n : ℤ) (h1 : x + 1315 + n - 1569 = 11901) (h2 : x = 88320) :
  n = -75165 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_n_l2551_255194


namespace NUMINAMATH_CALUDE_toy_problem_solution_l2551_255154

/-- Represents the toy purchasing and pricing problem -/
structure ToyProblem where
  total_toys : ℕ
  total_cost : ℕ
  purchase_price_A : ℕ
  purchase_price_B : ℕ
  original_price_A : ℕ
  original_daily_sales : ℕ
  sales_increase_rate : ℕ
  desired_daily_profit : ℕ

/-- The solution to the toy problem -/
structure ToySolution where
  num_A : ℕ
  num_B : ℕ
  new_price_A : ℕ

/-- Theorem stating the correct solution for the given problem -/
theorem toy_problem_solution (p : ToyProblem) 
  (h1 : p.total_toys = 50)
  (h2 : p.total_cost = 1320)
  (h3 : p.purchase_price_A = 28)
  (h4 : p.purchase_price_B = 24)
  (h5 : p.original_price_A = 40)
  (h6 : p.original_daily_sales = 8)
  (h7 : p.sales_increase_rate = 1)
  (h8 : p.desired_daily_profit = 96) :
  ∃ (s : ToySolution), 
    s.num_A = 30 ∧ 
    s.num_B = 20 ∧ 
    s.new_price_A = 36 ∧
    s.num_A + s.num_B = p.total_toys ∧
    s.num_A * p.purchase_price_A + s.num_B * p.purchase_price_B = p.total_cost ∧
    (s.new_price_A - p.purchase_price_A) * (p.original_daily_sales + (p.original_price_A - s.new_price_A) * p.sales_increase_rate) = p.desired_daily_profit :=
by
  sorry


end NUMINAMATH_CALUDE_toy_problem_solution_l2551_255154


namespace NUMINAMATH_CALUDE_min_journey_cost_l2551_255193

-- Define the cities and distances
def XY : ℝ := 3500
def XZ : ℝ := 4000

-- Define the cost functions
def train_cost (distance : ℝ) : ℝ := 0.20 * distance
def taxi_cost (distance : ℝ) : ℝ := 150 + 0.15 * distance

-- Define the theorem
theorem min_journey_cost :
  let YZ : ℝ := Real.sqrt (XZ^2 - XY^2)
  let XY_cost : ℝ := min (train_cost XY) (taxi_cost XY)
  let YZ_cost : ℝ := min (train_cost YZ) (taxi_cost YZ)
  let ZX_cost : ℝ := min (train_cost XZ) (taxi_cost XZ)
  XY_cost + YZ_cost + ZX_cost = 1812.30 := by sorry

end NUMINAMATH_CALUDE_min_journey_cost_l2551_255193


namespace NUMINAMATH_CALUDE_tom_roses_count_l2551_255174

/-- The number of roses in a dozen -/
def roses_per_dozen : ℕ := 12

/-- The number of dozens Tom sends per day -/
def dozens_per_day : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of roses Tom sent in a week -/
def total_roses : ℕ := days_in_week * dozens_per_day * roses_per_dozen

theorem tom_roses_count : total_roses = 168 := by
  sorry

end NUMINAMATH_CALUDE_tom_roses_count_l2551_255174


namespace NUMINAMATH_CALUDE_gcd_180_270_l2551_255123

theorem gcd_180_270 : Nat.gcd 180 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_270_l2551_255123


namespace NUMINAMATH_CALUDE_integer_root_of_polynomial_l2551_255179

theorem integer_root_of_polynomial (b c : ℚ) : 
  (∃ (x : ℝ), x^3 + b*x + c = 0 ∧ x = 3 - Real.sqrt 5) → 
  (-6 : ℝ)^3 + b*(-6) + c = 0 := by
sorry

end NUMINAMATH_CALUDE_integer_root_of_polynomial_l2551_255179


namespace NUMINAMATH_CALUDE_complex_on_negative_y_axis_l2551_255181

theorem complex_on_negative_y_axis (a : ℝ) : 
  (∃ y : ℝ, y < 0 ∧ (a + Complex.I)^2 = Complex.I * y) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_negative_y_axis_l2551_255181


namespace NUMINAMATH_CALUDE_cost_of_item_d_l2551_255151

/-- Represents the prices and taxes for items in a shopping scenario -/
structure ShoppingScenario where
  total_spent : ℝ
  total_abc : ℝ
  total_tax : ℝ
  tax_rate_a : ℝ
  tax_rate_b : ℝ
  tax_rate_c : ℝ
  discount_a : ℝ
  discount_b : ℝ

/-- Theorem stating that the cost of item D is 25 given the shopping scenario -/
theorem cost_of_item_d (s : ShoppingScenario)
  (h1 : s.total_spent = 250)
  (h2 : s.total_abc = 225)
  (h3 : s.total_tax = 30)
  (h4 : s.tax_rate_a = 0.05)
  (h5 : s.tax_rate_b = 0.12)
  (h6 : s.tax_rate_c = 0.18)
  (h7 : s.discount_a = 0.1)
  (h8 : s.discount_b = 0.05) :
  s.total_spent - s.total_abc = 25 := by
  sorry

#check cost_of_item_d

end NUMINAMATH_CALUDE_cost_of_item_d_l2551_255151


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2551_255171

/-- Two vectors are parallel if and only if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_k_value :
  ∀ k : ℝ,
  let a : ℝ × ℝ := (2 * k + 2, 4)
  let b : ℝ × ℝ := (k + 1, 8)
  are_parallel a b → k = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2551_255171


namespace NUMINAMATH_CALUDE_marathon_speed_fraction_l2551_255144

theorem marathon_speed_fraction (t₃ t₆ : ℝ) (h₁ : t₃ > 0) (h₂ : t₆ > 0) : 
  (3 * t₃ + 6 * t₆) / (t₃ + t₆) = 5 → t₃ / (t₃ + t₆) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_marathon_speed_fraction_l2551_255144


namespace NUMINAMATH_CALUDE_triangle_abc_acute_angled_l2551_255192

theorem triangle_abc_acute_angled (A B C : ℝ) 
  (h1 : A + B + C = 180) 
  (h2 : A = B) 
  (h3 : A = 2 * C) : 
  A < 90 ∧ B < 90 ∧ C < 90 := by
sorry


end NUMINAMATH_CALUDE_triangle_abc_acute_angled_l2551_255192


namespace NUMINAMATH_CALUDE_equation_solutions_l2551_255196

theorem equation_solutions :
  (∃ x : ℚ, 4 - 3 * x = 6 - 5 * x ∧ x = 1) ∧
  (∃ x : ℚ, 7 - 3 * (x - 1) = -x ∧ x = 5) ∧
  (∃ x : ℚ, (3 * x - 1) / 2 = 1 - (x - 1) / 6 ∧ x = 1) ∧
  (∃ x : ℚ, (2 * x - 1) / 3 - x = (2 * x + 1) / 4 - 1 ∧ x = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2551_255196


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2551_255170

-- Define the circles C and D
variables (C D : ℝ → Prop)

-- Define the radii of circles C and D
variables (r_C r_D : ℝ)

-- Define the common arc length
variable (L : ℝ)

-- State the theorem
theorem circle_area_ratio 
  (h1 : L = (60 / 360) * (2 * Real.pi * r_C)) 
  (h2 : L = (45 / 360) * (2 * Real.pi * r_D)) 
  (h3 : 2 * Real.pi * r_D = 2 * (2 * Real.pi * r_C)) :
  (Real.pi * r_D^2) / (Real.pi * r_C^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2551_255170


namespace NUMINAMATH_CALUDE_car_trip_duration_l2551_255153

/-- Proves that a car trip with given conditions has a total duration of 6 hours -/
theorem car_trip_duration (initial_speed initial_time additional_speed average_speed : ℝ) :
  initial_speed = 75 →
  initial_time = 4 →
  additional_speed = 60 →
  average_speed = 70 →
  ∃ (total_time : ℝ),
    total_time > initial_time ∧
    (initial_speed * initial_time + additional_speed * (total_time - initial_time)) /
      total_time = average_speed ∧
    total_time = 6 :=
by sorry

end NUMINAMATH_CALUDE_car_trip_duration_l2551_255153


namespace NUMINAMATH_CALUDE_h_equals_neg_f_of_six_minus_x_l2551_255120

-- Define a function that reflects a graph across the y-axis
def reflectY (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (-x)

-- Define a function that reflects a graph across the x-axis
def reflectX (f : ℝ → ℝ) : ℝ → ℝ := λ x => -f x

-- Define a function that shifts a graph to the right by a given amount
def shiftRight (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f (x - shift)

-- Define the composition of these transformations
def h (f : ℝ → ℝ) : ℝ → ℝ := shiftRight (reflectX (reflectY f)) 6

-- State the theorem
theorem h_equals_neg_f_of_six_minus_x (f : ℝ → ℝ) : 
  ∀ x : ℝ, h f x = -f (6 - x) := by sorry

end NUMINAMATH_CALUDE_h_equals_neg_f_of_six_minus_x_l2551_255120


namespace NUMINAMATH_CALUDE_probability_of_two_boys_l2551_255119

theorem probability_of_two_boys (total : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total = 12) 
  (h2 : boys = 8) 
  (h3 : girls = 4) 
  (h4 : total = boys + girls) : 
  (Nat.choose boys 2 : ℚ) / (Nat.choose total 2) = 14/33 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_two_boys_l2551_255119


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2551_255108

/-- The atomic weight of Nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Nitrogen atoms in the compound -/
def num_N : ℕ := 2

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := num_N * atomic_weight_N + num_O * atomic_weight_O

theorem compound_molecular_weight : molecular_weight = 76.02 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2551_255108


namespace NUMINAMATH_CALUDE_other_cat_weight_l2551_255173

/-- Represents the weights of animals in a household -/
structure HouseholdWeights where
  cat1 : ℝ
  cat2 : ℝ
  dog : ℝ

/-- Theorem stating the weight of the second cat given the conditions -/
theorem other_cat_weight (h : HouseholdWeights) 
    (h1 : h.cat1 = 10)
    (h2 : h.dog = 34)
    (h3 : h.dog = 2 * (h.cat1 + h.cat2)) : 
  h.cat2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_other_cat_weight_l2551_255173


namespace NUMINAMATH_CALUDE_negation_of_p_l2551_255150

theorem negation_of_p : ∀ x : ℝ, -2 < x ∧ x < 2 → |x - 1| + |x + 2| < 6 := by sorry

end NUMINAMATH_CALUDE_negation_of_p_l2551_255150


namespace NUMINAMATH_CALUDE_video_archive_space_theorem_l2551_255166

/-- Represents the number of days in the video archive -/
def days : ℕ := 15

/-- Represents the total disk space used by the archive in megabytes -/
def total_space : ℕ := 30000

/-- Calculates the total number of minutes in a given number of days -/
def total_minutes (d : ℕ) : ℕ := d * 24 * 60

/-- Calculates the average disk space per minute of video -/
def avg_space_per_minute : ℚ :=
  total_space / total_minutes days

theorem video_archive_space_theorem :
  abs (avg_space_per_minute - 1.388) < 0.001 :=
sorry

end NUMINAMATH_CALUDE_video_archive_space_theorem_l2551_255166


namespace NUMINAMATH_CALUDE_thomas_score_l2551_255191

def class_size : ℕ := 20
def initial_average : ℚ := 78
def final_average : ℚ := 79

theorem thomas_score :
  ∃ (score : ℚ),
    (class_size - 1) * initial_average + score = class_size * final_average ∧
    score = 98 := by
  sorry

end NUMINAMATH_CALUDE_thomas_score_l2551_255191


namespace NUMINAMATH_CALUDE_simon_change_theorem_l2551_255106

/-- Calculates the discounted price for a flower purchase -/
def discountedPrice (quantity : ℕ) (price : ℚ) (discount : ℚ) : ℚ :=
  (quantity : ℚ) * price * (1 - discount)

/-- Calculates the total price after tax -/
def totalPriceAfterTax (prices : List ℚ) (taxRate : ℚ) : ℚ :=
  let subtotal := prices.sum
  subtotal * (1 + taxRate)

theorem simon_change_theorem (pansyPrice petuniasPrice lilyPrice orchidPrice : ℚ)
    (pansyDiscount hydrangeaDiscount petuniaDiscount lilyDiscount orchidDiscount : ℚ)
    (hydrangeaPrice : ℚ) (taxRate : ℚ) :
    pansyPrice = 2.5 →
    petuniasPrice = 1 →
    lilyPrice = 5 →
    orchidPrice = 7.5 →
    hydrangeaPrice = 12.5 →
    pansyDiscount = 0.1 →
    hydrangeaDiscount = 0.15 →
    petuniaDiscount = 0.2 →
    lilyDiscount = 0.12 →
    orchidDiscount = 0.08 →
    taxRate = 0.06 →
    let pansies := discountedPrice 5 pansyPrice pansyDiscount
    let hydrangea := discountedPrice 1 hydrangeaPrice hydrangeaDiscount
    let petunias := discountedPrice 5 petuniasPrice petuniaDiscount
    let lilies := discountedPrice 3 lilyPrice lilyDiscount
    let orchids := discountedPrice 2 orchidPrice orchidDiscount
    let total := totalPriceAfterTax [pansies, hydrangea, petunias, lilies, orchids] taxRate
    100 - total = 43.95 := by sorry

end NUMINAMATH_CALUDE_simon_change_theorem_l2551_255106


namespace NUMINAMATH_CALUDE_triangle_sine_product_l2551_255180

theorem triangle_sine_product (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 3 →
  b + c = 3 →
  A = π / 3 →
  0 < B →
  B < π →
  0 < C →
  C < π →
  A + B + C = π →
  a = 2 * Real.sin (B / 2) * Real.sin (C / 2) →
  b = 2 * Real.sin (A / 2) * Real.sin (C / 2) →
  c = 2 * Real.sin (A / 2) * Real.sin (B / 2) →
  Real.sin B * Real.sin C = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_product_l2551_255180


namespace NUMINAMATH_CALUDE_seventh_observation_value_l2551_255175

theorem seventh_observation_value 
  (n : Nat) 
  (initial_avg : ℝ) 
  (new_avg : ℝ) 
  (h1 : n = 6) 
  (h2 : initial_avg = 11) 
  (h3 : new_avg = initial_avg - 1) : 
  (n : ℝ) * initial_avg - ((n + 1) : ℝ) * new_avg = -4 := by
  sorry

end NUMINAMATH_CALUDE_seventh_observation_value_l2551_255175


namespace NUMINAMATH_CALUDE_problem_statement_l2551_255116

/-- Given two real numbers a and b with average 110, and b and c with average 170,
    if a - c = 120, then c = -120 -/
theorem problem_statement (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110)
  (h2 : (b + c) / 2 = 170)
  (h3 : a - c = 120) :
  c = -120 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2551_255116


namespace NUMINAMATH_CALUDE_salary_calculation_l2551_255159

theorem salary_calculation (food_fraction rent_fraction clothes_fraction remainder : ℚ) 
  (h1 : food_fraction = 1/5)
  (h2 : rent_fraction = 1/10)
  (h3 : clothes_fraction = 3/5)
  (h4 : remainder = 18000) :
  let total_spent_fraction := food_fraction + rent_fraction + clothes_fraction
  let remaining_fraction := 1 - total_spent_fraction
  let salary := remainder / remaining_fraction
  salary = 180000 := by sorry

end NUMINAMATH_CALUDE_salary_calculation_l2551_255159


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2551_255126

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - 4) * (x - 6)^2 ≤ 0 ↔ -2 ≤ x ∧ x ≤ 2 ∨ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2551_255126


namespace NUMINAMATH_CALUDE_jason_bought_four_dozens_l2551_255101

/-- The number of cupcakes Jason gives to each cousin -/
def cupcakes_per_cousin : ℕ := 3

/-- The number of cousins Jason has -/
def number_of_cousins : ℕ := 16

/-- The number of cupcakes in a dozen -/
def cupcakes_per_dozen : ℕ := 12

/-- Theorem: Jason bought 4 dozens of cupcakes -/
theorem jason_bought_four_dozens :
  (cupcakes_per_cousin * number_of_cousins) / cupcakes_per_dozen = 4 := by
  sorry

end NUMINAMATH_CALUDE_jason_bought_four_dozens_l2551_255101


namespace NUMINAMATH_CALUDE_quadratic_equation_identification_l2551_255162

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation from option A -/
def eq_A (x : ℝ) : ℝ := 3 * x + 1

/-- The equation from option B -/
def eq_B (x : ℝ) : ℝ := x^2 - (2 * x - 3 * x^2)

/-- The equation from option C -/
def eq_C (x y : ℝ) : ℝ := x^2 - y + 5

/-- The equation from option D -/
def eq_D (x y : ℝ) : ℝ := x - x * y - 1 - x^2

theorem quadratic_equation_identification :
  ¬ is_quadratic_equation eq_A ∧
  is_quadratic_equation eq_B ∧
  ¬ (∃ f : ℝ → ℝ, ∀ x y, eq_C x y = f x) ∧
  ¬ (∃ f : ℝ → ℝ, ∀ x y, eq_D x y = f x) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_identification_l2551_255162


namespace NUMINAMATH_CALUDE_last_digit_product_divisible_by_six_l2551_255187

/-- The last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- The remaining digits of a natural number -/
def remainingDigits (n : ℕ) : ℕ := n / 10

/-- Theorem: For all n > 3, the product of the last digit of 2^n and the remaining digits is divisible by 6 -/
theorem last_digit_product_divisible_by_six (n : ℕ) (h : n > 3) :
  ∃ k : ℕ, (lastDigit (2^n) * remainingDigits (2^n)) = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_last_digit_product_divisible_by_six_l2551_255187


namespace NUMINAMATH_CALUDE_sandy_jessica_marble_ratio_l2551_255137

/-- Proves that Sandy has 4 times more red marbles than Jessica -/
theorem sandy_jessica_marble_ratio :
  let jessica_marbles : ℕ := 3 * 12 -- 3 dozen
  let sandy_marbles : ℕ := 144
  (sandy_marbles : ℚ) / jessica_marbles = 4 := by
  sorry

end NUMINAMATH_CALUDE_sandy_jessica_marble_ratio_l2551_255137


namespace NUMINAMATH_CALUDE_carpet_area_needed_l2551_255125

-- Define the room dimensions in feet
def room_length : ℝ := 18
def room_width : ℝ := 12

-- Define the conversion factor from feet to yards
def feet_per_yard : ℝ := 3

-- Define the area already covered in square yards
def area_covered : ℝ := 4

-- Theorem to prove
theorem carpet_area_needed : 
  let length_yards := room_length / feet_per_yard
  let width_yards := room_width / feet_per_yard
  let total_area := length_yards * width_yards
  total_area - area_covered = 20 := by sorry

end NUMINAMATH_CALUDE_carpet_area_needed_l2551_255125


namespace NUMINAMATH_CALUDE_samantha_calculation_l2551_255178

theorem samantha_calculation : 
  let incorrect_input := 125 * 320
  let correct_product := 0.125 * 3.2
  let final_result := correct_product + 2.5
  incorrect_input = 40000 ∧ final_result = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_samantha_calculation_l2551_255178


namespace NUMINAMATH_CALUDE_steps_on_sunday_l2551_255169

def target_average : ℕ := 9000
def days_in_week : ℕ := 7
def known_days : ℕ := 4
def friday_saturday_average : ℕ := 9050

def steps_known_days : List ℕ := [9100, 8300, 9200, 8900]

theorem steps_on_sunday (
  target_total : target_average * days_in_week = 63000)
  (known_total : steps_known_days.sum = 35500)
  (friday_saturday_total : friday_saturday_average * 2 = 18100)
  : 63000 - 35500 - 18100 = 9400 := by
  sorry

end NUMINAMATH_CALUDE_steps_on_sunday_l2551_255169


namespace NUMINAMATH_CALUDE_club_probability_theorem_l2551_255141

theorem club_probability_theorem (total_members : ℕ) (boys : ℕ) (girls : ℕ) :
  total_members = 15 →
  boys = 8 →
  girls = 7 →
  total_members = boys + girls →
  (Nat.choose total_members 2 - Nat.choose girls 2 : ℚ) / Nat.choose total_members 2 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_club_probability_theorem_l2551_255141


namespace NUMINAMATH_CALUDE_exchange_rate_change_l2551_255115

theorem exchange_rate_change 
  (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : 0 < y ∧ y < 1) 
  (hz : 0 < z ∧ z < 1) 
  (h_eq : (1 + x) * (1 + y) * (1 + z) = (1 - x) * (1 - y) * (1 - z)) : 
  (1 - x) * (1 - y) * (1 - z) < 1 :=
by sorry

end NUMINAMATH_CALUDE_exchange_rate_change_l2551_255115


namespace NUMINAMATH_CALUDE_apple_orchard_problem_l2551_255104

theorem apple_orchard_problem (total : ℝ) (fuji : ℝ) (gala : ℝ) : 
  (0.1 * total = total - fuji - gala) →
  (fuji + 0.1 * total = 238) →
  (fuji = 0.75 * total) →
  (gala = 42) := by
sorry

end NUMINAMATH_CALUDE_apple_orchard_problem_l2551_255104


namespace NUMINAMATH_CALUDE_first_ring_at_start_of_day_l2551_255113

-- Define the clock's properties
def ring_interval : ℕ := 3
def rings_per_day : ℕ := 8
def hours_per_day : ℕ := 24

-- Theorem to prove
theorem first_ring_at_start_of_day :
  ring_interval * rings_per_day = hours_per_day →
  ring_interval ∣ hours_per_day →
  (0 : ℕ) = hours_per_day % ring_interval :=
by
  sorry

#check first_ring_at_start_of_day

end NUMINAMATH_CALUDE_first_ring_at_start_of_day_l2551_255113


namespace NUMINAMATH_CALUDE_sixteenth_occurrence_shift_l2551_255121

/-- Represents the number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- Calculates the sum of the first n even numbers -/
def sum_even (n : ℕ) : ℕ := n * (n + 1)

/-- Calculates the shift for the nth occurrence of a letter -/
def shift (n : ℕ) : ℕ := sum_even n % alphabet_size

/-- Theorem: The 16th occurrence of a letter is shifted by 16 places -/
theorem sixteenth_occurrence_shift :
  shift 16 = 16 := by sorry

end NUMINAMATH_CALUDE_sixteenth_occurrence_shift_l2551_255121


namespace NUMINAMATH_CALUDE_cosecant_330_degrees_l2551_255122

theorem cosecant_330_degrees :
  let csc (θ : ℝ) := 1 / Real.sin θ
  let π : ℝ := Real.pi
  ∀ (θ : ℝ), Real.sin (2 * π - θ) = -Real.sin θ
  → Real.sin (π / 6) = 1 / 2
  → csc (11 * π / 6) = -2 := by
  sorry

end NUMINAMATH_CALUDE_cosecant_330_degrees_l2551_255122


namespace NUMINAMATH_CALUDE_equation_solutions_l2551_255131

theorem equation_solutions :
  (∀ x : ℝ, 4 * (x - 2)^2 = 9 ↔ x = 7/2 ∨ x = 1/2) ∧
  (∀ x : ℝ, x^2 + 6*x - 1 = 0 ↔ x = -3 + Real.sqrt 10 ∨ x = -3 - Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2551_255131


namespace NUMINAMATH_CALUDE_simplify_expression_l2551_255176

theorem simplify_expression (q : ℝ) : 
  ((6 * q - 2) - 3 * q * 5) * 2 + (5 - 2 / 4) * (8 * q - 12) = 18 * q - 58 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2551_255176


namespace NUMINAMATH_CALUDE_expression_value_at_two_l2551_255182

theorem expression_value_at_two :
  let x : ℚ := 2
  (x^2 - x - 6) / (x - 3) = 4 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_two_l2551_255182


namespace NUMINAMATH_CALUDE_tetrahedrons_from_triangular_prism_l2551_255107

/-- The number of tetrahedrons that can be formed from a regular triangular prism -/
def tetrahedrons_from_prism (n : ℕ) : ℕ :=
  Nat.choose n 4 - 3

/-- Theorem stating that the number of tetrahedrons formed from a regular triangular prism with 6 vertices is 12 -/
theorem tetrahedrons_from_triangular_prism : 
  tetrahedrons_from_prism 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedrons_from_triangular_prism_l2551_255107


namespace NUMINAMATH_CALUDE_rounding_inequality_l2551_255100

/-- The number of digits in a natural number -/
def num_digits (k : ℕ) : ℕ := sorry

/-- Rounds a natural number to the nearest power of 10 -/
def round_to_power_of_10 (k : ℕ) (power : ℕ) : ℕ := sorry

/-- Applies n-1 roundings to the nearest power of 10 -/
def apply_n_minus_1_roundings (k : ℕ) : ℕ := sorry

theorem rounding_inequality (k : ℕ) (h1 : k = 10 * 106) :
  let n := num_digits k
  let k_bar := apply_n_minus_1_roundings k
  k_bar < (18 : ℚ) / 13 * k := by sorry

end NUMINAMATH_CALUDE_rounding_inequality_l2551_255100


namespace NUMINAMATH_CALUDE_circle_and_symmetry_line_l2551_255111

-- Define the center of the circle
def C : ℝ × ℝ := (-1, 0)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 3 = 0

-- Define the symmetry line
def symmetry_line (m x y : ℝ) : Prop := m * x + y + 1 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 4

-- Theorem statement
theorem circle_and_symmetry_line :
  ∃ (r : ℝ), r > 0 ∧
  (∀ x y : ℝ, (x - C.1)^2 + (y - C.2)^2 = r^2 →
    (∃ x' y' : ℝ, tangent_line x' y' ∧ (x' - C.1)^2 + (y' - C.2)^2 = r^2)) ∧
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    ∃ m : ℝ, symmetry_line m x₁ y₁ ∧ symmetry_line m x₂ y₂ ∧ x₁ ≠ x₂) →
  (∀ x y : ℝ, circle_equation x y ↔ (x - C.1)^2 + (y - C.2)^2 = 4) ∧
  (∃! m : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_equation x₁ y₁ ∧ circle_equation x₂ y₂ ∧
    symmetry_line m x₁ y₁ ∧ symmetry_line m x₂ y₂ ∧ x₁ ≠ x₂ ∧ m = 1) :=
sorry

end NUMINAMATH_CALUDE_circle_and_symmetry_line_l2551_255111


namespace NUMINAMATH_CALUDE_problem_N4_l2551_255149

theorem problem_N4 (a b : ℕ+) 
  (h : ∀ n : ℕ+, n > 2020^2020 → 
    ∃ m : ℕ+, Nat.Coprime m.val n.val ∧ (a^n.val + b^n.val ∣ a^m.val + b^m.val)) :
  a = b := by
  sorry

end NUMINAMATH_CALUDE_problem_N4_l2551_255149


namespace NUMINAMATH_CALUDE_complex_sum_real_part_l2551_255160

theorem complex_sum_real_part (z₁ z₂ z₃ : ℂ) (r : ℝ) 
  (h₁ : Complex.abs z₁ = 1) 
  (h₂ : Complex.abs z₂ = 1) 
  (h₃ : Complex.abs z₃ = 1) 
  (h₄ : Complex.abs (z₁ + z₂ + z₃) = r) : 
  (z₁ / z₂ + z₂ / z₃ + z₃ / z₁).re = (r^2 - 3) / 2 := by
  sorry

#check complex_sum_real_part

end NUMINAMATH_CALUDE_complex_sum_real_part_l2551_255160


namespace NUMINAMATH_CALUDE_sentence_B_is_error_free_l2551_255177

/-- Represents a sentence in the problem --/
inductive Sentence
| A : Sentence
| B : Sentence
| C : Sentence
| D : Sentence

/-- Checks if a sentence is free from linguistic errors --/
def is_error_free (s : Sentence) : Prop :=
  match s with
  | Sentence.A => False
  | Sentence.B => True
  | Sentence.C => False
  | Sentence.D => False

/-- The main theorem stating that Sentence B is free from linguistic errors --/
theorem sentence_B_is_error_free : is_error_free Sentence.B := by
  sorry

end NUMINAMATH_CALUDE_sentence_B_is_error_free_l2551_255177


namespace NUMINAMATH_CALUDE_intersection_length_and_product_l2551_255124

noncomputable section

-- Define the line L
def line (α : Real) (t : Real) : Real × Real :=
  (2 + t * Real.cos α, Real.sqrt 3 + t * Real.sin α)

-- Define the curve C
def curve (θ : Real) : Real × Real :=
  (2 * Real.cos θ, Real.sin θ)

-- Define point P
def P : Real × Real := (2, Real.sqrt 3)

-- Define the origin O
def O : Real × Real := (0, 0)

-- Define the distance between two points
def distance (p1 p2 : Real × Real) : Real :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_length_and_product (α : Real) :
  (α = Real.pi/3 → ∃ A B : Real × Real,
    A ≠ B ∧
    (∃ t : Real, line α t = A) ∧
    (∃ θ : Real, curve θ = A) ∧
    (∃ t : Real, line α t = B) ∧
    (∃ θ : Real, curve θ = B) ∧
    distance A B = 8 * Real.sqrt 10 / 13) ∧
  (Real.tan α = Real.sqrt 5 / 4 →
    ∃ A B : Real × Real,
    A ≠ B ∧
    (∃ t : Real, line α t = A) ∧
    (∃ θ : Real, curve θ = A) ∧
    (∃ t : Real, line α t = B) ∧
    (∃ θ : Real, curve θ = B) ∧
    distance P A * distance P B = distance O P ^ 2) :=
sorry

end

end NUMINAMATH_CALUDE_intersection_length_and_product_l2551_255124


namespace NUMINAMATH_CALUDE_trig_identity_l2551_255189

theorem trig_identity (θ φ : ℝ) 
  (h : (Real.sin θ)^6 / (Real.sin φ)^3 + (Real.cos θ)^6 / (Real.cos φ)^3 = 1) :
  (Real.sin φ)^6 / (Real.sin θ)^3 + (Real.cos φ)^6 / (Real.cos θ)^3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trig_identity_l2551_255189


namespace NUMINAMATH_CALUDE_tan_squared_f_equals_neg_cos_double_l2551_255110

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  1 / ((x / (x - 1)))

-- State the theorem
theorem tan_squared_f_equals_neg_cos_double (t : ℝ) 
  (h1 : 0 ≤ t) (h2 : t ≤ π/2) : f (Real.tan t ^ 2) = -Real.cos (2 * t) :=
by
  sorry


end NUMINAMATH_CALUDE_tan_squared_f_equals_neg_cos_double_l2551_255110


namespace NUMINAMATH_CALUDE_log_relation_l2551_255112

theorem log_relation (a b : ℝ) : 
  a = Real.log 1024 / Real.log 16 → b = Real.log 32 / Real.log 2 → a = (1/2) * b := by
  sorry

end NUMINAMATH_CALUDE_log_relation_l2551_255112


namespace NUMINAMATH_CALUDE_display_window_configurations_l2551_255143

/-- The number of permutations of n distinct objects -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The number of configurations for a single window with n books -/
def window_configurations (n : ℕ) : ℕ := factorial n

/-- The total number of configurations for two windows -/
def total_configurations (left_window : ℕ) (right_window : ℕ) : ℕ :=
  window_configurations left_window * window_configurations right_window

theorem display_window_configurations :
  total_configurations 3 3 = 36 :=
by sorry

end NUMINAMATH_CALUDE_display_window_configurations_l2551_255143


namespace NUMINAMATH_CALUDE_insert_books_combinations_l2551_255114

theorem insert_books_combinations (n m : ℕ) : 
  n = 5 → m = 3 → (n + 1) * (n + 2) * (n + 3) = 336 := by
  sorry

end NUMINAMATH_CALUDE_insert_books_combinations_l2551_255114


namespace NUMINAMATH_CALUDE_geometric_mean_minimum_l2551_255105

theorem geometric_mean_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geom_mean : 3 = Real.sqrt (9^a * 27^b)) :
  (3/a + 2/b) ≥ 12 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3 = Real.sqrt (9^a₀ * 27^b₀) ∧ (3/a₀ + 2/b₀) = 12 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_minimum_l2551_255105


namespace NUMINAMATH_CALUDE_cubic_polynomial_relation_l2551_255158

/-- Given a cubic polynomial f(x) = x^3 + 3x^2 + x + 1, and another cubic polynomial h
    such that h(0) = 1 and the roots of h are the cubes of the roots of f,
    prove that h(-8) = -115. -/
theorem cubic_polynomial_relation (f h : ℝ → ℝ) : 
  (∀ x, f x = x^3 + 3*x^2 + x + 1) →
  (∃ a b c : ℝ, ∀ x, h x = (x - a^3) * (x - b^3) * (x - c^3)) →
  h 0 = 1 →
  (∀ x, f x = 0 ↔ h (x^3) = 0) →
  h (-8) = -115 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_relation_l2551_255158


namespace NUMINAMATH_CALUDE_sturgeon_books_problem_l2551_255135

theorem sturgeon_books_problem (total_volumes : ℕ) (paperback_cost hardcover_cost total_cost : ℕ) 
  (h : total_volumes = 10)
  (hp : paperback_cost = 15)
  (hh : hardcover_cost = 25)
  (ht : total_cost = 220) :
  ∃ (hardcover_count : ℕ), 
    hardcover_count * hardcover_cost + (total_volumes - hardcover_count) * paperback_cost = total_cost ∧
    hardcover_count = 7 := by
  sorry

end NUMINAMATH_CALUDE_sturgeon_books_problem_l2551_255135


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l2551_255139

theorem ceiling_floor_sum : ⌈(7 : ℚ) / 3⌉ + ⌊-(7 : ℚ) / 3⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l2551_255139


namespace NUMINAMATH_CALUDE_fifteenth_set_sum_l2551_255132

def first_element (n : ℕ) : ℕ := 
  1 + (n - 1) * n / 2

def last_element (n : ℕ) : ℕ := 
  first_element n + n - 1

def set_sum (n : ℕ) : ℕ := 
  n * (first_element n + last_element n) / 2

theorem fifteenth_set_sum : set_sum 15 = 1695 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_set_sum_l2551_255132


namespace NUMINAMATH_CALUDE_sport_formulation_corn_syrup_amount_l2551_255145

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation of the drink -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation of the drink -/
def sport_ratio : DrinkRatio :=
  { flavoring := standard_ratio.flavoring,
    corn_syrup := standard_ratio.corn_syrup / 3,
    water := standard_ratio.water * 2 }

/-- Amount of water in the large bottle of sport formulation -/
def water_amount : ℚ := 45

theorem sport_formulation_corn_syrup_amount :
  (water_amount * sport_ratio.corn_syrup) / sport_ratio.water = water_amount :=
sorry

end NUMINAMATH_CALUDE_sport_formulation_corn_syrup_amount_l2551_255145


namespace NUMINAMATH_CALUDE_sets_intersection_empty_l2551_255138

-- Define set A
def A : Set ℝ := {x | x^2 + 5*x + 6 ≤ 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.sqrt (-x^2 + 2*x + 15)}

-- Define set C
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem statement
theorem sets_intersection_empty (a : ℝ) : (A ∪ B) ∩ C a = ∅ ↔ a ≥ 5 ∨ a ≤ -4 := by
  sorry

end NUMINAMATH_CALUDE_sets_intersection_empty_l2551_255138


namespace NUMINAMATH_CALUDE_rhombus_diagonals_l2551_255190

/-- Given a rhombus with perimeter 100 cm and sum of diagonals 62 cm, 
    prove that its diagonals are 48 cm and 14 cm. -/
theorem rhombus_diagonals (s : ℝ) (d₁ d₂ : ℝ) 
  (h_perimeter : 4 * s = 100)
  (h_diag_sum : d₁ + d₂ = 62)
  (h_pythag : s^2 = (d₁/2)^2 + (d₂/2)^2) :
  (d₁ = 48 ∧ d₂ = 14) ∨ (d₁ = 14 ∧ d₂ = 48) := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_l2551_255190


namespace NUMINAMATH_CALUDE_second_catch_up_race_result_l2551_255142

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ

/-- Represents the state of the race -/
structure RaceState where
  runner1 : Runner
  runner2 : Runner
  laps_completed : ℝ

/-- The race setup with initial conditions -/
def initial_race : RaceState :=
  { runner1 := { speed := 3 },
    runner2 := { speed := 1 },
    laps_completed := 0.5 }

/-- The race state after the second runner doubles their speed -/
def race_after_speed_up (r : RaceState) : RaceState :=
  { runner1 := r.runner1,
    runner2 := { speed := 2 * r.runner2.speed },
    laps_completed := r.laps_completed }

/-- Theorem stating that the first runner will catch up again at 2.5 laps -/
theorem second_catch_up (r : RaceState) :
  let r' := race_after_speed_up r
  r'.runner1.speed > r'.runner2.speed →
  r.runner1.speed = 3 * r.runner2.speed →
  r.laps_completed = 0.5 →
  ∃ t : ℝ, t > 0 ∧ r'.runner1.speed * t = (2.5 - r.laps_completed + 1) * r'.runner2.speed * t :=
by
  sorry

/-- Main theorem combining all conditions and results -/
theorem race_result :
  let r := initial_race
  let r' := race_after_speed_up r
  r'.runner1.speed > r'.runner2.speed ∧
  r.runner1.speed = 3 * r.runner2.speed ∧
  r.laps_completed = 0.5 ∧
  (∃ t : ℝ, t > 0 ∧ r'.runner1.speed * t = (2.5 - r.laps_completed + 1) * r'.runner2.speed * t) :=
by
  sorry

end NUMINAMATH_CALUDE_second_catch_up_race_result_l2551_255142
