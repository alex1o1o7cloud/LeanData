import Mathlib

namespace percentage_problem_l1837_183779

-- Define the percentage P
def P : ℝ := sorry

-- Theorem to prove
theorem percentage_problem : P = 45 := by
  -- Define the conditions
  have h1 : P / 100 * 60 = 35 / 100 * 40 + 13 := sorry
  
  -- Prove that P equals 45
  sorry

end percentage_problem_l1837_183779


namespace probability_red_in_middle_l1837_183788

/- Define the types of rosebushes -/
inductive Rosebush
| Red
| White

/- Define a row of rosebushes -/
def Row := List Rosebush

/- Define a function to check if the middle two rosebushes are red -/
def middleTwoAreRed (row : Row) : Bool :=
  match row with
  | [_, Rosebush.Red, Rosebush.Red, _] => true
  | _ => false

/- Define a function to generate all possible arrangements -/
def allArrangements : List Row :=
  sorry

/- Define a function to count arrangements with red rosebushes in the middle -/
def countRedInMiddle (arrangements : List Row) : Nat :=
  sorry

/- Theorem statement -/
theorem probability_red_in_middle :
  let arrangements := allArrangements
  let total := arrangements.length
  let favorable := countRedInMiddle arrangements
  (favorable : ℚ) / total = 1 / 6 := by
  sorry

end probability_red_in_middle_l1837_183788


namespace greatest_abcba_divisible_by_13_l1837_183778

def is_abcba (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = a * 10000 + b * 1000 + c * 100 + b * 10 + a

theorem greatest_abcba_divisible_by_13 :
  ∀ n : ℕ,
    10000 ≤ n ∧ n < 100000 ∧
    is_abcba n ∧
    n % 13 = 0 →
    n ≤ 96769 :=
by sorry

end greatest_abcba_divisible_by_13_l1837_183778


namespace r_value_l1837_183782

/-- The polynomial 8x^3 - 4x^2 - 42x + 45 -/
def P (x : ℝ) : ℝ := 8 * x^3 - 4 * x^2 - 42 * x + 45

/-- (x - r)^2 divides P(x) -/
def divides (r : ℝ) : Prop := ∃ Q : ℝ → ℝ, ∀ x, P x = (x - r)^2 * Q x

theorem r_value : ∃ r : ℝ, divides r ∧ r = 3/2 := by sorry

end r_value_l1837_183782


namespace smallest_k_for_inequality_l1837_183733

theorem smallest_k_for_inequality : ∃ k : ℕ, k = 8 ∧ 
  (∀ m : ℕ, 64^m > 4^22 → m ≥ k) ∧ 64^k > 4^22 := by
  sorry

end smallest_k_for_inequality_l1837_183733


namespace max_value_theorem_l1837_183790

theorem max_value_theorem (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_constraint : a^2 + b^2 + 4*c^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 2 ∧ ∀ x, x = a*b + 2*a*c + 3*Real.sqrt 2*b*c → x ≤ max :=
sorry

end max_value_theorem_l1837_183790


namespace q_undetermined_l1837_183740

theorem q_undetermined (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬p) : 
  (q ∨ ¬q) := by sorry

end q_undetermined_l1837_183740


namespace sum_of_max_min_f_l1837_183744

/-- Given a > 0, prove that the sum of the maximum and minimum values of the function
f(x) = (2009^(x+1) + 2007) / (2009^x + 1) + sin x on the interval [-a, a] is equal to 4016. -/
theorem sum_of_max_min_f (a : ℝ) (h : a > 0) : 
  let f : ℝ → ℝ := λ x ↦ (2009^(x+1) + 2007) / (2009^x + 1) + Real.sin x
  (⨆ x ∈ Set.Icc (-a) a, f x) + (⨅ x ∈ Set.Icc (-a) a, f x) = 4016 := by
  sorry

end sum_of_max_min_f_l1837_183744


namespace friends_hiking_distance_l1837_183732

-- Define the hiking scenario
structure HikingScenario where
  total_time : Real
  birgit_time_diff : Real
  birgit_time : Real
  birgit_distance : Real

-- Define the theorem
theorem friends_hiking_distance (h : HikingScenario) 
  (h_total_time : h.total_time = 3.5) 
  (h_birgit_time_diff : h.birgit_time_diff = 4) 
  (h_birgit_time : h.birgit_time = 48) 
  (h_birgit_distance : h.birgit_distance = 8) : 
  (h.total_time * 60) / (h.birgit_time / h.birgit_distance + h.birgit_time_diff) = 21 := by
  sorry


end friends_hiking_distance_l1837_183732


namespace sum_of_cubes_l1837_183720

theorem sum_of_cubes (x y z c d : ℝ) 
  (h1 : x * y * z = c)
  (h2 : 1 / x^3 + 1 / y^3 + 1 / z^3 = d) :
  (x + y + z)^3 = d * c^3 + 3 * c - 3 * c * d :=
by sorry

end sum_of_cubes_l1837_183720


namespace max_value_sum_l1837_183774

theorem max_value_sum (a b c : ℝ) (h : a^2 + 2*b^2 + 3*c^2 = 6) :
  ∃ (M : ℝ), M = Real.sqrt 11 ∧ a + b + c ≤ M ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ + b₀ + c₀ = M := by
  sorry

end max_value_sum_l1837_183774


namespace valid_arrangements_count_l1837_183728

/-- Represents the four grade levels --/
inductive Grade
| Freshman
| Sophomore
| Junior
| Senior

/-- Represents a student --/
structure Student where
  grade : Grade
  isTwin : Bool

/-- Represents the arrangement of students in a car --/
def CarArrangement := List Student

/-- Total number of students --/
def totalStudents : Nat := 8

/-- Number of students per grade --/
def studentsPerGrade : Nat := 2

/-- Number of students per car --/
def studentsPerCar : Nat := 4

/-- The twin sisters are freshmen --/
def twinSisters : List Student := [
  { grade := Grade.Freshman, isTwin := true },
  { grade := Grade.Freshman, isTwin := true }
]

/-- Checks if an arrangement has exactly two students from the same grade --/
def hasTwoSameGrade (arrangement : CarArrangement) : Bool :=
  sorry

/-- Counts the number of valid arrangements for Car A --/
def countValidArrangements : Nat :=
  sorry

theorem valid_arrangements_count :
  countValidArrangements = 24 := by sorry

end valid_arrangements_count_l1837_183728


namespace complex_arithmetic_equality_l1837_183785

theorem complex_arithmetic_equality : 
  4 * (7 * 24) / 3 + 5 * (13 * 15) - 2 * (6 * 28) + 7 * (3 * 19) / 2 = 1062.5 := by
  sorry

end complex_arithmetic_equality_l1837_183785


namespace min_value_theorem_l1837_183723

theorem min_value_theorem (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2) ≥ 4 * Real.sqrt 5 + 8 ∧
  ((x^2 + 1) / (y - 2) + (y^2 + 1) / (x - 2) = 4 * Real.sqrt 5 + 8 ↔ x = Real.sqrt 5 + 2 ∧ y = Real.sqrt 5 + 2) :=
by sorry

end min_value_theorem_l1837_183723


namespace median_sum_bounds_l1837_183745

/-- Given a triangle ABC with medians m_a, m_b, m_c, and perimeter p,
    prove that the sum of the medians is between 3/2 and 2 times the perimeter. -/
theorem median_sum_bounds (m_a m_b m_c p : ℝ) (h_positive : m_a > 0 ∧ m_b > 0 ∧ m_c > 0 ∧ p > 0)
  (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = p ∧
    m_a^2 = (2*b^2 + 2*c^2 - a^2) / 4 ∧
    m_b^2 = (2*a^2 + 2*c^2 - b^2) / 4 ∧
    m_c^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  (3/2) * p < m_a + m_b + m_c ∧ m_a + m_b + m_c < 2 * p := by
  sorry

end median_sum_bounds_l1837_183745


namespace right_triangle_sin_c_l1837_183709

theorem right_triangle_sin_c (A B C : ℝ) (h_right_angle : A + B + C = π) 
  (h_B_90 : B = π / 2) (h_cos_A : Real.cos A = 3 / 5) : Real.sin C = 4 / 5 := by
  sorry

end right_triangle_sin_c_l1837_183709


namespace sixteen_tourists_remain_l1837_183734

/-- Calculates the number of tourists remaining after a dangerous rainforest tour --/
def tourists_remaining (initial : ℕ) : ℕ :=
  let after_anaconda := initial - 3
  let poisoned := (2 * after_anaconda) / 3
  let recovered := (2 * poisoned) / 9
  let after_poison := after_anaconda - poisoned + recovered
  let snake_bitten := after_poison / 4
  let saved_from_snakes := (3 * snake_bitten) / 5
  after_poison - snake_bitten + saved_from_snakes

/-- Theorem stating that 16 tourists remain at the end of the tour --/
theorem sixteen_tourists_remain : tourists_remaining 42 = 16 := by
  sorry


end sixteen_tourists_remain_l1837_183734


namespace range_of_m_l1837_183710

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) : 
  (1 / x + 4 / y = 1) → 
  (∃ x y, x > 0 ∧ y > 0 ∧ 1 / x + 4 / y = 1 ∧ x + y / 4 < m^2 + 3*m) ↔ 
  (m < -4 ∨ m > 1) :=
by sorry

end range_of_m_l1837_183710


namespace all_statements_imply_theorem_l1837_183725

theorem all_statements_imply_theorem (p q r : Prop) : 
  ((p ∧ ¬q ∧ r) ∨ (¬p ∧ ¬q ∧ r) ∨ (p ∧ ¬q ∧ ¬r) ∨ (¬p ∧ q ∧ r)) → ((p → q) → r) := by
  sorry

#check all_statements_imply_theorem

end all_statements_imply_theorem_l1837_183725


namespace scenic_spot_selections_l1837_183738

theorem scenic_spot_selections (num_classes : ℕ) (num_spots : ℕ) : 
  num_classes = 3 → num_spots = 5 → (num_spots ^ num_classes) = 125 := by
  sorry

end scenic_spot_selections_l1837_183738


namespace collinear_vectors_y_value_l1837_183729

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

/-- The problem statement -/
theorem collinear_vectors_y_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (-6, y)
  collinear a b → y = -9 :=
by
  sorry

end collinear_vectors_y_value_l1837_183729


namespace commuter_distance_commuter_distance_is_12_sqrt_2_l1837_183719

/-- The distance from the starting point after a commuter drives 21 miles east, 
    15 miles south, 9 miles west, and 3 miles north. -/
theorem commuter_distance : ℝ :=
  let east : ℝ := 21
  let south : ℝ := 15
  let west : ℝ := 9
  let north : ℝ := 3
  let net_east_west : ℝ := east - west
  let net_south_north : ℝ := south - north
  Real.sqrt (net_east_west ^ 2 + net_south_north ^ 2)

/-- Proof that the commuter's distance from the starting point is 12√2 miles. -/
theorem commuter_distance_is_12_sqrt_2 : commuter_distance = 12 * Real.sqrt 2 := by
  sorry

end commuter_distance_commuter_distance_is_12_sqrt_2_l1837_183719


namespace circle_area_through_isosceles_triangle_vertices_l1837_183712

/-- The area of a circle passing through the vertices of an isosceles triangle -/
theorem circle_area_through_isosceles_triangle_vertices (a b c : ℝ) :
  a = 4 →  -- Two sides of the triangle are 4 units long
  b = 4 →  -- Two sides of the triangle are 4 units long
  c = 3 →  -- The base of the triangle is 3 units long
  a = b →  -- The triangle is isosceles
  ∃ (r : ℝ), r > 0 ∧ π * r^2 = (256/55) * π := by
  sorry

end circle_area_through_isosceles_triangle_vertices_l1837_183712


namespace batsman_highest_score_l1837_183736

theorem batsman_highest_score 
  (total_innings : ℕ) 
  (average : ℚ) 
  (score_difference : ℕ) 
  (average_excluding_extremes : ℚ) 
  (h : total_innings = 46)
  (h1 : average = 60)
  (h2 : score_difference = 180)
  (h3 : average_excluding_extremes = 58) :
  ∃ (highest_score lowest_score : ℕ),
    (highest_score : ℚ) - lowest_score = score_difference ∧
    (highest_score + lowest_score : ℚ) = 
      total_innings * average - (total_innings - 2) * average_excluding_extremes ∧
    highest_score = 194 := by
  sorry

end batsman_highest_score_l1837_183736


namespace fastest_growing_function_l1837_183796

/-- Proves that 0.001e^x grows faster than 1000ln(x), x^1000, and 1000⋅2^x as x approaches infinity -/
theorem fastest_growing_function :
  ∀ (ε : ℝ), ε > 0 → ∃ (X : ℝ), ∀ (x : ℝ), x > X →
    (0.001 * Real.exp x > 1000 * Real.log x) ∧
    (0.001 * Real.exp x > x^1000) ∧
    (0.001 * Real.exp x > 1000 * 2^x) :=
sorry

end fastest_growing_function_l1837_183796


namespace motel_room_rate_l1837_183701

theorem motel_room_rate (total_rent : ℕ) (lower_rate : ℕ) (reduction_percentage : ℚ) 
  (num_rooms_changed : ℕ) (h1 : total_rent = 2000) (h2 : lower_rate = 40) 
  (h3 : reduction_percentage = 1/10) (h4 : num_rooms_changed = 10) : 
  ∃ (higher_rate : ℕ), 
    (∃ (num_lower_rooms num_higher_rooms : ℕ), 
      total_rent = lower_rate * num_lower_rooms + higher_rate * num_higher_rooms ∧
      total_rent - (reduction_percentage * total_rent) = 
        lower_rate * (num_lower_rooms + num_rooms_changed) + 
        higher_rate * (num_higher_rooms - num_rooms_changed)) ∧
    higher_rate = 60 := by
  sorry

end motel_room_rate_l1837_183701


namespace min_value_sum_l1837_183783

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ 3 / Real.rpow 162 (1/3) ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    a' / (3 * b') + b' / (6 * c') + c' / (9 * a') = 3 / Real.rpow 162 (1/3) :=
sorry

end min_value_sum_l1837_183783


namespace problem_solid_surface_area_l1837_183753

/-- Represents a solid formed by unit cubes -/
structure CubeSolid where
  length : ℕ
  width : ℕ
  height : ℕ
  additional_cubes : ℕ

/-- Calculates the surface area of the CubeSolid -/
def surface_area (solid : CubeSolid) : ℕ :=
  2 * (solid.length * solid.height + solid.additional_cubes) + -- front and back
  (solid.length * solid.width + (solid.length * solid.width - solid.additional_cubes)) + -- top and bottom
  2 * (solid.width * solid.height) -- left and right

/-- The specific solid described in the problem -/
def problem_solid : CubeSolid :=
  { length := 4
    width := 3
    height := 1
    additional_cubes := 2 }

theorem problem_solid_surface_area :
  surface_area problem_solid = 42 := by
  sorry

end problem_solid_surface_area_l1837_183753


namespace m_range_l1837_183704

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∀ x, -m * x^2 + 2*x - m > 0

def q (m : ℝ) : Prop := ∀ x > 0, (4/x + x - (m - 1)) > 2

-- Define the theorem
theorem m_range :
  (∃ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) →
  (∃ m : ℝ, m ≥ -1 ∧ m < 3) ∧ (∀ m : ℝ, m < -1 ∨ m ≥ 3 → ¬(p m ∨ q m) ∨ (p m ∧ q m)) :=
sorry

end m_range_l1837_183704


namespace x_squared_plus_inverse_squared_l1837_183708

theorem x_squared_plus_inverse_squared (x : ℝ) (h : x^4 + 1/x^4 = 2398) : 
  x^2 + 1/x^2 = 20 * Real.sqrt 6 := by
  sorry

end x_squared_plus_inverse_squared_l1837_183708


namespace complex_magnitude_l1837_183775

theorem complex_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 9)
  (h3 : Complex.abs (z + w) = 2) :
  Complex.abs z = 3 * Real.sqrt 5 := by
  sorry

end complex_magnitude_l1837_183775


namespace squirrel_count_ratio_l1837_183798

theorem squirrel_count_ratio :
  ∀ (first_count second_count : ℕ),
  first_count = 12 →
  first_count + second_count = 28 →
  second_count > first_count →
  (second_count : ℚ) / first_count = 4 / 3 := by
sorry

end squirrel_count_ratio_l1837_183798


namespace g_of_three_equals_twentyone_l1837_183786

-- Define the function g
noncomputable def g : ℝ → ℝ := sorry

-- State the theorem
theorem g_of_three_equals_twentyone :
  (∀ x : ℝ, g (2 * x - 5) = 3 * x + 9) →
  g 3 = 21 := by
  sorry

end g_of_three_equals_twentyone_l1837_183786


namespace five_black_cards_taken_out_l1837_183758

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (original_black_cards : ℕ)
  (remaining_black_cards : ℕ)

/-- Defines a standard deck with 52 total cards and 26 black cards -/
def standard_deck : Deck :=
  { total_cards := 52,
    original_black_cards := 26,
    remaining_black_cards := 21 }

/-- Calculates the number of black cards taken out from a deck -/
def black_cards_taken_out (d : Deck) : ℕ :=
  d.original_black_cards - d.remaining_black_cards

/-- Theorem stating that 5 black cards were taken out from the standard deck -/
theorem five_black_cards_taken_out :
  black_cards_taken_out standard_deck = 5 := by
  sorry

end five_black_cards_taken_out_l1837_183758


namespace aquarium_feeding_ratio_l1837_183724

/-- The ratio of buckets fed to other sea animals to buckets fed to sharks -/
def ratio_other_to_sharks : ℚ := 5

theorem aquarium_feeding_ratio : 
  let sharks_buckets : ℕ := 4
  let dolphins_buckets : ℕ := sharks_buckets / 2
  let total_buckets : ℕ := 546
  let days : ℕ := 21
  
  ∃ (other_buckets : ℚ),
    other_buckets = ratio_other_to_sharks * sharks_buckets ∧
    total_buckets = (sharks_buckets + dolphins_buckets + other_buckets) * days :=
by sorry

end aquarium_feeding_ratio_l1837_183724


namespace inverse_variation_problem_l1837_183773

theorem inverse_variation_problem (x w : ℝ) (h : ∃ (c : ℝ), ∀ (x w : ℝ), x^4 * w^(1/4) = c) :
  (x = 3 ∧ w = 16) → (x = 6 → w = 1/4096) :=
by sorry

end inverse_variation_problem_l1837_183773


namespace min_lcm_a_c_l1837_183799

theorem min_lcm_a_c (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 24) :
  ∃ (a' c' : ℕ), Nat.lcm a' c' = 30 ∧ 
  (∀ (x y : ℕ), Nat.lcm x b = 20 → Nat.lcm b y = 24 → Nat.lcm a' c' ≤ Nat.lcm x y) :=
sorry

end min_lcm_a_c_l1837_183799


namespace coaching_start_date_l1837_183737

/-- Represents a date in a year -/
structure Date :=
  (month : Nat)
  (day : Nat)

/-- Calculates the number of days from the start of the year to a given date in a non-leap year -/
def daysFromYearStart (d : Date) : Nat :=
  sorry

/-- Calculates the date that is a given number of days before another date in a non-leap year -/
def dateBeforeDays (d : Date) (days : Nat) : Date :=
  sorry

theorem coaching_start_date :
  let end_date : Date := ⟨9, 4⟩  -- September 4
  let coaching_duration : Nat := 245
  let start_date := dateBeforeDays end_date coaching_duration
  start_date = ⟨1, 2⟩  -- January 2
  :=
sorry

end coaching_start_date_l1837_183737


namespace fraction_equality_l1837_183721

theorem fraction_equality (x y z : ℝ) 
  (h : (x + 4) / 2 = (y + 9) / (z - 3) ∧ (x + 4) / 2 = (x + 5) / (z - 5)) : 
  x / y = 1 / 2 := by
  sorry

end fraction_equality_l1837_183721


namespace friends_not_going_to_movies_l1837_183793

theorem friends_not_going_to_movies (total_friends : ℕ) (friends_going : ℕ) : 
  total_friends = 15 → friends_going = 8 → total_friends - friends_going = 7 := by
  sorry

end friends_not_going_to_movies_l1837_183793


namespace cousins_ages_sum_l1837_183731

theorem cousins_ages_sum : 
  ∀ (a b c d : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    0 < a ∧ a < 10 →
    0 < b ∧ b < 10 →
    0 < c ∧ c < 10 →
    0 < d ∧ d < 10 →
    (a * b = 40 ∨ a * c = 40 ∨ a * d = 40 ∨ b * c = 40 ∨ b * d = 40 ∨ c * d = 40) →
    (a * b = 36 ∨ a * c = 36 ∨ a * d = 36 ∨ b * c = 36 ∨ b * d = 36 ∨ c * d = 36) →
    a + b + c + d = 26 :=
by sorry

end cousins_ages_sum_l1837_183731


namespace phil_quarters_left_l1837_183750

def initial_amount : ℚ := 40
def pizza_cost : ℚ := 2.75
def soda_cost : ℚ := 1.50
def jeans_cost : ℚ := 11.50
def quarter_value : ℚ := 0.25

theorem phil_quarters_left : 
  let total_spent := pizza_cost + soda_cost + jeans_cost
  let remaining_amount := initial_amount - total_spent
  (remaining_amount / quarter_value).floor = 97 := by sorry

end phil_quarters_left_l1837_183750


namespace unique_intersection_l1837_183746

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 - x + 1

/-- The line equation -/
def line (k : ℝ) (x : ℝ) : ℝ := 4*x + k

/-- Theorem stating the condition for exactly one intersection point -/
theorem unique_intersection (k : ℝ) : 
  (∃! x, parabola x = line k x) ↔ k = -21/4 := by
  sorry

end unique_intersection_l1837_183746


namespace high_correlation_implies_r_close_to_one_l1837_183772

-- Define a type for variables
def Variable : Type := ℝ

-- Define a correlation coefficient
def correlation_coefficient (x y : Variable) : ℝ := sorry

-- Define what it means for the degree of linear correlation to be very high
def high_linear_correlation (x y : Variable) : Prop :=
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ abs (correlation_coefficient x y) > 1 - ε

-- The theorem to prove
theorem high_correlation_implies_r_close_to_one (x y : Variable) :
  high_linear_correlation x y → ∃ (δ : ℝ), δ > 0 ∧ δ < 0.1 ∧ abs (correlation_coefficient x y) > 1 - δ :=
sorry

end high_correlation_implies_r_close_to_one_l1837_183772


namespace remaining_coin_value_l1837_183756

/-- Represents the number and type of coins --/
structure Coins where
  quarters : Nat
  dimes : Nat
  nickels : Nat

/-- Calculates the total value of coins in cents --/
def coinValue (c : Coins) : Nat :=
  c.quarters * 25 + c.dimes * 10 + c.nickels * 5

/-- Represents Olivia's initial coins --/
def initialCoins : Coins :=
  { quarters := 11, dimes := 15, nickels := 7 }

/-- Represents the coins spent on purchases --/
def purchasedCoins : Coins :=
  { quarters := 1, dimes := 8, nickels := 3 }

/-- Calculates the remaining coins after purchases --/
def remainingCoins (initial : Coins) (purchased : Coins) : Coins :=
  { quarters := initial.quarters - purchased.quarters,
    dimes := initial.dimes - purchased.dimes,
    nickels := initial.nickels - purchased.nickels }

theorem remaining_coin_value :
  coinValue (remainingCoins initialCoins purchasedCoins) = 340 := by
  sorry


end remaining_coin_value_l1837_183756


namespace product_equals_24255_l1837_183730

theorem product_equals_24255 : 3^2 * 5 * 7^2 * 11 = 24255 := by
  sorry

end product_equals_24255_l1837_183730


namespace log_equation_holds_l1837_183768

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 4) * (Real.log 7 / Real.log x) = Real.log 7 / Real.log 4 :=
by sorry

end log_equation_holds_l1837_183768


namespace left_handed_rock_lovers_l1837_183764

theorem left_handed_rock_lovers (total : ℕ) (left_handed : ℕ) (rock_lovers : ℕ) (right_handed_non_rock : ℕ) :
  total = 25 →
  left_handed = 10 →
  rock_lovers = 18 →
  right_handed_non_rock = 3 →
  left_handed + (total - left_handed) = total →
  ∃ (left_handed_rock : ℕ),
    left_handed_rock + (left_handed - left_handed_rock) + (rock_lovers - left_handed_rock) + right_handed_non_rock = total ∧
    left_handed_rock = 6 := by
  sorry

end left_handed_rock_lovers_l1837_183764


namespace product_equals_zero_l1837_183777

theorem product_equals_zero : (3 - 5) * (3 - 4) * (3 - 3) * (3 - 2) * (3 - 1) * 3 = 0 := by
  sorry

end product_equals_zero_l1837_183777


namespace negation_of_universal_proposition_l1837_183751

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0) :=
by sorry

end negation_of_universal_proposition_l1837_183751


namespace negation_of_implication_negation_of_x_squared_positive_l1837_183718

theorem negation_of_implication (P Q : Prop) :
  ¬(P → Q) ↔ (P ∧ ¬Q) :=
by sorry

theorem negation_of_x_squared_positive :
  ¬(∀ x : ℝ, x > 0 → x^2 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 ≤ 0) :=
by sorry

end negation_of_implication_negation_of_x_squared_positive_l1837_183718


namespace f_min_value_iff_a_range_l1837_183761

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then x^2 - 2*a*x - 2 else x + 36/x - 6*a

-- State the theorem
theorem f_min_value_iff_a_range (a : ℝ) :
  (∀ x : ℝ, f a 2 ≤ f a x) ↔ 2 ≤ a ∧ a ≤ 5 :=
sorry

end f_min_value_iff_a_range_l1837_183761


namespace average_mark_five_subjects_l1837_183787

/-- Given a student's marks in six subjects, prove that the average mark for five subjects
    (excluding physics) is 70, when the total marks are 350 more than the physics marks. -/
theorem average_mark_five_subjects (physics_mark : ℕ) : 
  let total_marks : ℕ := physics_mark + 350
  let remaining_marks : ℕ := total_marks - physics_mark
  let num_subjects : ℕ := 5
  (remaining_marks : ℚ) / num_subjects = 70 := by
  sorry

end average_mark_five_subjects_l1837_183787


namespace f_max_min_implies_a_range_l1837_183776

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + a + 6

/-- Theorem: If f has both a maximum and a minimum, then a < -3 or a > 6 -/
theorem f_max_min_implies_a_range (a : ℝ) : 
  (∃ (x_max x_min : ℝ), ∀ x, f a x ≤ f a x_max ∧ f a x_min ≤ f a x) →
  a < -3 ∨ a > 6 :=
sorry

end f_max_min_implies_a_range_l1837_183776


namespace sqrt_two_thirds_times_sqrt_six_equals_two_l1837_183722

theorem sqrt_two_thirds_times_sqrt_six_equals_two :
  Real.sqrt (2 / 3) * Real.sqrt 6 = 2 := by
  sorry

end sqrt_two_thirds_times_sqrt_six_equals_two_l1837_183722


namespace alpha_value_proof_l1837_183749

theorem alpha_value_proof (α : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x^α) 
  (h2 : (deriv f) (-1) = -4) : α = 4 := by
  sorry

end alpha_value_proof_l1837_183749


namespace survey_sample_size_l1837_183735

/-- Represents a survey conducted in an urban area -/
structure UrbanSurvey where
  year : Nat
  month : Nat
  investigators : Nat
  households : Nat
  questionnaires : Nat

/-- Definition of sample size for an urban survey -/
def sampleSize (survey : UrbanSurvey) : Nat :=
  survey.questionnaires

/-- Theorem stating that the sample size of the given survey is 30,000 -/
theorem survey_sample_size :
  let survey : UrbanSurvey := {
    year := 2010
    month := 5  -- May
    investigators := 400
    households := 10000
    questionnaires := 30000
  }
  sampleSize survey = 30000 := by
  sorry


end survey_sample_size_l1837_183735


namespace cookie_calories_l1837_183705

/-- Calculates the number of calories per cookie in a box of cookies. -/
def calories_per_cookie (cookies_per_bag : ℕ) (bags_per_box : ℕ) (total_calories : ℕ) : ℕ :=
  total_calories / (cookies_per_bag * bags_per_box)

/-- Theorem: Given a box of cookies with 4 bags, 20 cookies per bag, and a total of 1600 calories,
    each cookie contains 20 calories. -/
theorem cookie_calories :
  calories_per_cookie 20 4 1600 = 20 := by
  sorry

end cookie_calories_l1837_183705


namespace log_comparison_l1837_183792

theorem log_comparison (a : ℝ) (h : a > 1) : Real.log a / Real.log (a - 1) > Real.log (a + 1) / Real.log a := by
  sorry

end log_comparison_l1837_183792


namespace lines_intersect_at_point_l1837_183781

/-- Represents a 2D point --/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a line parameterized by a point and a direction vector --/
structure Line where
  point : Point
  direction : Point

def line1 : Line :=
  { point := { x := 2, y := 3 },
    direction := { x := 3, y := -4 } }

def line2 : Line :=
  { point := { x := 4, y := 1 },
    direction := { x := 5, y := 3 } }

def intersection : Point :=
  { x := 26/11, y := 19/11 }

/-- Returns a point on the line for a given parameter value --/
def pointOnLine (l : Line) (t : ℚ) : Point :=
  { x := l.point.x + t * l.direction.x,
    y := l.point.y + t * l.direction.y }

theorem lines_intersect_at_point :
  ∃ (t u : ℚ), pointOnLine line1 t = intersection ∧ pointOnLine line2 u = intersection ∧
  ∀ (p : Point), (∃ (t' : ℚ), pointOnLine line1 t' = p) ∧ (∃ (u' : ℚ), pointOnLine line2 u' = p) →
  p = intersection := by
  sorry

end lines_intersect_at_point_l1837_183781


namespace four_digit_count_l1837_183727

/-- The count of four-digit numbers -/
def count_four_digit_numbers : ℕ := 9999 - 1000 + 1

/-- The first four-digit number -/
def first_four_digit : ℕ := 1000

/-- The last four-digit number -/
def last_four_digit : ℕ := 9999

theorem four_digit_count :
  count_four_digit_numbers = 9000 :=
by sorry

end four_digit_count_l1837_183727


namespace disjunction_truth_implication_false_l1837_183791

theorem disjunction_truth_implication_false : 
  ¬(∀ (p q : Prop), (p ∨ q) → (p ∧ q)) := by sorry

end disjunction_truth_implication_false_l1837_183791


namespace tangent_line_slope_l1837_183716

/-- The slope of a line tangent to the circle x^2 + y^2 - 4x + 2 = 0 is either 1 or -1 -/
theorem tangent_line_slope (m : ℝ) :
  (∀ x y : ℝ, y = m * x → x^2 + y^2 - 4*x + 2 = 0 → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
      (x' - x)^2 + (y' - y)^2 < δ^2 → x'^2 + y'^2 - 4*x' + 2 > 0) →
  m = 1 ∨ m = -1 :=
sorry

end tangent_line_slope_l1837_183716


namespace unique_solution_for_equation_l1837_183760

theorem unique_solution_for_equation :
  ∀ x y : ℝ,
    (Real.sqrt (1 / (4 - x^2)) + Real.sqrt (y^2 / (y - 1)) = 5/2) →
    (x = 0 ∧ y = 2) :=
by sorry

end unique_solution_for_equation_l1837_183760


namespace gcd_of_powers_of_101_plus_one_l1837_183748

theorem gcd_of_powers_of_101_plus_one (h : Nat.Prime 101) :
  Nat.gcd (101^5 + 1) (101^5 + 101^3 + 1) = 1 := by
  sorry

end gcd_of_powers_of_101_plus_one_l1837_183748


namespace equation_is_quadratic_l1837_183794

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ) (ha : a ≠ 0), ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 2

/-- Theorem: The given equation is a quadratic equation -/
theorem equation_is_quadratic : is_quadratic_equation f := by
  sorry


end equation_is_quadratic_l1837_183794


namespace company_gender_ratio_l1837_183784

/-- Represents the number of employees of each gender in a company -/
structure Company where
  male : ℕ
  female : ℕ

/-- The ratio of male to female employees -/
def genderRatio (c : Company) : ℚ :=
  c.male / c.female

theorem company_gender_ratio (c : Company) :
  c.male = 189 ∧ 
  genderRatio {male := c.male + 3, female := c.female} = 8 / 9 →
  genderRatio c = 7 / 8 := by
  sorry

end company_gender_ratio_l1837_183784


namespace divisibility_implies_p_q_values_l1837_183717

/-- A polynomial is divisible by (x + 2)(x - 2) if and only if it equals zero when x = 2 and x = -2 -/
def is_divisible_by_x2_minus4 (f : ℝ → ℝ) : Prop :=
  f 2 = 0 ∧ f (-2) = 0

/-- The polynomial x^5 - x^4 + x^3 - px^2 + qx - 8 -/
def polynomial (p q : ℝ) (x : ℝ) : ℝ :=
  x^5 - x^4 + x^3 - p*x^2 + q*x - 8

theorem divisibility_implies_p_q_values :
  ∀ p q : ℝ, is_divisible_by_x2_minus4 (polynomial p q) → p = -2 ∧ q = -12 := by
  sorry

end divisibility_implies_p_q_values_l1837_183717


namespace scientific_notation_56_9_billion_l1837_183739

def billion : ℝ := 1000000000

theorem scientific_notation_56_9_billion :
  56.9 * billion = 5.69 * (10 : ℝ) ^ 9 :=
sorry

end scientific_notation_56_9_billion_l1837_183739


namespace johns_investment_l1837_183706

theorem johns_investment (total_investment : ℝ) (alpha_rate beta_rate : ℝ) 
  (total_after_year : ℝ) (alpha_investment : ℝ) :
  total_investment = 1500 →
  alpha_rate = 0.04 →
  beta_rate = 0.06 →
  total_after_year = 1575 →
  alpha_investment = 750 →
  alpha_investment * (1 + alpha_rate) + 
    (total_investment - alpha_investment) * (1 + beta_rate) = total_after_year :=
by sorry

end johns_investment_l1837_183706


namespace product_evaluation_l1837_183754

theorem product_evaluation (n : ℕ) (h : n = 3) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 120 := by
  sorry

end product_evaluation_l1837_183754


namespace problem_1_problem_2_l1837_183762

-- Problem 1
theorem problem_1 : -1^2023 * ((-8) + 2 / (1/2)) - |(-3)| = 1 := by sorry

-- Problem 2
theorem problem_2 : ∃ x : ℚ, (x + 2) / 3 - (x - 1) / 2 = x + 2 ∧ x = -5/7 := by sorry

end problem_1_problem_2_l1837_183762


namespace unique_intersection_point_l1837_183769

/-- The function g(x) = x^3 - 9x^2 + 27x - 29 -/
def g (x : ℝ) : ℝ := x^3 - 9*x^2 + 27*x - 29

/-- The point (1, 1) is the unique intersection of y = g(x) and y = g^(-1)(x) -/
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, p.1 = g p.2 ∧ p.2 = g p.1 ∧ p = (1, 1) := by sorry

end unique_intersection_point_l1837_183769


namespace correct_average_l1837_183742

theorem correct_average (n : ℕ) (initial_avg : ℚ) (correction1 correction2 : ℚ) :
  n = 10 ∧ 
  initial_avg = 40.2 ∧ 
  correction1 = -19 ∧ 
  correction2 = 18 →
  (n * initial_avg + correction1 + correction2) / n = 40.1 := by
sorry

end correct_average_l1837_183742


namespace min_participants_in_race_l1837_183789

/-- Represents a participant in the race -/
structure Participant where
  name : String
  position : Nat

/-- Represents the race with its participants -/
structure Race where
  participants : List Participant

/-- Checks if the given race satisfies the conditions for Andrei -/
def satisfiesAndreiCondition (race : Race) : Prop :=
  ∃ (x : Nat), 3 * x + 1 = race.participants.length

/-- Checks if the given race satisfies the conditions for Dima -/
def satisfiesDimaCondition (race : Race) : Prop :=
  ∃ (y : Nat), 4 * y + 1 = race.participants.length

/-- Checks if the given race satisfies the conditions for Lenya -/
def satisfiesLenyaCondition (race : Race) : Prop :=
  ∃ (z : Nat), 5 * z + 1 = race.participants.length

/-- Checks if all participants have unique finishing positions -/
def uniqueFinishingPositions (race : Race) : Prop :=
  ∀ p1 p2 : Participant, p1 ∈ race.participants → p2 ∈ race.participants → 
    p1 ≠ p2 → p1.position ≠ p2.position

/-- The main theorem stating the minimum number of participants -/
theorem min_participants_in_race : 
  ∀ race : Race, 
    uniqueFinishingPositions race →
    satisfiesAndreiCondition race →
    satisfiesDimaCondition race →
    satisfiesLenyaCondition race →
    race.participants.length ≥ 61 :=
by
  sorry

end min_participants_in_race_l1837_183789


namespace fencing_calculation_l1837_183759

/-- The total fencing length for a square playground and a rectangular garden -/
def total_fencing (playground_side : ℝ) (garden_length garden_width : ℝ) : ℝ :=
  4 * playground_side + 2 * (garden_length + garden_width)

/-- Theorem: The total fencing for a square playground with side length 27 yards
    and a rectangular garden of 12 yards by 9 yards is equal to 150 yards -/
theorem fencing_calculation :
  total_fencing 27 12 9 = 150 := by
  sorry

end fencing_calculation_l1837_183759


namespace partition_sum_exists_l1837_183743

theorem partition_sum_exists : ∃ (A B : Finset ℕ),
  A ∪ B = Finset.range 14 \ {0} ∧
  A ∩ B = ∅ ∧
  A.card = 5 ∧
  B.card = 8 ∧
  3 * (A.sum id) + 7 * (B.sum id) = 433 := by
  sorry

end partition_sum_exists_l1837_183743


namespace tommy_balloons_l1837_183747

/-- Prove that Tommy started with 26 balloons given the conditions of the problem -/
theorem tommy_balloons (initial : ℕ) (from_mom : ℕ) (after_mom : ℕ) (total : ℕ) : 
  after_mom = 26 → total = 60 → initial + from_mom = total → initial = 26 := by
  sorry

end tommy_balloons_l1837_183747


namespace function_value_at_negative_five_hundred_l1837_183715

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) + 2 * x = x * f y + 2 * f x

theorem function_value_at_negative_five_hundred
  (f : ℝ → ℝ)
  (h1 : FunctionalEquation f)
  (h2 : f (-2) = 11) :
  f (-500) = -487 := by
sorry

end function_value_at_negative_five_hundred_l1837_183715


namespace sum_of_solutions_is_four_l1837_183767

theorem sum_of_solutions_is_four :
  let f : ℝ → ℝ := λ N ↦ N * (N - 4) - 12
  ∃ N₁ N₂ : ℝ, (f N₁ = 0 ∧ f N₂ = 0) ∧ N₁ + N₂ = 4 :=
by sorry

end sum_of_solutions_is_four_l1837_183767


namespace school_survey_most_suitable_for_census_l1837_183711

/-- Represents a survey type --/
inductive SurveyType
  | CityResidents
  | CarBatch
  | LightTubeBatch
  | SchoolStudents

/-- Determines if a survey type is suitable for a census --/
def isSuitableForCensus (s : SurveyType) : Prop :=
  match s with
  | .SchoolStudents => True
  | _ => False

/-- Theorem stating that the school students survey is the most suitable for a census --/
theorem school_survey_most_suitable_for_census :
  ∀ s : SurveyType, isSuitableForCensus s ↔ s = SurveyType.SchoolStudents :=
by sorry

end school_survey_most_suitable_for_census_l1837_183711


namespace two_tangents_iff_a_in_range_l1837_183741

/-- Definition of the circle C -/
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + a*x + 2*y + a^2 = 0

/-- Point A -/
def point_A : ℝ × ℝ := (1, 2)

/-- Condition for exactly two tangents -/
def has_two_tangents (a : ℝ) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (circle_C a (center.1) (center.2)) ∧
    ((point_A.1 - center.1)^2 + (point_A.2 - center.2)^2 > radius^2) ∧
    (radius^2 > 0)

/-- Main theorem -/
theorem two_tangents_iff_a_in_range :
  ∀ a : ℝ, has_two_tangents a ↔ -2*(3:ℝ).sqrt/3 < a ∧ a < 2*(3:ℝ).sqrt/3 :=
sorry

end two_tangents_iff_a_in_range_l1837_183741


namespace hotel_arrangement_count_l1837_183713

/-- Represents the number of ways to arrange people in rooms -/
def arrangement_count (n : ℕ) (r : ℕ) (m : ℕ) : ℕ := sorry

/-- The number of people -/
def total_people : ℕ := 5

/-- The number of rooms -/
def total_rooms : ℕ := 3

/-- The number of people who cannot be in the same room -/
def restricted_people : ℕ := 2

/-- Theorem stating the number of possible arrangements -/
theorem hotel_arrangement_count :
  arrangement_count total_people total_rooms restricted_people = 114 := by
  sorry

end hotel_arrangement_count_l1837_183713


namespace average_age_after_leaving_l1837_183770

theorem average_age_after_leaving (initial_people : ℕ) (initial_avg : ℚ) (leaving_age : ℕ) (remaining_people : ℕ) :
  initial_people = 7 →
  initial_avg = 32 →
  leaving_age = 22 →
  remaining_people = 6 →
  (initial_people * initial_avg - leaving_age) / remaining_people = 34 := by
sorry

end average_age_after_leaving_l1837_183770


namespace unique_number_exists_l1837_183752

def is_valid_number (n : ℕ) : Prop :=
  ∀ k ∈ Finset.range 10, n % (k + 3) = k + 2

theorem unique_number_exists : 
  ∃! n : ℕ, is_valid_number n ∧ n = 27719 := by sorry

end unique_number_exists_l1837_183752


namespace problem_solution_l1837_183702

def f (x : ℝ) := |2*x - 7| + 1

def g (x : ℝ) := f x - 2*|x - 1|

theorem problem_solution :
  (∀ x : ℝ, f x ≤ x ↔ (8/3 ≤ x ∧ x ≤ 6)) ∧
  (∀ x : ℝ, g x ≥ -4) ∧
  (∀ a : ℝ, (∃ x : ℝ, g x ≤ a) ↔ a ≥ -4) := by
  sorry

end problem_solution_l1837_183702


namespace not_right_triangle_l1837_183703

theorem not_right_triangle (A B C : ℝ) (h : A + B + C = 180) 
  (h_ratio : A / 3 = B / 4 ∧ B / 4 = C / 5) : 
  A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90 := by
  sorry

end not_right_triangle_l1837_183703


namespace bakers_earnings_l1837_183766

/-- The baker's earnings problem -/
theorem bakers_earnings (cakes_sold : ℕ) (cake_price : ℕ) (pies_sold : ℕ) (pie_price : ℕ) 
  (h1 : cakes_sold = 453)
  (h2 : cake_price = 12)
  (h3 : pies_sold = 126)
  (h4 : pie_price = 7) :
  cakes_sold * cake_price + pies_sold * pie_price = 6318 := by
  sorry


end bakers_earnings_l1837_183766


namespace crickets_collected_l1837_183780

theorem crickets_collected (total : ℕ) (more_needed : ℕ) (h : total = 11 ∧ more_needed = 4) : 
  total - more_needed = 7 := by
  sorry

end crickets_collected_l1837_183780


namespace max_value_cube_plus_one_l1837_183757

/-- Given that x + y = 1, prove that (x³+1)(y³+1) achieves its maximum value
    when x = (1 ± √5)/2 and y = (1 ∓ √5)/2 -/
theorem max_value_cube_plus_one (x y : ℝ) (h : x + y = 1) :
  ∃ (max_x max_y : ℝ), 
    (max_x = (1 + Real.sqrt 5) / 2 ∧ max_y = (1 - Real.sqrt 5) / 2) ∨
    (max_x = (1 - Real.sqrt 5) / 2 ∧ max_y = (1 + Real.sqrt 5) / 2) ∧
    ∀ (a b : ℝ), a + b = 1 → 
      (x^3 + 1) * (y^3 + 1) ≤ (max_x^3 + 1) * (max_y^3 + 1) :=
sorry

end max_value_cube_plus_one_l1837_183757


namespace positive_rational_number_l1837_183700

theorem positive_rational_number : ∃! x : ℚ, (x > 0) ∧
  (x = 1/2 ∨ x = Real.sqrt 2 * (-1) ∨ x = 0 ∨ x = Real.sqrt 3) := by
  sorry

end positive_rational_number_l1837_183700


namespace house_size_ratio_l1837_183765

/-- The size of Kennedy's house in square feet -/
def kennedy_house_size : ℝ := 10000

/-- The size of Benedict's house in square feet -/
def benedict_house_size : ℝ := 2350

/-- The additional size in square feet added to the ratio of house sizes -/
def additional_size : ℝ := 600

/-- Theorem stating that the ratio of (Kennedy's house size - additional size) to Benedict's house size is 4 -/
theorem house_size_ratio : 
  (kennedy_house_size - additional_size) / benedict_house_size = 4 := by
  sorry

end house_size_ratio_l1837_183765


namespace pizza_difference_l1837_183771

/-- Given that Seung-hyeon gave Su-yeon 2 pieces of pizza and then had 5 more pieces than Su-yeon,
    prove that Seung-hyeon had 9 more pieces than Su-yeon before giving. -/
theorem pizza_difference (s y : ℕ) : 
  s - 2 = y + 2 + 5 → s - y = 9 := by
  sorry

end pizza_difference_l1837_183771


namespace extreme_points_count_l1837_183714

-- Define the derivative of f(x)
def f_prime (x : ℝ) : ℝ := (x + 1)^2 * (x - 1) * (x - 2)

-- Define what an extreme point is
def is_extreme_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), y ≠ x → f y ≠ f x

-- State the theorem
theorem extreme_points_count :
  ∃ (f : ℝ → ℝ), (∀ x, deriv f x = f_prime x) ∧ 
  (∃ (a b : ℝ), a ≠ b ∧ 
    is_extreme_point f a ∧ 
    is_extreme_point f b ∧ 
    ∀ c, is_extreme_point f c → (c = a ∨ c = b)) :=
sorry

end extreme_points_count_l1837_183714


namespace hit_target_probability_l1837_183763

/-- The probability of hitting a target at least 2 times out of 3 independent shots,
    given that the probability of hitting the target in a single shot is 0.6 -/
theorem hit_target_probability :
  let p : ℝ := 0.6  -- Probability of hitting the target in a single shot
  let n : ℕ := 3    -- Total number of shots
  let k : ℕ := 2    -- Minimum number of successful hits

  -- Probability of hitting the target at least k times out of n shots
  (Finset.sum (Finset.range (n - k + 1)) (fun i =>
    (n.choose (k + i)) * p^(k + i) * (1 - p)^(n - k - i))) = 81 / 125 :=
by sorry

end hit_target_probability_l1837_183763


namespace intersection_of_M_and_N_l1837_183707

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end intersection_of_M_and_N_l1837_183707


namespace valid_combinations_l1837_183755

/-- A combination is valid if it satisfies the given equation and range constraints. -/
def is_valid_combination (x y z : ℕ) : Prop :=
  10 ≤ x ∧ x ≤ 20 ∧
  10 ≤ y ∧ y ≤ 20 ∧
  10 ≤ z ∧ z ≤ 20 ∧
  3 * x^2 - y^2 - 7 * z = 99

/-- The theorem states that there are exactly three valid combinations. -/
theorem valid_combinations :
  (∀ x y z : ℕ, is_valid_combination x y z ↔ 
    ((x = 15 ∧ y = 10 ∧ z = 68) ∨ 
     (x = 16 ∧ y = 12 ∧ z = 75) ∨ 
     (x = 18 ∧ y = 15 ∧ z = 78))) :=
by sorry

end valid_combinations_l1837_183755


namespace clothing_color_theorem_l1837_183726

-- Define the colors
inductive Color
| Red
| Blue

-- Define a structure for clothing
structure Clothing :=
  (tshirt : Color)
  (shorts : Color)

-- Define a function to check if two colors are different
def different_colors (c1 c2 : Color) : Prop :=
  c1 ≠ c2

-- Define the problem statement
theorem clothing_color_theorem 
  (alyna bohdan vika grysha : Clothing) : 
  (alyna.tshirt = Color.Red) →
  (bohdan.tshirt = Color.Red) →
  (different_colors alyna.shorts bohdan.shorts) →
  (different_colors vika.tshirt grysha.tshirt) →
  (vika.shorts = Color.Blue) →
  (grysha.shorts = Color.Blue) →
  (different_colors alyna.tshirt vika.tshirt) →
  (different_colors alyna.shorts vika.shorts) →
  (alyna = ⟨Color.Red, Color.Red⟩ ∧
   bohdan = ⟨Color.Red, Color.Blue⟩ ∧
   vika = ⟨Color.Blue, Color.Blue⟩ ∧
   grysha = ⟨Color.Red, Color.Blue⟩) :=
by sorry


end clothing_color_theorem_l1837_183726


namespace brainiac_teaser_ratio_l1837_183795

/-- Represents the number of brainiacs who like rebus teasers -/
def R : ℕ := 58

/-- Represents the number of brainiacs who like math teasers -/
def M : ℕ := 38

/-- The total number of brainiacs surveyed -/
def total : ℕ := 100

/-- The number of brainiacs who like both rebus and math teasers -/
def both : ℕ := 18

/-- The number of brainiacs who like neither rebus nor math teasers -/
def neither : ℕ := 4

/-- The number of brainiacs who like math teasers but not rebus teasers -/
def mathOnly : ℕ := 20

theorem brainiac_teaser_ratio :
  R = 58 ∧ M = 38 ∧ 
  total = 100 ∧
  both = 18 ∧
  neither = 4 ∧
  mathOnly = 20 →
  R * 19 = M * 29 := by
  sorry

end brainiac_teaser_ratio_l1837_183795


namespace inscribed_box_radius_l1837_183797

/-- A rectangular box inscribed in a sphere -/
structure InscribedBox where
  a : ℝ
  b : ℝ
  c : ℝ
  s : ℝ

/-- Properties of the inscribed box -/
def InscribedBoxProperties (box : InscribedBox) : Prop :=
  (box.a + box.b + box.c = 40) ∧
  (box.a * box.b + box.b * box.c + box.c * box.a = 432) ∧
  (4 * box.s^2 = box.a^2 + box.b^2 + box.c^2)

theorem inscribed_box_radius (box : InscribedBox) 
  (h : InscribedBoxProperties box) : box.s = 2 * Real.sqrt 46 := by
  sorry

end inscribed_box_radius_l1837_183797
