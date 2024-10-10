import Mathlib

namespace inequality_proof_l3641_364108

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  let p := x + y + z
  let q := x*y + y*z + z*x
  let r := x*y*z
  (p^2 ≥ 3*q) ∧
  (p^3 ≥ 27*r) ∧
  (p*q ≥ 9*r) ∧
  (q^2 ≥ 3*p*r) ∧
  (p^2*q + 3*p*r ≥ 4*q^2) ∧
  (p^3 + 9*r ≥ 4*p*q) ∧
  (p*q^2 ≥ 2*p^2*r + 3*q*r) ∧
  (p*q^2 + 3*q*r ≥ 4*p^2*r) ∧
  (2*q^3 + 9*r^2 ≥ 7*p*q*r) ∧
  (p^4 + 4*q^2 + 6*p*r ≥ 5*p^2*q) := by
  sorry

end inequality_proof_l3641_364108


namespace sum_remainder_l3641_364145

theorem sum_remainder (f y : ℤ) (hf : f % 5 = 3) (hy : y % 5 = 4) : (f + y) % 5 = 2 := by
  sorry

end sum_remainder_l3641_364145


namespace backpack_price_l3641_364164

/-- The price of a backpack and three ring-binders, given price changes and total spent --/
theorem backpack_price (B : ℕ) : 
  (∃ (new_backpack_price new_binder_price : ℕ),
    -- Original price of each ring-binder
    20 = 20 ∧
    -- New backpack price is $5 more than original
    new_backpack_price = B + 5 ∧
    -- New ring-binder price is $2 less than original
    new_binder_price = 20 - 2 ∧
    -- Total spent is $109
    new_backpack_price + 3 * new_binder_price = 109) →
  -- Original backpack price was $50
  B = 50 := by
sorry

end backpack_price_l3641_364164


namespace pencils_purchased_l3641_364107

/-- The number of pens purchased -/
def num_pens : ℕ := 30

/-- The total cost of pens and pencils -/
def total_cost : ℚ := 630

/-- The average price of a pencil -/
def pencil_price : ℚ := 2

/-- The average price of a pen -/
def pen_price : ℚ := 16

/-- The number of pencils purchased -/
def num_pencils : ℕ := 75

theorem pencils_purchased : 
  (num_pens : ℚ) * pen_price + (num_pencils : ℚ) * pencil_price = total_cost := by
  sorry

end pencils_purchased_l3641_364107


namespace road_trip_gas_cost_jennas_road_trip_cost_l3641_364162

/-- Calculates the cost of a road trip given driving times, speeds, and gas efficiency --/
theorem road_trip_gas_cost 
  (time1 : ℝ) (speed1 : ℝ) (time2 : ℝ) (speed2 : ℝ) 
  (gas_efficiency : ℝ) (gas_price : ℝ) : ℝ :=
  let distance1 := time1 * speed1
  let distance2 := time2 * speed2
  let total_distance := distance1 + distance2
  let gas_used := total_distance / gas_efficiency
  let total_cost := gas_used * gas_price
  total_cost

/-- Proves that Jenna's road trip gas cost is $18 --/
theorem jennas_road_trip_cost : 
  road_trip_gas_cost 2 60 3 50 30 2 = 18 := by
  sorry

end road_trip_gas_cost_jennas_road_trip_cost_l3641_364162


namespace max_product_sum_200_l3641_364176

theorem max_product_sum_200 : 
  ∀ x y : ℤ, x + y = 200 → x * y ≤ 10000 :=
by
  sorry

end max_product_sum_200_l3641_364176


namespace smallest_number_of_oranges_l3641_364182

/-- Represents the number of oranges in a container --/
def container_capacity : ℕ := 15

/-- Represents the number of containers that are not full --/
def short_containers : ℕ := 3

/-- Represents the number of oranges missing from each short container --/
def missing_oranges : ℕ := 2

/-- Represents the minimum number of oranges --/
def min_oranges : ℕ := 201

theorem smallest_number_of_oranges (n : ℕ) : 
  n * container_capacity - short_containers * missing_oranges > min_oranges →
  ∃ (m : ℕ), m * container_capacity - short_containers * missing_oranges > min_oranges ∧
             m * container_capacity - short_containers * missing_oranges ≤ 
             n * container_capacity - short_containers * missing_oranges →
  n * container_capacity - short_containers * missing_oranges ≥ 204 :=
by sorry

#check smallest_number_of_oranges

end smallest_number_of_oranges_l3641_364182


namespace sum_mod_nine_l3641_364128

theorem sum_mod_nine : (88135 + 88136 + 88137 + 88138 + 88139 + 88140) % 9 = 3 := by
  sorry

end sum_mod_nine_l3641_364128


namespace dinner_tasks_is_four_l3641_364135

/-- Represents Trey's chore list for Sunday -/
structure ChoreList where
  clean_house_tasks : Nat
  shower_tasks : Nat
  dinner_tasks : Nat
  time_per_task : Nat
  total_time : Nat

/-- Calculates the number of dinner tasks given the chore list -/
def calculate_dinner_tasks (chores : ChoreList) : Nat :=
  (chores.total_time - (chores.clean_house_tasks + chores.shower_tasks) * chores.time_per_task) / chores.time_per_task

/-- Theorem stating that the number of dinner tasks is 4 -/
theorem dinner_tasks_is_four (chores : ChoreList) 
  (h1 : chores.clean_house_tasks = 7)
  (h2 : chores.shower_tasks = 1)
  (h3 : chores.time_per_task = 10)
  (h4 : chores.total_time = 120) :
  calculate_dinner_tasks chores = 4 := by
  sorry

#eval calculate_dinner_tasks { clean_house_tasks := 7, shower_tasks := 1, dinner_tasks := 0, time_per_task := 10, total_time := 120 }

end dinner_tasks_is_four_l3641_364135


namespace rose_price_calculation_l3641_364100

/-- Calculates the price per rose given the initial number of roses, 
    remaining roses, and total earnings -/
def price_per_rose (initial_roses : ℕ) (remaining_roses : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / (initial_roses - remaining_roses)

theorem rose_price_calculation (initial_roses remaining_roses total_earnings : ℕ) 
  (h1 : initial_roses = 9)
  (h2 : remaining_roses = 4)
  (h3 : total_earnings = 35) :
  price_per_rose initial_roses remaining_roses total_earnings = 7 := by
  sorry

end rose_price_calculation_l3641_364100


namespace range_of_positive_integers_in_consecutive_list_l3641_364178

def consecutive_integers (start : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => start + i)

def positive_integers (l : List ℤ) : List ℤ :=
  l.filter (λ x => x > 0)

def range (l : List ℤ) : ℤ :=
  l.maximum.getD 0 - l.minimum.getD 0

theorem range_of_positive_integers_in_consecutive_list :
  let D := consecutive_integers (-4) 12
  let positives := positive_integers D
  range positives = 6 := by sorry

end range_of_positive_integers_in_consecutive_list_l3641_364178


namespace exclusive_multiples_of_6_or_8_less_than_151_l3641_364113

def count_multiples (n m : ℕ) : ℕ := (n - 1) / m

def count_exclusive_multiples (upper bound1 bound2 : ℕ) : ℕ :=
  let lcm := Nat.lcm bound1 bound2
  (count_multiples upper bound1) + (count_multiples upper bound2) - 2 * (count_multiples upper lcm)

theorem exclusive_multiples_of_6_or_8_less_than_151 :
  count_exclusive_multiples 151 6 8 = 31 := by sorry

end exclusive_multiples_of_6_or_8_less_than_151_l3641_364113


namespace max_value_at_two_a_is_max_point_l3641_364106

-- Define the function f(x) = -x^3 + 12x
def f (x : ℝ) : ℝ := -x^3 + 12*x

-- State the theorem
theorem max_value_at_two : 
  ∀ x : ℝ, f x ≤ f 2 := by
sorry

-- Define a as the point where f reaches its maximum value
def a : ℝ := 2

-- State that a is indeed the point of maximum value
theorem a_is_max_point : 
  ∀ x : ℝ, f x ≤ f a := by
sorry

end max_value_at_two_a_is_max_point_l3641_364106


namespace translate_linear_function_l3641_364129

/-- A linear function in the Cartesian coordinate system. -/
def LinearFunction (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x + b

/-- Vertical translation of a function. -/
def VerticalTranslate (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x ↦ f x - k

theorem translate_linear_function :
  let f := LinearFunction 5 0
  let g := VerticalTranslate f 5
  ∀ x, g x = 5 * x - 5 := by sorry

end translate_linear_function_l3641_364129


namespace sum_of_multiples_l3641_364155

def smallest_three_digit_multiple_of_5 : ℕ := 100

def smallest_four_digit_multiple_of_7 : ℕ := 1001

theorem sum_of_multiples : 
  smallest_three_digit_multiple_of_5 + smallest_four_digit_multiple_of_7 = 1101 := by
  sorry

end sum_of_multiples_l3641_364155


namespace three_digit_perfect_cube_divisible_by_16_l3641_364196

theorem three_digit_perfect_cube_divisible_by_16 : 
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, n = m^3 ∧ 16 ∣ n :=
by sorry

end three_digit_perfect_cube_divisible_by_16_l3641_364196


namespace system_solution_existence_l3641_364168

theorem system_solution_existence (a b : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = a^2 ∧ |x| + |y| = |b|) ↔ |a| ≤ |b| ∧ |b| ≤ Real.sqrt 2 * |a| :=
sorry

end system_solution_existence_l3641_364168


namespace bobik_distance_l3641_364121

/-- The problem of Seryozha, Valera, and Bobik's movement --/
theorem bobik_distance (distance : ℝ) (speed_seryozha speed_valera speed_bobik : ℝ) :
  distance = 21 →
  speed_seryozha = 4 →
  speed_valera = 3 →
  speed_bobik = 11 →
  speed_bobik * (distance / (speed_seryozha + speed_valera)) = 33 :=
by sorry

end bobik_distance_l3641_364121


namespace diana_biking_time_l3641_364158

def total_distance : ℝ := 10
def initial_speed : ℝ := 3
def initial_duration : ℝ := 2
def tired_speed : ℝ := 1

theorem diana_biking_time : 
  let initial_distance := initial_speed * initial_duration
  let remaining_distance := total_distance - initial_distance
  let tired_duration := remaining_distance / tired_speed
  initial_duration + tired_duration = 6 := by sorry

end diana_biking_time_l3641_364158


namespace bobbys_blocks_l3641_364169

/-- The number of blocks Bobby's father gave him -/
def blocks_from_father (initial_blocks final_blocks : ℕ) : ℕ :=
  final_blocks - initial_blocks

/-- Proof that Bobby's father gave him 6 blocks -/
theorem bobbys_blocks :
  blocks_from_father 2 8 = 6 := by
  sorry

end bobbys_blocks_l3641_364169


namespace middle_circle_radius_l3641_364126

/-- A configuration of five circles tangent to each other and two parallel lines -/
structure CircleConfiguration where
  /-- The radii of the five circles, from smallest to largest -/
  radii : Fin 5 → ℝ
  /-- The radii are positive -/
  radii_pos : ∀ i, 0 < radii i
  /-- The radii are in ascending order -/
  radii_ascending : ∀ i j, i < j → radii i < radii j

/-- The theorem stating that if the smallest and largest radii are 8 and 18, 
    then the middle radius is 12 -/
theorem middle_circle_radius (c : CircleConfiguration)
    (h_smallest : c.radii 0 = 8)
    (h_largest : c.radii 4 = 18) :
    c.radii 2 = 12 := by
  sorry

end middle_circle_radius_l3641_364126


namespace education_allocation_l3641_364110

def town_budget : ℕ := 32000000

theorem education_allocation :
  let policing : ℕ := town_budget / 2
  let public_spaces : ℕ := 4000000
  let education : ℕ := town_budget - (policing + public_spaces)
  education = 12000000 := by sorry

end education_allocation_l3641_364110


namespace relationship_abc_l3641_364153

theorem relationship_abc (a b c : ℝ) : 
  a = (1/2)^(2/3) → b = (1/5)^(2/3) → c = (1/2)^(1/3) → b < a ∧ a < c := by
  sorry

end relationship_abc_l3641_364153


namespace arithmetic_sum_equals_square_l3641_364185

theorem arithmetic_sum_equals_square (n : ℕ) :
  let first_term := 1
  let last_term := 2*n + 3
  let num_terms := n + 2
  (num_terms * (first_term + last_term)) / 2 = (n + 2)^2 := by
sorry

end arithmetic_sum_equals_square_l3641_364185


namespace part_one_part_two_l3641_364192

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Part 1
theorem part_one (m : ℝ) (h1 : m > 0) 
  (h2 : Set.Icc (-2 : ℝ) 2 = {x | f (x + 1/2) ≤ 2*m + 1}) : 
  m = 3/2 := by sorry

-- Part 2
theorem part_two : 
  (∃ a : ℝ, ∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) ∧ 
  (∀ a : ℝ, (∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) → a ≥ 4) := by sorry

end part_one_part_two_l3641_364192


namespace number_difference_l3641_364123

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 125000)
  (b_div_100 : 100 ∣ b)
  (a_eq_b_div_100 : a = b / 100)
  (a_div_5 : 5 ∣ a)
  (b_div_5 : 5 ∣ b) :
  b - a = 122265 := by
sorry

end number_difference_l3641_364123


namespace inequality_range_l3641_364157

theorem inequality_range (m : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, x^2 - 4*x ≥ m) → m ≤ -3 := by
  sorry

end inequality_range_l3641_364157


namespace octagon_side_length_l3641_364124

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  w : Point
  x : Point
  y : Point
  z : Point

/-- Represents an octagon -/
structure Octagon where
  a : Point
  b : Point
  c : Point
  d : Point
  e : Point
  f : Point
  g : Point
  h : Point

def is_on_line (p q r : Point) : Prop := sorry

def is_equilateral (oct : Octagon) : Prop := sorry

def is_convex (oct : Octagon) : Prop := sorry

def side_length (oct : Octagon) : ℝ := sorry

theorem octagon_side_length 
  (rect : Rectangle)
  (oct : Octagon)
  (h1 : rect.z.x - rect.w.x = 10)
  (h2 : rect.y.y - rect.z.y = 8)
  (h3 : is_on_line rect.w oct.a rect.z)
  (h4 : is_on_line rect.w oct.b rect.z)
  (h5 : is_on_line rect.z oct.c rect.y)
  (h6 : is_on_line rect.z oct.d rect.y)
  (h7 : is_on_line rect.y oct.e rect.w)
  (h8 : is_on_line rect.y oct.f rect.w)
  (h9 : is_on_line rect.x oct.g rect.w)
  (h10 : is_on_line rect.x oct.h rect.w)
  (h11 : oct.a.x - rect.w.x = rect.z.x - oct.b.x)
  (h12 : oct.a.x - rect.w.x ≤ 5)
  (h13 : is_equilateral oct)
  (h14 : is_convex oct) :
  side_length oct = -9 + Real.sqrt 652 := by sorry

end octagon_side_length_l3641_364124


namespace arithmetic_progression_contains_10_start_l3641_364127

/-- An infinite increasing arithmetic progression of natural numbers contains a number starting with 10 -/
theorem arithmetic_progression_contains_10_start (a d : ℕ) (h : 0 < d) :
  ∃ k : ℕ, ∃ m : ℕ, (a + k * d) = 10 * 10^m + (a + k * d - 10 * 10^m) ∧ 
    10 * 10^m ≤ (a + k * d) ∧ (a + k * d) < 11 * 10^m := by
  sorry

end arithmetic_progression_contains_10_start_l3641_364127


namespace jamie_quiz_score_l3641_364159

def school_quiz (total_questions correct_answers incorrect_answers unanswered_questions : ℕ)
  (points_correct points_incorrect points_unanswered : ℚ) : Prop :=
  total_questions = correct_answers + incorrect_answers + unanswered_questions ∧
  (correct_answers : ℚ) * points_correct +
  (incorrect_answers : ℚ) * points_incorrect +
  (unanswered_questions : ℚ) * points_unanswered = 28

theorem jamie_quiz_score :
  school_quiz 30 16 10 4 2 (-1/2) (1/4) :=
by sorry

end jamie_quiz_score_l3641_364159


namespace unique_valid_number_l3641_364190

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a > b ∧ b > c ∧
    (100 * a + 10 * b + c) - (100 * c + 10 * b + a) = n

theorem unique_valid_number :
  ∃! n, is_valid_number n ∧ n = 495 :=
sorry

end unique_valid_number_l3641_364190


namespace sweet_potato_harvest_l3641_364195

theorem sweet_potato_harvest (sold_to_adams : ℕ) (sold_to_lenon : ℕ) (not_sold : ℕ) :
  sold_to_adams = 20 →
  sold_to_lenon = 15 →
  not_sold = 45 →
  sold_to_adams + sold_to_lenon + not_sold = 80 :=
by sorry

end sweet_potato_harvest_l3641_364195


namespace division_problem_l3641_364179

theorem division_problem : (120 : ℚ) / ((6 / 2) + 4) = 17 + 1/7 := by sorry

end division_problem_l3641_364179


namespace no_solution_implies_m_equals_one_l3641_364151

theorem no_solution_implies_m_equals_one (m : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → (x - 3) / (x - 2) ≠ m / (2 - x)) →
  m = 1 := by
  sorry

end no_solution_implies_m_equals_one_l3641_364151


namespace ammonium_nitrate_reaction_l3641_364120

-- Define the chemical species as types
structure ChemicalSpecies where
  name : String
  formula : String

-- Define the reaction
def reaction : List (ℕ × ChemicalSpecies) → List (ℕ × ChemicalSpecies) → Prop :=
  sorry

-- Define the chemical species involved
def nh4no3 : ChemicalSpecies := ⟨"Ammonium nitrate", "NH4NO3"⟩
def naoh : ChemicalSpecies := ⟨"Sodium hydroxide", "NaOH"⟩
def nano3 : ChemicalSpecies := ⟨"Sodium nitrate", "NaNO3"⟩
def nh3 : ChemicalSpecies := ⟨"Ammonia", "NH3"⟩
def h2o : ChemicalSpecies := ⟨"Water", "H2O"⟩

-- State the theorem
theorem ammonium_nitrate_reaction 
  (balanced_equation : reaction [(1, nh4no3), (1, naoh)] [(1, nano3), (1, nh3), (1, h2o)])
  (naoh_reacted : ℕ) (nano3_formed : ℕ) (nh3_formed : ℕ)
  (h1 : naoh_reacted = 3)
  (h2 : nano3_formed = 3)
  (h3 : nh3_formed = 3) :
  ∃ (nh4no3_required : ℕ) (h2o_formed : ℕ),
    nh4no3_required = 3 ∧ h2o_formed = 3 :=
  sorry

end ammonium_nitrate_reaction_l3641_364120


namespace gcf_36_45_l3641_364136

theorem gcf_36_45 : Nat.gcd 36 45 = 9 := by
  sorry

end gcf_36_45_l3641_364136


namespace room_width_calculation_l3641_364102

theorem room_width_calculation (room_length : ℝ) (carpet_width : ℝ) (carpet_cost_per_sqm : ℝ) (total_cost : ℝ) :
  room_length = 13 →
  carpet_width = 0.75 →
  carpet_cost_per_sqm = 12 →
  total_cost = 1872 →
  room_length * (total_cost / (room_length * carpet_cost_per_sqm)) = 12 :=
by
  sorry

end room_width_calculation_l3641_364102


namespace kids_on_other_days_l3641_364183

/-- 
Given that Julia played tag with some kids from Monday to Friday,
prove that the number of kids she played with on Monday, Thursday, and Friday combined
is equal to the total number of kids minus the number of kids on Tuesday and Wednesday.
-/
theorem kids_on_other_days 
  (total_kids : ℕ) 
  (tuesday_wednesday_kids : ℕ) 
  (h1 : total_kids = 75) 
  (h2 : tuesday_wednesday_kids = 36) : 
  total_kids - tuesday_wednesday_kids = 39 := by
sorry

end kids_on_other_days_l3641_364183


namespace smallest_m_inequality_l3641_364166

theorem smallest_m_inequality (a b c : ℤ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  ∃ (m : ℝ), (∀ (x y z : ℤ), x > 0 → y > 0 → z > 0 → x + y + z = 1 → 
    m * (x^3 + y^3 + z^3 : ℝ) ≥ 6 * (x^2 + y^2 + z^2 : ℝ) + 1) ∧ 
  m = 27 ∧
  ∀ (n : ℝ), (∀ (x y z : ℤ), x > 0 → y > 0 → z > 0 → x + y + z = 1 → 
    n * (x^3 + y^3 + z^3 : ℝ) ≥ 6 * (x^2 + y^2 + z^2 : ℝ) + 1) → n ≥ 27 :=
by sorry

end smallest_m_inequality_l3641_364166


namespace geometric_sequence_common_ratio_l3641_364115

/-- A geometric sequence with a_3 = 2 and a_6 = 16 has a common ratio of 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 1))
  (h_a3 : a 3 = 2)
  (h_a6 : a 6 = 16) : 
  a 1 = 2 := by
sorry

end geometric_sequence_common_ratio_l3641_364115


namespace bryan_bookshelves_l3641_364175

/-- Given that Bryan has a total number of books and each bookshelf contains a fixed number of books,
    calculate the number of bookshelves he has. -/
def calculate_bookshelves (total_books : ℕ) (books_per_shelf : ℕ) : ℕ :=
  total_books / books_per_shelf

/-- Prove that Bryan has 19 bookshelves given the conditions. -/
theorem bryan_bookshelves :
  calculate_bookshelves 38 2 = 19 := by
  sorry

end bryan_bookshelves_l3641_364175


namespace common_ratio_is_two_l3641_364161

/-- An increasing geometric sequence with specific conditions -/
structure IncreasingGeometricSequence where
  a : ℕ → ℝ
  is_increasing : ∀ n, a n < a (n + 1)
  is_geometric : ∃ q : ℝ, q > 1 ∧ ∀ n, a (n + 1) = a n * q
  a2_eq_2 : a 2 = 2
  a4_minus_a3_eq_4 : a 4 - a 3 = 4

/-- The common ratio of the increasing geometric sequence is 2 -/
theorem common_ratio_is_two (seq : IncreasingGeometricSequence) : 
  ∃ q : ℝ, (∀ n, seq.a (n + 1) = seq.a n * q) ∧ q = 2 :=
sorry

end common_ratio_is_two_l3641_364161


namespace infinitely_many_consecutive_squares_sum_square_infinitely_many_solutions_for_non_square_l3641_364141

-- Part a
def ConsecutiveSquaresSum (n : ℕ+) (x : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun k => (x + k) ^ 2)

def IsConsecutiveSquaresSumSquare (n : ℕ+) : Prop :=
  ∃ x k : ℕ, ConsecutiveSquaresSum n x = k ^ 2

theorem infinitely_many_consecutive_squares_sum_square :
  Set.Infinite {n : ℕ+ | IsConsecutiveSquaresSumSquare n} :=
sorry

-- Part b
theorem infinitely_many_solutions_for_non_square (n : ℕ+) (h : ¬ ∃ m : ℕ, n = m ^ 2) :
  (∃ x k : ℕ, ConsecutiveSquaresSum n x = k ^ 2) →
  Set.Infinite {y : ℕ | ∃ k : ℕ, ConsecutiveSquaresSum n y = k ^ 2} :=
sorry

end infinitely_many_consecutive_squares_sum_square_infinitely_many_solutions_for_non_square_l3641_364141


namespace sin_cos_inequality_l3641_364143

-- Define an odd function that is monotonically decreasing on [-1, 0]
def is_odd_and_decreasing (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ 0 → f x > f y)

-- Define acute angles of a triangle
def is_acute_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < Real.pi / 2

-- Theorem statement
theorem sin_cos_inequality 
  (f : ℝ → ℝ) 
  (A B : ℝ) 
  (h_f : is_odd_and_decreasing f) 
  (h_A : is_acute_angle A) 
  (h_B : is_acute_angle B) : 
  f (Real.sin A) < f (Real.cos B) :=
sorry

end sin_cos_inequality_l3641_364143


namespace debt_average_payment_l3641_364134

/-- Proves that the average payment for a debt with specific conditions is $465 --/
theorem debt_average_payment 
  (total_installments : ℕ) 
  (first_payment_count : ℕ) 
  (first_payment_amount : ℚ) 
  (additional_amount : ℚ) : 
  total_installments = 52 →
  first_payment_count = 8 →
  first_payment_amount = 410 →
  additional_amount = 65 →
  let remaining_payment_count := total_installments - first_payment_count
  let remaining_payment_amount := first_payment_amount + additional_amount
  let total_amount := 
    (first_payment_count * first_payment_amount) + 
    (remaining_payment_count * remaining_payment_amount)
  total_amount / total_installments = 465 := by
  sorry

end debt_average_payment_l3641_364134


namespace floor_of_e_equals_two_l3641_364181

noncomputable def e : ℝ := Real.exp 1

theorem floor_of_e_equals_two : ⌊e⌋ = 2 := by
  sorry

end floor_of_e_equals_two_l3641_364181


namespace marathon_positions_l3641_364193

/-- Represents a marathon with participants -/
structure Marathon where
  total_participants : ℕ
  john_from_right : ℕ
  john_from_left : ℕ
  mike_ahead : ℕ

/-- Theorem about the marathon positions -/
theorem marathon_positions (m : Marathon) 
  (h1 : m.john_from_right = 28)
  (h2 : m.john_from_left = 42)
  (h3 : m.mike_ahead = 10) :
  m.total_participants = 69 ∧ 
  m.john_from_left - m.mike_ahead = 32 ∧ 
  m.john_from_right - m.mike_ahead = 18 := by
  sorry


end marathon_positions_l3641_364193


namespace triangle_properties_l3641_364111

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  a + b + c = 10 →
  Real.sin B + Real.sin C = 4 * Real.sin A →
  b * c = 16 →
  (a = 2 ∧ Real.cos A = 7/8) := by
sorry

end triangle_properties_l3641_364111


namespace triangle_property_l3641_364171

theorem triangle_property (a b c : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_condition : a^3 - b^3 = a^2*b - a*b^2 + a*c^2 - b*c^2) : 
  a = b ∨ a^2 + b^2 = c^2 := by
sorry

end triangle_property_l3641_364171


namespace prize_orders_count_l3641_364138

/-- Represents the number of bowlers in the tournament -/
def num_bowlers : ℕ := 6

/-- Represents the number of games played in the tournament -/
def num_games : ℕ := 5

/-- Calculates the number of possible outcomes for a single game -/
def outcomes_per_game : ℕ := 2

/-- Calculates the total number of possible prize orders -/
def total_prize_orders : ℕ := outcomes_per_game ^ num_games

/-- Theorem stating that the total number of possible prize orders is 32 -/
theorem prize_orders_count : total_prize_orders = 32 := by
  sorry


end prize_orders_count_l3641_364138


namespace balance_difference_approx_l3641_364133

def angela_deposit : ℝ := 9000
def bob_deposit : ℝ := 11000
def angela_rate : ℝ := 0.08
def bob_rate : ℝ := 0.09
def years : ℕ := 25

def angela_balance : ℝ := angela_deposit * (1 + angela_rate) ^ years
def bob_balance : ℝ := bob_deposit * (1 + bob_rate * years)

theorem balance_difference_approx :
  ‖angela_balance - bob_balance - 25890‖ < 1 := by sorry

end balance_difference_approx_l3641_364133


namespace hemisphere_container_volume_l3641_364194

/-- Given a total volume of water and the number of hemisphere containers needed,
    calculate the volume of each hemisphere container. -/
theorem hemisphere_container_volume
  (total_volume : ℝ)
  (num_containers : ℕ)
  (h_total_volume : total_volume = 10976)
  (h_num_containers : num_containers = 2744) :
  total_volume / num_containers = 4 := by
  sorry

end hemisphere_container_volume_l3641_364194


namespace imaginary_part_of_i_over_one_plus_i_l3641_364118

theorem imaginary_part_of_i_over_one_plus_i : Complex.im (Complex.I / (1 + Complex.I)) = 1 / 2 := by
  sorry

end imaginary_part_of_i_over_one_plus_i_l3641_364118


namespace complement_intersection_theorem_l3641_364165

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 5}
def B : Set Nat := {2, 3, 5}

theorem complement_intersection_theorem :
  (U \ B) ∩ A = {1} := by sorry

end complement_intersection_theorem_l3641_364165


namespace arithmetic_equality_l3641_364139

theorem arithmetic_equality : 239 - 27 + 45 + 33 - 11 = 279 := by
  sorry

end arithmetic_equality_l3641_364139


namespace fraction_equality_proof_l3641_364146

theorem fraction_equality_proof (a b z : ℕ+) (h : a * b = z^2 + 1) :
  ∃ (x y : ℕ+), (a : ℚ) / b = ((x^2 : ℚ) + 1) / ((y^2 : ℚ) + 1) := by
  sorry

end fraction_equality_proof_l3641_364146


namespace matches_played_is_ten_l3641_364198

/-- The number of matches a player has played, given their current average and the effect of a future match on that average. -/
def matches_played (current_average : ℚ) (future_score : ℚ) (average_increase : ℚ) : ℕ :=
  let n : ℕ := sorry
  n

/-- Theorem stating that the number of matches played is 10 under the given conditions. -/
theorem matches_played_is_ten :
  matches_played 32 76 4 = 10 := by sorry

end matches_played_is_ten_l3641_364198


namespace box_width_proof_l3641_364137

/-- Given a rectangular box with length 12 cm, height 6 cm, and volume 1152 cubic cm,
    prove that the width of the box is 16 cm. -/
theorem box_width_proof (length : ℝ) (height : ℝ) (volume : ℝ) (width : ℝ) 
    (h1 : length = 12)
    (h2 : height = 6)
    (h3 : volume = 1152)
    (h4 : volume = length * width * height) :
  width = 16 := by
  sorry

end box_width_proof_l3641_364137


namespace proportion_with_one_half_one_third_l3641_364150

def forms_proportion (a b c d : ℚ) : Prop := a / b = c / d

theorem proportion_with_one_half_one_third :
  forms_proportion (1/2) (1/3) 3 2 ∧
  ¬forms_proportion (1/2) (1/3) 5 4 ∧
  ¬forms_proportion (1/2) (1/3) (1/3) (1/4) ∧
  ¬forms_proportion (1/2) (1/3) (1/3) (1/2) :=
by sorry

end proportion_with_one_half_one_third_l3641_364150


namespace oreo_shop_combinations_l3641_364130

/-- Represents the number of flavors for each product type -/
structure Flavors where
  oreos : Nat
  milk : Nat
  cookies : Nat

/-- Represents the purchasing rules for Alpha and Gamma -/
structure PurchaseRules where
  alpha_max_items : Nat
  alpha_allows_repeats : Bool
  gamma_allowed_products : List String
  gamma_allows_repeats : Bool

/-- Calculates the number of ways to purchase items given the rules and flavors -/
def purchase_combinations (flavors : Flavors) (rules : PurchaseRules) (total_items : Nat) : Nat :=
  sorry

/-- The main theorem stating the number of purchase combinations -/
theorem oreo_shop_combinations :
  let flavors := Flavors.mk 5 3 2
  let rules := PurchaseRules.mk 2 false ["oreos", "cookies"] true
  purchase_combinations flavors rules 4 = 2100 := by
  sorry

end oreo_shop_combinations_l3641_364130


namespace intersection_M_N_l3641_364122

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N : Set ℝ := {-2, -1, 1, 2}

theorem intersection_M_N : M ∩ N = {1, 2} := by
  sorry

end intersection_M_N_l3641_364122


namespace complex_roots_on_circle_l3641_364189

theorem complex_roots_on_circle :
  ∀ (z : ℂ), (z + 2)^6 = 64 * z^6 →
  Complex.abs (z + 2/3) = 2/3 := by sorry

end complex_roots_on_circle_l3641_364189


namespace odd_periodic_function_property_l3641_364132

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic_2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

theorem odd_periodic_function_property (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : is_odd f) (h_periodic : is_periodic_2 f) (h_value : f (1 + a) = 1) :
  f (1 - a) = -1 := by
  sorry

end odd_periodic_function_property_l3641_364132


namespace system_of_equations_solution_l3641_364180

theorem system_of_equations_solution :
  ∀ s t : ℝ,
  (11 * s + 7 * t = 240) →
  (s = (1/2) * t + 3) →
  (t = 414/25 ∧ s = 11.28) :=
by
  sorry

end system_of_equations_solution_l3641_364180


namespace number_operation_result_l3641_364105

theorem number_operation_result (x : ℝ) : x + 7 = 27 → ((x / 5) + 5) * 7 = 63 := by
  sorry

end number_operation_result_l3641_364105


namespace polygon_with_two_diagonals_has_five_sides_l3641_364125

/-- A polygon is a closed plane figure with straight sides. -/
structure Polygon where
  sides : ℕ
  sides_pos : sides > 0

/-- The number of diagonals from any vertex in a polygon. -/
def diagonals_from_vertex (p : Polygon) : ℕ := p.sides - 3

/-- Theorem: A polygon with 2 diagonals from any vertex has 5 sides. -/
theorem polygon_with_two_diagonals_has_five_sides (p : Polygon) 
  (h : diagonals_from_vertex p = 2) : p.sides = 5 := by
  sorry

#check polygon_with_two_diagonals_has_five_sides

end polygon_with_two_diagonals_has_five_sides_l3641_364125


namespace company_size_l3641_364154

/-- Represents the number of employees in a company -/
structure Company where
  total : ℕ
  senior : ℕ
  sample_size : ℕ
  sample_senior : ℕ

/-- Given a company with 15 senior-titled employees and a stratified sample of 30 employees
    containing 3 senior-titled employees, the total number of employees is 150 -/
theorem company_size (c : Company)
  (h1 : c.senior = 15)
  (h2 : c.sample_size = 30)
  (h3 : c.sample_senior = 3)
  : c.total = 150 := by
  sorry

end company_size_l3641_364154


namespace calen_current_pencils_l3641_364184

-- Define the number of pencils for each person
def candy_pencils : ℕ := 9
def caleb_pencils : ℕ := 2 * candy_pencils - 3
def calen_original_pencils : ℕ := caleb_pencils + 5
def calen_lost_pencils : ℕ := 10

-- Theorem to prove
theorem calen_current_pencils :
  calen_original_pencils - calen_lost_pencils = 10 := by
  sorry

end calen_current_pencils_l3641_364184


namespace projection_property_l3641_364187

/-- A projection that takes [4, 4] to [60/13, 12/13] -/
def projection (v : ℝ × ℝ) : ℝ × ℝ :=
  sorry

theorem projection_property : 
  projection (4, 4) = (60/13, 12/13) ∧ 
  projection (-2, 2) = (-20/13, -4/13) := by
  sorry

end projection_property_l3641_364187


namespace least_addition_for_divisibility_l3641_364114

theorem least_addition_for_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬(9 ∣ (51234 + m))) ∧ (9 ∣ (51234 + n)) → n = 3 := by
  sorry

end least_addition_for_divisibility_l3641_364114


namespace row_col_product_equality_l3641_364167

theorem row_col_product_equality 
  (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) 
  (h_row_col_sum : 
    a₁ + a₂ + a₃ = b₁ + b₂ + b₃ ∧ 
    b₁ + b₂ + b₃ = c₁ + c₂ + c₃ ∧ 
    c₁ + c₂ + c₃ = a₁ + b₁ + c₁ ∧ 
    a₁ + b₁ + c₁ = a₂ + b₂ + c₂ ∧ 
    a₂ + b₂ + c₂ = a₃ + b₃ + c₃) : 
  a₁*b₁*c₁ + a₂*b₂*c₂ + a₃*b₃*c₃ = a₁*a₂*a₃ + b₁*b₂*b₃ + c₁*c₂*c₃ :=
by
  sorry

end row_col_product_equality_l3641_364167


namespace roller_coaster_probability_l3641_364163

/-- The number of cars in the roller coaster -/
def num_cars : ℕ := 5

/-- The number of times the passenger rides the roller coaster -/
def num_rides : ℕ := 5

/-- The probability of riding in a specific car on a single ride -/
def prob_single_car : ℚ := 1 / num_cars

/-- The probability of riding in each of the cars over the given number of rides -/
def prob_all_cars : ℚ := (num_cars.factorial : ℚ) / num_cars ^ num_rides

theorem roller_coaster_probability :
  prob_all_cars = 24 / 625 := by
  sorry

end roller_coaster_probability_l3641_364163


namespace triangle_sum_formula_l3641_364148

def triangleSum (n : ℕ) : ℕ := 8 * 2^n - 4

theorem triangle_sum_formula (n : ℕ) : 
  n ≥ 1 → 
  (∀ k, k ≥ 2 → triangleSum k = 2 * triangleSum (k-1) + 4) → 
  triangleSum 1 = 4 → 
  triangleSum n = 8 * 2^n - 4 := by
  sorry

end triangle_sum_formula_l3641_364148


namespace unique_m_solution_l3641_364149

def S (n : ℕ) : ℕ := n^2

def a (n : ℕ) : ℕ := 2*n - 1

theorem unique_m_solution :
  ∃! m : ℕ+, 
    (∀ n : ℕ, S n = n^2) ∧ 
    (S m = (a m.val + a (m.val + 1)) / 2) :=
by
  sorry

end unique_m_solution_l3641_364149


namespace prime_factors_of_n_smallest_prime_factors_difference_l3641_364174

def n : ℕ := 172561

-- Define a function to check if a number is prime
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

-- Define the prime factors of n
theorem prime_factors_of_n :
  ∃ (p q r : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧
  p < q ∧ q < r ∧ n = p * q * r :=
sorry

-- Prove that the positive difference between the two smallest prime factors is 26
theorem smallest_prime_factors_difference :
  ∃ (p q r : ℕ), is_prime p ∧ is_prime q ∧ is_prime r ∧
  p < q ∧ q < r ∧ n = p * q * r ∧ q - p = 26 :=
sorry

end prime_factors_of_n_smallest_prime_factors_difference_l3641_364174


namespace arithmetic_calculations_l3641_364197

theorem arithmetic_calculations :
  ((294.4 - 19.2 * 6) / (6 + 8) = 12.8) ∧
  (12.5 * 0.4 * 8 * 2.5 = 100) ∧
  (333 * 334 + 999 * 222 = 333000) ∧
  (999 + 99.9 + 9.99 + 0.999 = 1109.889) := by
  sorry

end arithmetic_calculations_l3641_364197


namespace x_plus_y_equals_30_l3641_364142

theorem x_plus_y_equals_30 (x y : ℝ) 
  (h1 : |x| - x + y = 6) 
  (h2 : x + |y| + y = 8) : 
  x + y = 30 := by
sorry

end x_plus_y_equals_30_l3641_364142


namespace train_crossing_time_l3641_364186

/-- A train crosses a platform and an electric pole -/
theorem train_crossing_time (train_speed : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ) :
  train_speed = 10 →
  platform_length = 320 →
  platform_crossing_time = 44 →
  (platform_length + train_speed * platform_crossing_time) / train_speed = 12 :=
by
  sorry

end train_crossing_time_l3641_364186


namespace fibonacci_divisibility_l3641_364140

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_divisibility (m n : ℕ) (h1 : m ≥ 1) (h2 : n > 1) :
  ∃ k : ℕ, fib (m * n - 1) - (fib (n - 1))^m = k * (fib n)^2 := by
  sorry

end fibonacci_divisibility_l3641_364140


namespace average_pqr_l3641_364199

theorem average_pqr (p q r : ℝ) (h : (5 / 4) * (p + q + r) = 15) :
  (p + q + r) / 3 = 4 := by
sorry

end average_pqr_l3641_364199


namespace gcd_of_three_numbers_l3641_364116

theorem gcd_of_three_numbers (A B C : ℕ+) 
  (h_lcm : Nat.lcm A.val (Nat.lcm B.val C.val) = 1540)
  (h_prod : A.val * B.val * C.val = 1230000) :
  Nat.gcd A.val (Nat.gcd B.val C.val) = 20 := by
  sorry

end gcd_of_three_numbers_l3641_364116


namespace stating_min_weighings_to_find_lighter_ball_l3641_364188

/-- Represents the number of balls -/
def num_balls : ℕ := 9

/-- Represents the number of heavier balls -/
def num_heavy : ℕ := 8

/-- Represents the weight of the heavier balls in grams -/
def heavy_weight : ℕ := 10

/-- Represents the weight of the lighter ball in grams -/
def light_weight : ℕ := 9

/-- Represents the availability of a balance scale -/
def has_balance_scale : Prop := True

/-- 
Theorem stating that the minimum number of weighings required to find the lighter ball is 2
given the conditions of the problem.
-/
theorem min_weighings_to_find_lighter_ball :
  ∀ (balls : Fin num_balls → ℕ),
  (∃ (i : Fin num_balls), balls i = light_weight) ∧
  (∀ (i : Fin num_balls), balls i = light_weight ∨ balls i = heavy_weight) ∧
  has_balance_scale →
  (∃ (n : ℕ), n = 2 ∧ 
    ∀ (m : ℕ), (∃ (strategy : ℕ → ℕ → Bool), 
      (∀ (i : Fin num_balls), balls i = light_weight → 
        ∃ (k : Fin m), strategy k (balls i) = true) ∧
      (∀ (i j : Fin num_balls), i ≠ j → balls i ≠ balls j → 
        ∃ (k : Fin m), strategy k (balls i) ≠ strategy k (balls j))) → 
    m ≥ n) :=
sorry

end stating_min_weighings_to_find_lighter_ball_l3641_364188


namespace max_area_semicircle_l3641_364177

/-- A semicircle with diameter AB and radius R -/
structure Semicircle where
  R : ℝ
  A : Point
  B : Point

/-- Points C and D on the semicircle -/
structure PointsOnSemicircle (S : Semicircle) where
  C : Point
  D : Point

/-- The area of quadrilateral ACDB -/
def area (S : Semicircle) (P : PointsOnSemicircle S) : ℝ :=
  sorry

/-- C and D divide the semicircle into three equal parts -/
def equalParts (S : Semicircle) (P : PointsOnSemicircle S) : Prop :=
  sorry

theorem max_area_semicircle (S : Semicircle) :
  ∃ (P : PointsOnSemicircle S),
    equalParts S P ∧
    ∀ (Q : PointsOnSemicircle S), area S Q ≤ area S P ∧
    area S P = (3 * Real.sqrt 3 / 4) * S.R^2 :=
  sorry

end max_area_semicircle_l3641_364177


namespace unique_c_value_l3641_364104

theorem unique_c_value (c : ℝ) : c ≠ 0 ∧
  (∃! (b₁ b₂ b₃ : ℝ), b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0 ∧ b₁ ≠ b₂ ∧ b₂ ≠ b₃ ∧ b₁ ≠ b₃ ∧
    (∀ x : ℝ, x^2 + 2*(b₁ + 1/b₁)*x + c = 0 → 
      ∃! y : ℝ, y^2 + 2*(b₁ + 1/b₁)*y + c = 0) ∧
    (∀ x : ℝ, x^2 + 2*(b₂ + 1/b₂)*x + c = 0 → 
      ∃! y : ℝ, y^2 + 2*(b₂ + 1/b₂)*y + c = 0) ∧
    (∀ x : ℝ, x^2 + 2*(b₃ + 1/b₃)*x + c = 0 → 
      ∃! y : ℝ, y^2 + 2*(b₃ + 1/b₃)*y + c = 0)) →
  c = 4 :=
by sorry

end unique_c_value_l3641_364104


namespace abnormal_segregation_in_secondary_spermatocyte_l3641_364117

-- Define the alleles
inductive Allele
| E  -- normal eye
| e  -- eyeless

-- Define a genotype as a list of alleles
def Genotype := List Allele

-- Define the parents' genotypes
def male_parent : Genotype := [Allele.E, Allele.e]
def female_parent : Genotype := [Allele.e, Allele.e]

-- Define the offspring's genotype
def offspring : Genotype := [Allele.E, Allele.E, Allele.e]

-- Define the possible cell types where segregation could have occurred abnormally
inductive CellType
| PrimarySpermatocyte
| PrimaryOocyte
| SecondarySpermatocyte
| SecondaryOocyte

-- Define the property of no crossing over
def no_crossing_over : Prop := sorry

-- Define the dominance of E over e
def E_dominant_over_e : Prop := sorry

-- Theorem statement
theorem abnormal_segregation_in_secondary_spermatocyte :
  E_dominant_over_e →
  no_crossing_over →
  (∃ (abnormal_cell : CellType), 
    abnormal_cell = CellType.SecondarySpermatocyte ∧
    (∀ (other_cell : CellType), 
      other_cell ≠ CellType.SecondarySpermatocyte → 
      ¬(offspring = [Allele.E, Allele.E, Allele.e]))) :=
sorry

end abnormal_segregation_in_secondary_spermatocyte_l3641_364117


namespace tangent_line_at_point_l3641_364119

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 3*x + 1

-- Define the point
def p : ℝ × ℝ := (0, 1)

-- Define the slope of the tangent line
def m : ℝ := 3

-- Define the equation of the tangent line
def tangent_line (x y : ℝ) : Prop := 3*x - y + 1 = 0

-- Theorem statement
theorem tangent_line_at_point :
  tangent_line p.1 p.2 ∧
  ∀ x y, tangent_line x y ↔ y - f p.1 = m * (x - p.1) :=
sorry

end tangent_line_at_point_l3641_364119


namespace raja_monthly_savings_l3641_364191

/-- Raja's monthly savings calculation --/
theorem raja_monthly_savings :
  let monthly_income : ℝ := 24999.999999999993
  let household_percentage : ℝ := 0.60
  let clothes_percentage : ℝ := 0.10
  let medicines_percentage : ℝ := 0.10
  let total_spent_percentage : ℝ := household_percentage + clothes_percentage + medicines_percentage
  let savings_percentage : ℝ := 1 - total_spent_percentage
  let savings : ℝ := savings_percentage * monthly_income
  ⌊savings⌋ = 5000 := by sorry

end raja_monthly_savings_l3641_364191


namespace sqrt_sum_equals_five_l3641_364109

theorem sqrt_sum_equals_five (x y : ℝ) (h : y = Real.sqrt (x - 9) - Real.sqrt (9 - x) + 4) :
  Real.sqrt x + Real.sqrt y = 5 := by
  sorry

end sqrt_sum_equals_five_l3641_364109


namespace units_digit_of_sum_factorial_and_square_10_l3641_364103

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def sum_factorial_and_square (n : ℕ) : ℕ := 
  (List.range n).foldl (λ acc i => acc + factorial (i + 1) + (i + 1)^2) 0

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_sum_factorial_and_square_10 : 
  units_digit (sum_factorial_and_square 10) = 8 := by sorry

end units_digit_of_sum_factorial_and_square_10_l3641_364103


namespace sequence_ratio_proof_l3641_364170

theorem sequence_ratio_proof (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a n > 0) →
  (∀ n, (a (n + 1))^2 = (a n) * (a (n + 2))) →
  (S 3 = 13) →
  (a 1 = 1) →
  ((a 3 + a 4) / (a 1 + a 2) = 9) :=
by
  sorry

end sequence_ratio_proof_l3641_364170


namespace inscribed_circle_radius_ratio_l3641_364173

/-- An equilateral triangle with an inscribed circle -/
structure EquilateralTriangleWithInscribedCircle where
  /-- The side length of the equilateral triangle -/
  side_length : ℝ
  /-- The radius of the inscribed circle -/
  circle_radius : ℝ
  /-- The points of tangency are on the sides of the triangle -/
  tangency_points_on_sides : True

/-- The ratio of the inscribed circle's radius to the triangle's side length is 1/16 -/
theorem inscribed_circle_radius_ratio 
  (triangle : EquilateralTriangleWithInscribedCircle) : 
  triangle.circle_radius / triangle.side_length = 1/16 := by
  sorry

end inscribed_circle_radius_ratio_l3641_364173


namespace cases_in_2007_l3641_364160

/-- Calculates the number of disease cases in a given year, assuming a linear decrease --/
def diseaseCases (initialYear initialCases finalYear finalCases targetYear : ℕ) : ℕ :=
  let totalYears := finalYear - initialYear
  let totalDecrease := initialCases - finalCases
  let annualDecrease := totalDecrease / totalYears
  let targetYearsSinceInitial := targetYear - initialYear
  initialCases - (annualDecrease * targetYearsSinceInitial)

/-- The number of disease cases in 2007, given the conditions --/
theorem cases_in_2007 :
  diseaseCases 1980 300000 2016 1000 2007 = 75738 := by
  sorry

#eval diseaseCases 1980 300000 2016 1000 2007

end cases_in_2007_l3641_364160


namespace polynomial_evaluation_l3641_364112

theorem polynomial_evaluation (x : ℝ) (h1 : x > 0) (h2 : x^2 + 3*x - 9 = 0) :
  x^3 + 3*x^2 - 9*x - 5 = 22 := by
  sorry

end polynomial_evaluation_l3641_364112


namespace monotonic_function_a_range_l3641_364144

/-- The function f(x) = x^2/2 - a*ln(x) is monotonic on [1,2] if and only if a ∈ (0, 1] ∪ [4, +∞) -/
theorem monotonic_function_a_range (a : ℝ) (h : a > 0) :
  (∀ x ∈ Set.Icc 1 2, Monotone (fun x => x^2 / 2 - a * Real.log x)) ↔ 
  a ∈ Set.Ioo 0 1 ∪ Set.Iic 1 ∪ Set.Ici 4 := by
  sorry


end monotonic_function_a_range_l3641_364144


namespace function_derivative_value_l3641_364147

/-- Given a function f(x) = ax³ + 3x² + 2, prove that if f'(-1) = 4, then a = 10/3 -/
theorem function_derivative_value (a : ℝ) : 
  let f := λ x : ℝ => a * x^3 + 3 * x^2 + 2
  let f' := λ x : ℝ => 3 * a * x^2 + 6 * x
  f' (-1) = 4 → a = 10/3 := by sorry

end function_derivative_value_l3641_364147


namespace wall_building_time_l3641_364156

/-- Given that 8 persons can build a 140m wall in 8 days, this theorem calculates
    the number of days it takes 30 persons to build a similar 100m wall. -/
theorem wall_building_time (persons1 persons2 : ℕ) (length1 length2 : ℝ) (days1 : ℝ) : 
  persons1 = 8 →
  persons2 = 30 →
  length1 = 140 →
  length2 = 100 →
  days1 = 8 →
  ∃ days2 : ℝ, days2 = (persons1 * days1 * length2) / (persons2 * length1) :=
by sorry

end wall_building_time_l3641_364156


namespace movie_ticket_cost_proof_l3641_364131

def movie_ticket_cost (total_money : ℚ) (change : ℚ) (num_sisters : ℕ) : ℚ :=
  (total_money - change) / num_sisters

theorem movie_ticket_cost_proof (total_money : ℚ) (change : ℚ) (num_sisters : ℕ) 
  (h1 : total_money = 25)
  (h2 : change = 9)
  (h3 : num_sisters = 2) :
  movie_ticket_cost total_money change num_sisters = 8 := by
  sorry

#eval movie_ticket_cost 25 9 2

end movie_ticket_cost_proof_l3641_364131


namespace line_circle_intersection_l3641_364101

/-- Given a line y = kx (k > 0) intersecting a circle (x-2)^2 + y^2 = 1 at two points A and B,
    where the distance AB = (2/5)√5, prove that k = 1/2 -/
theorem line_circle_intersection (k : ℝ) (h_k_pos : k > 0) : 
  (∃ A B : ℝ × ℝ, 
    (A.1 - 2)^2 + (k * A.1)^2 = 1 ∧ 
    (B.1 - 2)^2 + (k * B.1)^2 = 1 ∧ 
    (A.1 - B.1)^2 + (k * A.1 - k * B.1)^2 = (2/5)^2 * 5) → 
  k = 1/2 := by
sorry

end line_circle_intersection_l3641_364101


namespace average_length_of_writing_instruments_l3641_364172

theorem average_length_of_writing_instruments :
  let pen_length : ℝ := 20
  let pencil_length : ℝ := 16
  let number_of_instruments : ℕ := 2
  (pen_length + pencil_length) / number_of_instruments = 18 :=
by
  sorry

end average_length_of_writing_instruments_l3641_364172


namespace fruit_display_total_l3641_364152

/-- The number of bananas on the display -/
def num_bananas : ℕ := 5

/-- The number of oranges on the display -/
def num_oranges : ℕ := 2 * num_bananas

/-- The number of apples on the display -/
def num_apples : ℕ := 2 * num_oranges

/-- The total number of fruits on the display -/
def total_fruits : ℕ := num_bananas + num_oranges + num_apples

theorem fruit_display_total :
  total_fruits = 35 :=
by sorry

end fruit_display_total_l3641_364152
