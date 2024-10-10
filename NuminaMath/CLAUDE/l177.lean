import Mathlib

namespace door_crank_time_l177_17760

/-- Represents the time taken for various parts of the game show challenge -/
structure GameShowTimes where
  firstRun : Nat  -- Time for first run in seconds
  secondRun : Nat -- Time for second run in seconds
  totalTime : Nat -- Total time for the entire event in seconds

/-- Calculates the time taken to crank open the door -/
def timeToCrankDoor (times : GameShowTimes) : Nat :=
  times.totalTime - (times.firstRun + times.secondRun)

/-- Theorem stating that the time to crank open the door is 73 seconds -/
theorem door_crank_time (times : GameShowTimes) 
  (h1 : times.firstRun = 7 * 60 + 23)
  (h2 : times.secondRun = 5 * 60 + 58)
  (h3 : times.totalTime = 874) :
  timeToCrankDoor times = 73 := by
  sorry

#eval timeToCrankDoor { firstRun := 7 * 60 + 23, secondRun := 5 * 60 + 58, totalTime := 874 }

end door_crank_time_l177_17760


namespace identity_proof_l177_17763

theorem identity_proof (a b : ℝ) : a^4 + b^4 + (a+b)^4 = 2*(a^2 + a*b + b^2)^2 := by
  sorry

end identity_proof_l177_17763


namespace f_decreasing_l177_17712

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 12*x + b

-- State the theorem
theorem f_decreasing (b : ℝ) : 
  ∀ x ∈ Set.Icc (-2) 2, 
    ∀ y ∈ Set.Icc (-2) 2, 
      x < y → f b x > f b y :=
by
  sorry


end f_decreasing_l177_17712


namespace comparison_of_radicals_and_fractions_l177_17736

theorem comparison_of_radicals_and_fractions : 
  (2 * Real.sqrt 7 < 4 * Real.sqrt 2) ∧ ((Real.sqrt 5 - 1) / 2 > 0.5) := by
  sorry

end comparison_of_radicals_and_fractions_l177_17736


namespace bags_sold_thursday_l177_17721

/-- Calculates the number of bags sold on Thursday given the total stock and sales on other days --/
theorem bags_sold_thursday (total_stock : ℕ) (monday_sales tuesday_sales wednesday_sales friday_sales : ℕ)
  (h1 : total_stock = 600)
  (h2 : monday_sales = 25)
  (h3 : tuesday_sales = 70)
  (h4 : wednesday_sales = 100)
  (h5 : friday_sales = 145)
  (h6 : (total_stock : ℚ) * (25 : ℚ) / 100 = total_stock - (monday_sales + tuesday_sales + wednesday_sales + friday_sales + 110)) :
  110 = total_stock - (monday_sales + tuesday_sales + wednesday_sales + friday_sales + (total_stock : ℚ) * (25 : ℚ) / 100) :=
by sorry

end bags_sold_thursday_l177_17721


namespace parallelogram_base_l177_17786

/-- The area of a parallelogram -/
def area_parallelogram (base height : ℝ) : ℝ := base * height

/-- Theorem: Given a parallelogram with height 36 cm and area 1728 cm², its base is 48 cm -/
theorem parallelogram_base (height area : ℝ) (h1 : height = 36) (h2 : area = 1728) :
  ∃ base : ℝ, area_parallelogram base height = area ∧ base = 48 := by
  sorry

end parallelogram_base_l177_17786


namespace cycle_gain_percent_l177_17707

/-- The gain percent on a cycle sale --/
theorem cycle_gain_percent (cost_price selling_price : ℝ) 
  (h1 : cost_price = 900)
  (h2 : selling_price = 1180) : 
  (selling_price - cost_price) / cost_price * 100 = 31.11 := by
  sorry

end cycle_gain_percent_l177_17707


namespace exponent_square_negative_product_l177_17700

theorem exponent_square_negative_product (a b : ℝ) : (-a^3 * b)^2 = a^6 * b^2 := by sorry

end exponent_square_negative_product_l177_17700


namespace tenth_root_unity_l177_17720

theorem tenth_root_unity : 
  ∃ (n : ℕ) (h : n < 10), 
    (Complex.tan (Real.pi / 5) + Complex.I) / (Complex.tan (Real.pi / 5) - Complex.I) = 
    Complex.exp (Complex.I * (2 * ↑n * Real.pi / 10)) :=
by sorry

end tenth_root_unity_l177_17720


namespace jumping_contest_l177_17749

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump frog_jump mouse_jump : ℕ) 
  (h1 : grasshopper_jump = 25)
  (h2 : mouse_jump = 31)
  (h3 : mouse_jump + 26 = frog_jump)
  (h4 : frog_jump > grasshopper_jump) :
  frog_jump - grasshopper_jump = 32 := by
sorry


end jumping_contest_l177_17749


namespace count_even_perfect_square_factors_l177_17727

/-- The number of even perfect square factors of 2^6 * 7^3 * 3^8 -/
def evenPerfectSquareFactors : ℕ := 30

/-- The original number -/
def originalNumber : ℕ := 2^6 * 7^3 * 3^8

/-- A function that counts the number of even perfect square factors of originalNumber -/
def countEvenPerfectSquareFactors : ℕ := sorry

theorem count_even_perfect_square_factors :
  countEvenPerfectSquareFactors = evenPerfectSquareFactors := by sorry

end count_even_perfect_square_factors_l177_17727


namespace solution_difference_l177_17754

theorem solution_difference (r s : ℝ) : 
  ((r - 5) * (r + 5) = 26 * r - 130) →
  ((s - 5) * (s + 5) = 26 * s - 130) →
  r ≠ s →
  r > s →
  r - s = 16 := by
sorry

end solution_difference_l177_17754


namespace magician_performances_l177_17750

/-- The number of performances a magician has put on --/
def num_performances : ℕ := 100

/-- The probability that an audience member never reappears --/
def prob_never_reappear : ℚ := 1/10

/-- The probability that two people reappear instead of one --/
def prob_two_reappear : ℚ := 1/5

/-- The total number of people who have reappeared --/
def total_reappeared : ℕ := 110

theorem magician_performances :
  (1 - prob_never_reappear - prob_two_reappear) * num_performances +
  2 * prob_two_reappear * num_performances = total_reappeared :=
by sorry

end magician_performances_l177_17750


namespace discount_problem_l177_17708

theorem discount_problem (list_price : ℝ) (final_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  list_price = 70 →
  final_price = 59.85 →
  discount1 = 10 →
  (list_price * (1 - discount1 / 100) * (1 - discount2 / 100) = final_price) →
  discount2 = 5 := by
sorry

end discount_problem_l177_17708


namespace line_perp_two_planes_implies_parallel_l177_17729

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between two planes -/
def parallel (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Main theorem: If a line is perpendicular to two different planes, then the planes are parallel -/
theorem line_perp_two_planes_implies_parallel (l : Line3D) (α β : Plane3D) :
  α ≠ β → perpendicular l α → perpendicular l β → parallel α β :=
sorry

end line_perp_two_planes_implies_parallel_l177_17729


namespace fence_panels_count_l177_17710

/-- Represents the components of a fence panel -/
structure FencePanel where
  sheets : Nat
  beams : Nat

/-- Represents the composition of sheets and beams -/
structure MetalComposition where
  rods_per_sheet : Nat
  rods_per_beam : Nat

/-- Calculates the number of fence panels given the total rods and composition -/
def calculate_fence_panels (total_rods : Nat) (panel : FencePanel) (comp : MetalComposition) : Nat :=
  total_rods / (panel.sheets * comp.rods_per_sheet + panel.beams * comp.rods_per_beam)

theorem fence_panels_count (total_rods : Nat) (panel : FencePanel) (comp : MetalComposition) :
  total_rods = 380 →
  panel.sheets = 3 →
  panel.beams = 2 →
  comp.rods_per_sheet = 10 →
  comp.rods_per_beam = 4 →
  calculate_fence_panels total_rods panel comp = 10 := by
  sorry

#eval calculate_fence_panels 380 ⟨3, 2⟩ ⟨10, 4⟩

end fence_panels_count_l177_17710


namespace intersection_equation_l177_17780

theorem intersection_equation (a b : ℝ) (hb : b ≠ 0) :
  ∃ m n : ℤ, (m : ℝ)^3 - a*(m : ℝ)^2 - b*(m : ℝ) = a*(m : ℝ) + b ∧
             (m : ℝ)^3 - a*(m : ℝ)^2 - b*(m : ℝ) = (n : ℝ) →
  2*a - b + 8 = 0 := by
sorry

end intersection_equation_l177_17780


namespace max_odd_integers_l177_17777

/-- Given a list of six positive integers, returns true if their product is even -/
def productIsEven (nums : List Nat) : Prop :=
  nums.length = 6 ∧ nums.all (· > 0) ∧ (nums.prod % 2 = 0)

/-- Given a list of six positive integers, returns true if their sum is odd -/
def sumIsOdd (nums : List Nat) : Prop :=
  nums.length = 6 ∧ nums.all (· > 0) ∧ (nums.sum % 2 = 1)

/-- Returns the count of odd numbers in a list -/
def oddCount (nums : List Nat) : Nat :=
  nums.filter (· % 2 = 1) |>.length

theorem max_odd_integers (nums : List Nat) 
  (h1 : productIsEven nums) (h2 : sumIsOdd nums) : 
  oddCount nums ≤ 5 ∧ ∃ (nums' : List Nat), productIsEven nums' ∧ sumIsOdd nums' ∧ oddCount nums' = 5 :=
sorry

end max_odd_integers_l177_17777


namespace bee_hive_problem_l177_17701

theorem bee_hive_problem (B : ℕ) : 
  (B / 5 : ℚ) + (B / 3 : ℚ) + (3 * ((B / 3 : ℚ) - (B / 5 : ℚ))) + 1 = B → B = 15 := by
  sorry

end bee_hive_problem_l177_17701


namespace peanut_seed_germination_probability_l177_17718

/-- The probability of exactly k successes in n independent trials,
    each with probability p of success. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability that exactly 2 out of 4 planted seeds will germinate,
    given that the probability of each seed germinating is 4/5. -/
theorem peanut_seed_germination_probability :
  binomial_probability 4 2 (4/5) = 96/625 := by
  sorry

end peanut_seed_germination_probability_l177_17718


namespace bisected_parallelogram_perimeter_l177_17704

/-- Represents a parallelogram with a bisected angle -/
structure BisectedParallelogram where
  -- The length of one segment created by the angle bisector
  segment1 : ℝ
  -- The length of the other segment created by the angle bisector
  segment2 : ℝ
  -- Assumption that the segments are 7 and 14 (in either order)
  h_segments : (segment1 = 7 ∧ segment2 = 14) ∨ (segment1 = 14 ∧ segment2 = 7)

/-- The perimeter of the parallelogram is either 56 or 70 -/
theorem bisected_parallelogram_perimeter (p : BisectedParallelogram) :
  let perimeter := 2 * (p.segment1 + p.segment2)
  perimeter = 56 ∨ perimeter = 70 := by
  sorry


end bisected_parallelogram_perimeter_l177_17704


namespace pyramid_theorem_l177_17755

/-- Regular quadrangular pyramid with a plane through diagonal of base and height -/
structure RegularQuadPyramid where
  /-- Side length of the base -/
  a : ℝ
  /-- Angle between opposite slant heights -/
  α : ℝ
  /-- Ratio of section area to lateral surface area -/
  k : ℝ
  /-- Base side length is positive -/
  a_pos : 0 < a
  /-- Angle is between 0 and π -/
  α_range : 0 < α ∧ α < π
  /-- k is positive -/
  k_pos : 0 < k

/-- Theorem about the cosine of the angle between slant heights and permissible k values -/
theorem pyramid_theorem (p : RegularQuadPyramid) :
  (Real.cos p.α = 64 * p.k^2 - 1) ∧ 
  (p.k ≤ Real.sqrt 2 / 4) := by
  sorry

end pyramid_theorem_l177_17755


namespace point_divides_segment_l177_17762

/-- Given two points A and B in 2D space, and a point P, this function checks if P divides the line segment AB in the given ratio m:n -/
def divides_segment (A B P : ℝ × ℝ) (m n : ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x, y) := P
  x = (m * x₂ + n * x₁) / (m + n) ∧
  y = (m * y₂ + n * y₁) / (m + n)

/-- The theorem states that the point (3.5, 8.5) divides the line segment between (2, 10) and (8, 4) in the ratio 1:3 -/
theorem point_divides_segment :
  divides_segment (2, 10) (8, 4) (3.5, 8.5) 1 3 := by
  sorry

end point_divides_segment_l177_17762


namespace arithmetic_sequence_1500th_term_l177_17728

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_1500th_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_term1 : a 1 = m)
  (h_term2 : a 2 = m + 2*n)
  (h_term3 : a 3 = 5*m - n)
  (h_term4 : a 4 = 3*m + 3*n)
  (h_term5 : a 5 = 7*m - n)
  : a 1500 = 2 :=
by sorry

end arithmetic_sequence_1500th_term_l177_17728


namespace group_shopping_popularity_justified_l177_17752

/-- Represents the practice of group shopping -/
structure GroupShopping where
  risks : ℕ  -- Number of risks associated with group shopping
  countries : ℕ  -- Number of countries where group shopping is practiced

/-- Factors contributing to group shopping popularity -/
structure PopularityFactors where
  cost_savings : ℝ  -- Percentage of cost savings
  quality_assessment : ℝ  -- Measure of quality assessment improvement
  trust_dynamics : ℝ  -- Measure of trust within the community

/-- Theorem stating that group shopping popularity is justified -/
theorem group_shopping_popularity_justified 
  (gs : GroupShopping) 
  (pf : PopularityFactors) : 
  gs.risks > 0 → 
  gs.countries > 10 → 
  pf.cost_savings > 20 → 
  pf.quality_assessment > 0.5 → 
  pf.trust_dynamics > 0.7 → 
  True := by
  sorry


end group_shopping_popularity_justified_l177_17752


namespace sum_of_three_squares_mod_8_l177_17764

theorem sum_of_three_squares_mod_8 (a b c : ℤ) : (a^2 + b^2 + c^2) % 8 ≠ 7 := by
  sorry

end sum_of_three_squares_mod_8_l177_17764


namespace m_range_for_three_integer_solutions_l177_17725

def inequality_system (x m : ℝ) : Prop :=
  2 * x - 1 ≤ 5 ∧ x - m > 0

def has_three_integer_solutions (m : ℝ) : Prop :=
  ∃ x₁ x₂ x₃ : ℤ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    inequality_system x₁ m ∧ inequality_system x₂ m ∧ inequality_system x₃ m ∧
    ∀ x : ℤ, inequality_system x m → x = x₁ ∨ x = x₂ ∨ x = x₃

theorem m_range_for_three_integer_solutions :
  ∀ m : ℝ, has_three_integer_solutions m ↔ 0 ≤ m ∧ m < 1 :=
sorry

end m_range_for_three_integer_solutions_l177_17725


namespace second_player_wins_l177_17735

-- Define the graph structure
structure GameGraph where
  vertices : Finset Char
  edges : Finset (Char × Char)
  start : Char
  degree : Char → Nat

-- Define the game rules
structure GameRules where
  graph : GameGraph
  current_player : Nat
  used_edges : Finset (Char × Char)
  current_position : Char

-- Define a move
def valid_move (rules : GameRules) (next : Char) : Prop :=
  (rules.current_position, next) ∈ rules.graph.edges ∧
  (rules.current_position, next) ∉ rules.used_edges

-- Define the winning condition
def is_winning_position (rules : GameRules) : Prop :=
  ∀ next, ¬(valid_move rules next)

-- Theorem statement
theorem second_player_wins (g : GameGraph)
  (h1 : g.vertices = {'A', 'B', 'C', 'D', 'E', 'F'})
  (h2 : g.start = 'A')
  (h3 : g.degree 'A' = 4)
  (h4 : g.degree 'B' = 5)
  (h5 : g.degree 'C' = 5)
  (h6 : g.degree 'D' = 3)
  (h7 : g.degree 'E' = 3)
  (h8 : g.degree 'F' = 5)
  : ∃ (strategy : GameRules → Char),
    ∀ (rules : GameRules),
      rules.graph = g →
      rules.current_player = 2 →
      (valid_move rules (strategy rules) ∧
       is_winning_position
         { graph := rules.graph,
           current_player := 1,
           used_edges := insert (rules.current_position, strategy rules) rules.used_edges,
           current_position := strategy rules }) :=
sorry

end second_player_wins_l177_17735


namespace barry_votes_difference_l177_17759

def election_votes (marcy_votes barry_votes joey_votes : ℕ) : Prop :=
  marcy_votes = 3 * barry_votes ∧
  ∃ x, barry_votes = 2 * (joey_votes + x) ∧
  marcy_votes = 66 ∧
  joey_votes = 8

theorem barry_votes_difference :
  ∀ marcy_votes barry_votes joey_votes,
  election_votes marcy_votes barry_votes joey_votes →
  barry_votes - joey_votes = 14 := by
sorry

end barry_votes_difference_l177_17759


namespace heavy_equipment_operator_pay_is_140_l177_17717

/-- Calculates the daily pay for heavy equipment operators given the total number of people hired,
    total payroll, number of laborers, and daily pay for laborers. -/
def heavy_equipment_operator_pay (total_hired : ℕ) (total_payroll : ℕ) (laborers : ℕ) (laborer_pay : ℕ) : ℕ :=
  (total_payroll - laborers * laborer_pay) / (total_hired - laborers)

/-- Proves that given the specified conditions, the daily pay for heavy equipment operators is $140. -/
theorem heavy_equipment_operator_pay_is_140 :
  heavy_equipment_operator_pay 35 3950 19 90 = 140 := by
  sorry

end heavy_equipment_operator_pay_is_140_l177_17717


namespace mothers_carrots_count_l177_17730

/-- The number of carrots Nancy picked -/
def nancys_carrots : ℕ := 38

/-- The number of good carrots -/
def good_carrots : ℕ := 71

/-- The number of bad carrots -/
def bad_carrots : ℕ := 14

/-- The number of carrots Nancy's mother picked -/
def mothers_carrots : ℕ := (good_carrots + bad_carrots) - nancys_carrots

theorem mothers_carrots_count : mothers_carrots = 47 := by
  sorry

end mothers_carrots_count_l177_17730


namespace alices_favorite_number_l177_17719

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem alices_favorite_number :
  ∃! n : ℕ,
    100 < n ∧ n < 200 ∧
    n % 13 = 0 ∧
    n % 3 ≠ 0 ∧
    (sum_of_digits n) % 4 = 0 :=
by sorry

end alices_favorite_number_l177_17719


namespace smallest_x_absolute_value_l177_17711

theorem smallest_x_absolute_value (x : ℝ) : 
  (∀ y : ℝ, |5*y + 15| = 40 → y ≥ x) ↔ x = -11 ∧ |5*x + 15| = 40 := by
  sorry

end smallest_x_absolute_value_l177_17711


namespace negative_cube_squared_l177_17745

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end negative_cube_squared_l177_17745


namespace combined_mean_of_two_sets_l177_17753

theorem combined_mean_of_two_sets (set1_count set2_count : ℕ) 
  (set1_mean set2_mean : ℚ) : 
  set1_count = 7 → 
  set2_count = 8 → 
  set1_mean = 15 → 
  set2_mean = 30 → 
  let total_count := set1_count + set2_count
  let total_sum := set1_count * set1_mean + set2_count * set2_mean
  (total_sum / total_count : ℚ) = 23 := by
sorry

end combined_mean_of_two_sets_l177_17753


namespace signUpWaysCorrect_l177_17709

/-- The number of ways four students can sign up for three sports -/
def signUpWays : ℕ := 81

/-- The number of students -/
def numStudents : ℕ := 4

/-- The number of sports -/
def numSports : ℕ := 3

theorem signUpWaysCorrect : signUpWays = numSports ^ numStudents := by
  sorry

end signUpWaysCorrect_l177_17709


namespace distance_to_focus_of_parabola_l177_17784

/-- The distance from a point on a parabola to its focus -/
theorem distance_to_focus_of_parabola (x y : ℝ) :
  x^2 = 2*y →  -- Parabola equation
  y = 3 →      -- Ordinate of point P
  (y + 1/4) = 7/2  -- Distance to focus
  := by sorry

end distance_to_focus_of_parabola_l177_17784


namespace rent_increase_group_size_l177_17702

theorem rent_increase_group_size :
  ∀ (n : ℕ) (initial_average rent_increase new_average original_rent : ℚ),
    initial_average = 800 →
    new_average = 880 →
    original_rent = 1600 →
    rent_increase = 0.2 * original_rent →
    n * new_average = n * initial_average + rent_increase →
    n = 4 := by
  sorry

end rent_increase_group_size_l177_17702


namespace students_in_lunchroom_l177_17789

theorem students_in_lunchroom 
  (students_per_table : ℕ) 
  (number_of_tables : ℕ) 
  (h1 : students_per_table = 6) 
  (h2 : number_of_tables = 34) : 
  students_per_table * number_of_tables = 204 := by
  sorry

end students_in_lunchroom_l177_17789


namespace suraj_average_after_17th_innings_l177_17757

def average_after_17th_innings (initial_average : ℝ) (score_17th : ℝ) (average_increase : ℝ) : Prop :=
  let total_runs_16 := 16 * initial_average
  let total_runs_17 := total_runs_16 + score_17th
  let new_average := total_runs_17 / 17
  new_average = initial_average + average_increase

theorem suraj_average_after_17th_innings :
  ∃ (initial_average : ℝ),
    average_after_17th_innings initial_average 112 6 ∧
    initial_average + 6 = 16 :=
by
  sorry

#check suraj_average_after_17th_innings

end suraj_average_after_17th_innings_l177_17757


namespace lcm_problem_l177_17797

theorem lcm_problem (a b c : ℕ+) (h1 : a = 24) (h2 : b = 42) 
  (h3 : Nat.lcm a (Nat.lcm b c) = 504) : c = 3 := by
  sorry

end lcm_problem_l177_17797


namespace symmetry_of_f_2x_l177_17768

def center_of_symmetry (f : ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ k : ℤ, p = (k * Real.pi / 2 - Real.pi / 8, 0)}

theorem symmetry_of_f_2x (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x + 2 * f (-x) = 3 * Real.cos x - Real.sin x) :
  center_of_symmetry (fun x ↦ f (2 * x)) = 
    {p : ℝ × ℝ | ∃ k : ℤ, p = (k * Real.pi / 2 - Real.pi / 8, 0)} := by
  sorry

end symmetry_of_f_2x_l177_17768


namespace average_of_dataset_l177_17739

def dataset : List ℝ := [5, 9, 9, 3, 4]

theorem average_of_dataset : 
  (dataset.sum / dataset.length : ℝ) = 6 := by sorry

end average_of_dataset_l177_17739


namespace trapezoid_long_side_length_l177_17740

/-- Represents a square divided into two trapezoids and a quadrilateral -/
structure DividedSquare where
  side_length : ℝ
  segment_length : ℝ
  trapezoid_long_side : ℝ

/-- The conditions of the problem -/
def problem_conditions (s : DividedSquare) : Prop :=
  s.side_length = 2 ∧
  s.segment_length = 1 ∧
  (s.trapezoid_long_side + s.segment_length) * s.segment_length / 2 = s.side_length^2 / 3

/-- The theorem to be proved -/
theorem trapezoid_long_side_length (s : DividedSquare) :
  problem_conditions s → s.trapezoid_long_side = 5/3 :=
by sorry

end trapezoid_long_side_length_l177_17740


namespace after_school_program_enrollment_l177_17799

theorem after_school_program_enrollment (drama_students music_students both_students : ℕ) 
  (h1 : drama_students = 41)
  (h2 : music_students = 28)
  (h3 : both_students = 15) :
  drama_students + music_students - both_students = 54 := by
sorry

end after_school_program_enrollment_l177_17799


namespace equal_share_money_l177_17769

theorem equal_share_money (emani_money : ℕ) (difference : ℕ) : 
  emani_money = 150 →
  difference = 30 →
  (emani_money + (emani_money - difference)) / 2 = 135 := by
  sorry

end equal_share_money_l177_17769


namespace boys_playing_marbles_count_l177_17748

/-- The number of marbles Haley has -/
def total_marbles : ℕ := 26

/-- The number of marbles each boy receives -/
def marbles_per_boy : ℕ := 2

/-- The number of boys who love to play marbles -/
def boys_playing_marbles : ℕ := total_marbles / marbles_per_boy

theorem boys_playing_marbles_count : boys_playing_marbles = 13 := by
  sorry

end boys_playing_marbles_count_l177_17748


namespace value_of_x_l177_17792

theorem value_of_x (x y z : ℝ) : x = 3 * y ∧ y = z / 3 ∧ z = 90 → x = 90 := by
  sorry

end value_of_x_l177_17792


namespace subset_intersection_count_l177_17774

-- Define the set S with n elements
variable (n : ℕ)
variable (S : Finset (Fin n))

-- Define k subsets of S
variable (k : ℕ)
variable (A : Fin k → Finset (Fin n))

-- Conditions
variable (h1 : ∀ i, A i ⊆ S)
variable (h2 : ∀ i j, i ≠ j → (A i ∩ A j).Nonempty)
variable (h3 : ∀ X, X ⊆ S → (∀ i, (X ∩ A i).Nonempty) → ∃ i, X = A i)

-- Theorem statement
theorem subset_intersection_count : k = 2^(n-1) := by
  sorry

end subset_intersection_count_l177_17774


namespace probability_of_drawing_parts_l177_17788

def total_parts : ℕ := 10
def drawn_parts : ℕ := 6

def prob_draw_one (n m k : ℕ) : ℚ :=
  (n.choose k) / (m.choose k)

def prob_draw_two (n m k : ℕ) : ℚ :=
  ((n-2).choose (k-2)) / (m.choose k)

theorem probability_of_drawing_parts :
  (prob_draw_one (total_parts - 1) total_parts drawn_parts = 3/5) ∧
  (prob_draw_two (total_parts - 2) total_parts drawn_parts = 1/3) := by
  sorry

end probability_of_drawing_parts_l177_17788


namespace train_passing_time_l177_17737

/-- The time taken for a train to pass a man moving in the same direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 150 →
  train_speed = 62 * (1000 / 3600) →
  man_speed = 8 * (1000 / 3600) →
  (train_length / (train_speed - man_speed)) = 10 :=
by sorry

end train_passing_time_l177_17737


namespace sufficient_not_necessary_l177_17761

theorem sufficient_not_necessary (x : ℝ) : 
  (∀ x, (0 < x ∧ x < 2) → (x^2 - x - 2 < 0)) ∧ 
  (∃ x, (x^2 - x - 2 < 0) ∧ ¬(0 < x ∧ x < 2)) :=
by sorry

end sufficient_not_necessary_l177_17761


namespace escalator_time_l177_17794

theorem escalator_time (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) :
  escalator_speed = 8 →
  person_speed = 2 →
  escalator_length = 160 →
  escalator_length / (escalator_speed + person_speed) = 16 := by
  sorry

end escalator_time_l177_17794


namespace equation_solution_l177_17705

theorem equation_solution : ∃ (x : ℝ), 
  x ≠ 2 ∧ x ≠ 1 ∧ x ≠ -6 ∧
  (x = 3 * Real.sqrt 2 ∨ x = -3 * Real.sqrt 2) ∧
  (3 * x + 6) / (x^2 + 5*x - 6) = (x - 3) / (x - 2) := by
  sorry

end equation_solution_l177_17705


namespace cookies_left_after_three_days_l177_17765

/-- Calculates the number of cookies left after a specified number of days -/
def cookies_left (initial_cookies : ℕ) (first_day_consumption : ℕ) (julie_daily : ℕ) (matt_daily : ℕ) (days : ℕ) : ℕ :=
  initial_cookies - (first_day_consumption + (julie_daily + matt_daily) * days)

/-- Theorem stating the number of cookies left after 3 days -/
theorem cookies_left_after_three_days : 
  cookies_left 32 9 2 3 3 = 8 := by
  sorry

end cookies_left_after_three_days_l177_17765


namespace min_value_x_plus_two_over_x_l177_17731

theorem min_value_x_plus_two_over_x (x : ℝ) (h : x > 0) :
  x + 2 / x ≥ 2 * Real.sqrt 2 ∧
  (x + 2 / x = 2 * Real.sqrt 2 ↔ x = Real.sqrt 2) := by
  sorry

end min_value_x_plus_two_over_x_l177_17731


namespace sufficient_not_necessary_condition_l177_17715

theorem sufficient_not_necessary_condition : 
  (∃ x : ℝ, x ≠ 5 ∧ x^2 - 4*x - 5 = 0) ∧
  (∀ x : ℝ, x = 5 → x^2 - 4*x - 5 = 0) :=
by sorry

end sufficient_not_necessary_condition_l177_17715


namespace quadratic_equation_solution_l177_17795

theorem quadratic_equation_solution (k : ℝ) : 
  (∃ x : ℝ, x^2 + k*x - 2 = 0 ∧ (x + 1) / (x - 1) = 3) → 
  k = -1 ∧ ∃ y : ℝ, y ≠ 2 ∧ y^2 + k*y - 2 = 0 ∧ y = -1 := by
  sorry

end quadratic_equation_solution_l177_17795


namespace find_x_l177_17771

theorem find_x (a b : ℝ) (x y r : ℝ) (h1 : b ≠ 0) (h2 : r = (3*a)^(3*b)) (h3 : r = a^b * (x + y)^b) (h4 : y = 3*a) :
  x = 27*a^2 - 3*a := by
sorry

end find_x_l177_17771


namespace ball_contact_height_l177_17758

theorem ball_contact_height (horizontal_distance : ℝ) (hypotenuse : ℝ) (height : ℝ) : 
  horizontal_distance = 7 → hypotenuse = 53 → height ^ 2 + horizontal_distance ^ 2 = hypotenuse ^ 2 → height = 2 := by
  sorry

end ball_contact_height_l177_17758


namespace milk_problem_solution_l177_17781

/-- Calculates the final amount of milk in a storage tank given initial amount,
    pumping out rate and duration, and adding rate and duration. -/
def final_milk_amount (initial : ℝ) (pump_rate : ℝ) (pump_hours : ℝ) 
                      (add_rate : ℝ) (add_hours : ℝ) : ℝ :=
  initial - pump_rate * pump_hours + add_rate * add_hours

/-- Theorem stating that given the specific conditions from the problem,
    the final amount of milk in the storage tank is 28,980 gallons. -/
theorem milk_problem_solution :
  final_milk_amount 30000 2880 4 1500 7 = 28980 := by
  sorry

end milk_problem_solution_l177_17781


namespace angle_value_l177_17747

theorem angle_value (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.tan (α + β) = 3 →
  Real.tan β = 1/2 →
  α = π/4 := by sorry

end angle_value_l177_17747


namespace expected_girls_left_10_7_l177_17742

/-- The expected number of girls standing to the left of all boys in a random lineup -/
def expected_girls_left (num_boys num_girls : ℕ) : ℚ :=
  (num_girls : ℚ) / (num_boys + num_girls + 1 : ℚ)

/-- Theorem: In a random lineup of 10 boys and 7 girls, the expected number of girls 
    standing to the left of all boys is 7/11 -/
theorem expected_girls_left_10_7 :
  expected_girls_left 10 7 = 7 / 11 := by
  sorry

end expected_girls_left_10_7_l177_17742


namespace sally_cut_six_orchids_l177_17796

/-- The number of red orchids Sally cut from her garden -/
def orchids_cut (initial_red : ℕ) (final_red : ℕ) : ℕ :=
  final_red - initial_red

/-- Theorem stating that Sally cut 6 red orchids -/
theorem sally_cut_six_orchids (initial_red : ℕ) (final_red : ℕ) 
  (h1 : initial_red = 9)
  (h2 : final_red = 15) : 
  orchids_cut initial_red final_red = 6 := by
  sorry

end sally_cut_six_orchids_l177_17796


namespace function_value_when_previous_is_one_l177_17783

theorem function_value_when_previous_is_one 
  (f : ℤ → ℤ) 
  (h1 : ∀ n : ℤ, f n = f (n - 1) - n) 
  (h2 : f 4 = 12) :
  ∀ n : ℤ, f (n - 1) = 1 → f n = 7 := by
sorry

end function_value_when_previous_is_one_l177_17783


namespace largest_n_divisibility_l177_17746

theorem largest_n_divisibility : ∃ (n : ℕ), n = 20 ∧ 
  (∀ m : ℕ, m > n → ¬((m^3 + 150) % (m + 5) = 0)) ∧ 
  ((n^3 + 150) % (n + 5) = 0) := by
  sorry

end largest_n_divisibility_l177_17746


namespace base_b_problem_l177_17791

theorem base_b_problem (b : ℕ) : 
  b > 1 ∧ 
  (2 * b + 5 < b^2) ∧ 
  (5 * b + 2 < b^2) ∧ 
  (5 * b + 2 = 2 * (2 * b + 5)) → 
  b = 8 := by sorry

end base_b_problem_l177_17791


namespace abs_equation_solution_difference_l177_17787

theorem abs_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (abs (x₁ + 3) = 15 ∧ abs (x₂ + 3) = 15) ∧ 
  x₁ ≠ x₂ ∧
  abs (x₁ - x₂) = 30 := by
  sorry

end abs_equation_solution_difference_l177_17787


namespace z_in_second_quadrant_l177_17779

def z₁ : ℂ := Complex.I
def z₂ : ℂ := 1 + Complex.I

theorem z_in_second_quadrant : 
  let z : ℂ := z₁ * z₂
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end z_in_second_quadrant_l177_17779


namespace problem_solution_l177_17776

theorem problem_solution :
  ∀ (m x : ℝ),
    (m = 1 → (((x - 3*m) * (x - m) < 0 ∧ |x - 3| ≤ 1) ↔ (2 ≤ x ∧ x < 3))) ∧
    (m > 0 → ((∀ x, |x - 3| ≤ 1 → (x - 3*m) * (x - m) < 0) ∧
              (∃ x, (x - 3*m) * (x - m) < 0 ∧ |x - 3| > 1)) ↔
             (4/3 < m ∧ m < 2)) :=
by sorry

end problem_solution_l177_17776


namespace puppy_sleep_duration_l177_17772

theorem puppy_sleep_duration (connor_sleep : ℕ) (luke_sleep : ℕ) (puppy_sleep : ℕ) : 
  connor_sleep = 6 →
  luke_sleep = connor_sleep + 2 →
  puppy_sleep = 2 * luke_sleep →
  puppy_sleep = 16 := by
sorry

end puppy_sleep_duration_l177_17772


namespace a_gt_1_sufficient_not_necessary_for_a_sq_gt_1_l177_17766

theorem a_gt_1_sufficient_not_necessary_for_a_sq_gt_1 :
  (∀ a : ℝ, a > 1 → a^2 > 1) ∧
  (∃ a : ℝ, a ≤ 1 ∧ a^2 > 1) := by
  sorry

end a_gt_1_sufficient_not_necessary_for_a_sq_gt_1_l177_17766


namespace malcom_cards_left_l177_17716

theorem malcom_cards_left (brandon_cards : ℕ) (malcom_extra_cards : ℕ) : 
  brandon_cards = 20 →
  malcom_extra_cards = 8 →
  (brandon_cards + malcom_extra_cards) / 2 = 14 := by
sorry

end malcom_cards_left_l177_17716


namespace max_value_implies_a_values_exactly_two_a_values_l177_17743

/-- The function f for a given real number a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

/-- The theorem stating the possible values of a -/
theorem max_value_implies_a_values (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, f a x ≤ 2) ∧ 
  (∃ x ∈ Set.Icc 0 1, f a x = 2) → 
  a = -1 ∨ a = 2 := by
sorry

/-- The main theorem stating that there are exactly two possible values for a -/
theorem exactly_two_a_values : 
  ∃! s : Set ℝ, s = {-1, 2} ∧ 
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 1, f a x ≤ 2) ∧ 
           (∃ x ∈ Set.Icc 0 1, f a x = 2) → 
           a ∈ s := by
sorry

end max_value_implies_a_values_exactly_two_a_values_l177_17743


namespace arithmetic_sequence_count_l177_17726

def is_arithmetic (s : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, s (n + 1) = s n + d

def seq1 (n : ℕ) : ℚ := n + 4
def seq2 (n : ℕ) : ℚ := if n % 2 = 0 then 3 - 3 * (n / 2) else 0
def seq3 (n : ℕ) : ℚ := 0
def seq4 (n : ℕ) : ℚ := (n + 1) / 10

theorem arithmetic_sequence_count :
  (is_arithmetic seq1) ∧
  (¬ is_arithmetic seq2) ∧
  (is_arithmetic seq3) ∧
  (is_arithmetic seq4) := by sorry

end arithmetic_sequence_count_l177_17726


namespace product_of_four_consecutive_integers_plus_one_is_perfect_square_l177_17785

theorem product_of_four_consecutive_integers_plus_one_is_perfect_square (n : ℤ) :
  ∃ m : ℤ, (n - 1) * n * (n + 1) * (n + 2) + 1 = m ^ 2 := by
  sorry

end product_of_four_consecutive_integers_plus_one_is_perfect_square_l177_17785


namespace positive_number_square_root_l177_17714

theorem positive_number_square_root (x : ℝ) : 
  x > 0 → Real.sqrt ((7 * x) / 3) = x → x = 7 / 3 := by sorry

end positive_number_square_root_l177_17714


namespace smallest_prime_after_six_nonprimes_l177_17767

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_nonprimes (start : ℕ) (count : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ start ∧ k < start + count → ¬(is_prime k)

theorem smallest_prime_after_six_nonprimes :
  ∃ n : ℕ, consecutive_nonprimes n 6 ∧ 
           is_prime (n + 6) ∧ 
           ∀ m : ℕ, m < n → ¬(consecutive_nonprimes m 6 ∧ is_prime (m + 6)) ∧
           n + 6 = 37 :=
by sorry

end smallest_prime_after_six_nonprimes_l177_17767


namespace equation_solution_l177_17713

theorem equation_solution : ∃ x : ℝ, (9 / (x + 3 / 0.75) = 1) ∧ (x = 5) := by
  sorry

end equation_solution_l177_17713


namespace cube_sum_of_roots_l177_17770

theorem cube_sum_of_roots (a b c : ℂ) : 
  (5 * a^3 + 2003 * a + 3005 = 0) → 
  (5 * b^3 + 2003 * b + 3005 = 0) → 
  (5 * c^3 + 2003 * c + 3005 = 0) → 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 1803 := by
sorry

end cube_sum_of_roots_l177_17770


namespace tens_digit_of_1047_pow_1024_minus_1049_l177_17732

theorem tens_digit_of_1047_pow_1024_minus_1049 : ∃ n : ℕ, (1047^1024 - 1049) % 100 = 32 ∧ n * 10 + 3 = (1047^1024 - 1049) / 10 % 10 := by
  sorry

end tens_digit_of_1047_pow_1024_minus_1049_l177_17732


namespace range_of_a_range_of_m_l177_17744

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * |x - 1| - a

-- Theorem 1
theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, f a x - 2 * |x - 7| ≤ 0) → a ≥ -12 := by sorry

-- Theorem 2
theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, f 1 x + |x + 7| ≥ m) → m ≤ 7 := by sorry

end range_of_a_range_of_m_l177_17744


namespace imaginary_part_of_z_l177_17751

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := 3 * i / (1 + i)
  Complex.im z = 3 / 2 := by sorry

end imaginary_part_of_z_l177_17751


namespace total_fruits_l177_17778

def papaya_trees : ℕ := 2
def mango_trees : ℕ := 3
def papayas_per_tree : ℕ := 10
def mangos_per_tree : ℕ := 20

theorem total_fruits : 
  papaya_trees * papayas_per_tree + mango_trees * mangos_per_tree = 80 := by
  sorry

end total_fruits_l177_17778


namespace age_difference_l177_17798

theorem age_difference (a b c : ℕ) : 
  b = 10 →
  b = 2 * c →
  a + b + c = 27 →
  a = b + 2 :=
by sorry

end age_difference_l177_17798


namespace three_officers_from_six_people_l177_17722

/-- The number of ways to choose three distinct officers from a group of people. -/
def chooseThreeOfficers (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- Theorem stating that choosing three distinct officers from 6 people results in 120 ways. -/
theorem three_officers_from_six_people : chooseThreeOfficers 6 = 120 := by
  sorry

end three_officers_from_six_people_l177_17722


namespace fifteenth_term_of_sequence_l177_17773

def geometricSequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem fifteenth_term_of_sequence :
  let a₁ : ℚ := 27
  let r : ℚ := 1/6
  geometricSequence a₁ r 15 = 1/14155776 := by
sorry

end fifteenth_term_of_sequence_l177_17773


namespace birdseed_mix_problem_l177_17793

/-- Represents the composition of a birdseed brand -/
structure BirdseedBrand where
  millet : Float
  sunflower : Float

/-- Represents a mix of two birdseed brands -/
structure BirdseedMix where
  brandA : BirdseedBrand
  brandB : BirdseedBrand
  proportionA : Float

theorem birdseed_mix_problem (mixA : BirdseedBrand) (mixB : BirdseedBrand) (mix : BirdseedMix) :
  mixA.millet = 0.4 →
  mixA.sunflower = 0.6 →
  mixB.millet = 0.65 →
  mix.brandA = mixA →
  mix.brandB = mixB →
  mix.proportionA = 0.6 →
  (mix.proportionA * mixA.sunflower + (1 - mix.proportionA) * mixB.sunflower = 0.5) →
  mixB.sunflower = 0.35 := by
  sorry

#check birdseed_mix_problem

end birdseed_mix_problem_l177_17793


namespace certain_amount_added_l177_17723

theorem certain_amount_added (x y : ℝ) : 
  x = 18 → 3 * (2 * x + y) = 123 → y = 5 := by
  sorry

end certain_amount_added_l177_17723


namespace count_negative_numbers_l177_17790

theorem count_negative_numbers : let numbers := [-(-3), |-2|, (-2)^3, -3^2]
  ∃ (negative_count : ℕ), negative_count = (numbers.filter (λ x => x < 0)).length ∧ negative_count = 2 := by
  sorry

end count_negative_numbers_l177_17790


namespace substance_volume_l177_17733

/-- Given a substance where 1 gram occupies 10 cubic centimeters, 
    prove that 100 kg of this substance occupies 1 cubic meter. -/
theorem substance_volume (substance : Type) 
  (volume : substance → ℝ) 
  (mass : substance → ℝ) 
  (s : substance) 
  (h1 : volume s = mass s * 10 * 1000000⁻¹) 
  (h2 : mass s = 100) : 
  volume s = 1 := by
sorry

end substance_volume_l177_17733


namespace apple_products_cost_l177_17738

/-- Calculate the final cost of Apple products after discounts, taxes, and cashback -/
theorem apple_products_cost (iphone_price iwatch_price ipad_price : ℝ)
  (iphone_discount iwatch_discount ipad_discount : ℝ)
  (iphone_tax iwatch_tax ipad_tax : ℝ)
  (cashback : ℝ)
  (h1 : iphone_price = 800)
  (h2 : iwatch_price = 300)
  (h3 : ipad_price = 500)
  (h4 : iphone_discount = 0.15)
  (h5 : iwatch_discount = 0.10)
  (h6 : ipad_discount = 0.05)
  (h7 : iphone_tax = 0.07)
  (h8 : iwatch_tax = 0.05)
  (h9 : ipad_tax = 0.06)
  (h10 : cashback = 0.02) :
  ∃ (total_cost : ℝ), 
    abs (total_cost - 1484.31) < 0.01 ∧
    total_cost = 
      (1 - cashback) * 
      ((iphone_price * (1 - iphone_discount) * (1 + iphone_tax)) +
       (iwatch_price * (1 - iwatch_discount) * (1 + iwatch_tax)) +
       (ipad_price * (1 - ipad_discount) * (1 + ipad_tax))) :=
by sorry


end apple_products_cost_l177_17738


namespace product_87_93_l177_17706

theorem product_87_93 : 87 * 93 = 8091 := by
  sorry

end product_87_93_l177_17706


namespace angle_FDE_l177_17756

theorem angle_FDE (BAC : Real) (h : BAC = 70) : ∃ FDE : Real, FDE = 40 := by
  sorry

end angle_FDE_l177_17756


namespace upstream_speed_calculation_l177_17703

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  stillWater : ℝ
  downstream : ℝ

/-- Calculates the upstream speed given the rowing speeds in still water and downstream -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem stating that given the specific conditions, the upstream speed is 15 kmph -/
theorem upstream_speed_calculation (s : RowingSpeed) 
  (h1 : s.stillWater = 25) 
  (h2 : s.downstream = 35) : 
  upstreamSpeed s = 15 := by
  sorry

#eval upstreamSpeed { stillWater := 25, downstream := 35 }

end upstream_speed_calculation_l177_17703


namespace cubic_eq_given_quadratic_l177_17741

theorem cubic_eq_given_quadratic (x : ℝ) :
  x^2 + 5*x - 990 = 0 → x^3 + 6*x^2 - 985*x + 1012 = 2002 := by
  sorry

end cubic_eq_given_quadratic_l177_17741


namespace frame_width_is_five_l177_17724

/-- A rectangular frame containing three square photograph openings with uniform width. -/
structure PhotoFrame where
  /-- The side length of each square opening -/
  opening_side : ℝ
  /-- The width of the frame -/
  frame_width : ℝ

/-- The perimeter of one square opening -/
def opening_perimeter (frame : PhotoFrame) : ℝ :=
  4 * frame.opening_side

/-- The perimeter of the entire rectangular frame -/
def frame_perimeter (frame : PhotoFrame) : ℝ :=
  2 * (frame.opening_side + 2 * frame.frame_width) + 2 * (3 * frame.opening_side + 2 * frame.frame_width)

/-- Theorem stating that if the perimeter of one opening is 60 cm and the perimeter of the entire frame is 180 cm, 
    then the width of the frame is 5 cm -/
theorem frame_width_is_five (frame : PhotoFrame) 
  (h1 : opening_perimeter frame = 60) 
  (h2 : frame_perimeter frame = 180) : 
  frame.frame_width = 5 := by
  sorry

end frame_width_is_five_l177_17724


namespace probability_two_students_together_l177_17734

/-- The probability of two specific students standing together in a row of 4 students -/
theorem probability_two_students_together (n : ℕ) (h : n = 4) : 
  (2 * 3 * 2 * 1 : ℚ) / (n * (n - 1) * (n - 2) * (n - 3)) = 1 / 2 :=
by sorry

end probability_two_students_together_l177_17734


namespace range_of_2sin_squared_l177_17782

theorem range_of_2sin_squared (x : ℝ) : 0 ≤ 2 * (Real.sin x)^2 ∧ 2 * (Real.sin x)^2 ≤ 2 := by
  sorry

end range_of_2sin_squared_l177_17782


namespace collinear_points_m_value_l177_17775

/-- Given three points A, B, and C in a 2D plane, this function checks if they are collinear -/
def are_collinear (A B C : ℝ × ℝ) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Theorem stating that if A(0,2), B(3,0), and C(m,1-m) are collinear, then m = -9 -/
theorem collinear_points_m_value :
  ∀ m : ℝ, are_collinear (0, 2) (3, 0) (m, 1 - m) → m = -9 :=
by
  sorry

end collinear_points_m_value_l177_17775
