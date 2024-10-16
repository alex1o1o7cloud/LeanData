import Mathlib

namespace NUMINAMATH_CALUDE_common_ratio_is_two_l2145_214572

def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

theorem common_ratio_is_two 
  (a₁ : ℝ) 
  (q : ℝ) 
  (h_positive : ∀ n : ℕ, geometric_sequence a₁ q n > 0)
  (h_product : geometric_sequence a₁ q 1 * geometric_sequence a₁ q 5 = 16)
  (h_first_term : a₁ = 2) :
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_common_ratio_is_two_l2145_214572


namespace NUMINAMATH_CALUDE_four_card_selection_theorem_l2145_214598

/-- Represents a standard deck of 52 cards -/
def StandardDeck : Type := Unit

/-- Represents the four suits in a deck of cards -/
inductive Suit
| hearts | diamonds | clubs | spades

/-- Represents the rank of a card -/
inductive Rank
| ace | two | three | four | five | six | seven | eight | nine | ten
| jack | queen | king

/-- Determines if a rank is royal (J, Q, K) -/
def isRoyal (r : Rank) : Bool :=
  match r with
  | Rank.jack | Rank.queen | Rank.king => true
  | _ => false

/-- Represents a card with a suit and rank -/
structure Card where
  suit : Suit
  rank : Rank

/-- The number of ways to choose 4 cards from two standard decks -/
def numWaysToChoose4Cards (deck1 deck2 : StandardDeck) : ℕ := sorry

theorem four_card_selection_theorem (deck1 deck2 : StandardDeck) :
  numWaysToChoose4Cards deck1 deck2 = 438400 := by sorry

end NUMINAMATH_CALUDE_four_card_selection_theorem_l2145_214598


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2145_214505

def M : Set ℝ := {x | x > 2}
def P : Set ℝ := {x | x < 3}

theorem necessary_but_not_sufficient :
  (∀ x, x ∈ M ∩ P → (x ∈ M ∨ x ∈ P)) ∧
  (∃ x, (x ∈ M ∨ x ∈ P) ∧ x ∉ M ∩ P) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2145_214505


namespace NUMINAMATH_CALUDE_even_number_of_fours_l2145_214588

theorem even_number_of_fours (n₃ n₄ n₅ : ℕ) : 
  n₃ + n₄ + n₅ = 80 →
  3 * n₃ + 4 * n₄ + 5 * n₅ = 276 →
  Even n₄ := by
sorry

end NUMINAMATH_CALUDE_even_number_of_fours_l2145_214588


namespace NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_angled_l2145_214579

/-- If the angles of a triangle are in the ratio 1:2:3, then the triangle is right-angled. -/
theorem triangle_with_angle_ratio_1_2_3_is_right_angled (A B C : ℝ) 
  (h_angle_sum : A + B + C = 180) 
  (h_angle_ratio : ∃ (x : ℝ), A = x ∧ B = 2*x ∧ C = 3*x) : 
  C = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_angled_l2145_214579


namespace NUMINAMATH_CALUDE_ratio_problem_l2145_214503

theorem ratio_problem (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : a * b * c / (d * e * f) = 0.1875) :
  e / f = 0.125 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2145_214503


namespace NUMINAMATH_CALUDE_parallel_lines_b_value_l2145_214519

theorem parallel_lines_b_value (b : ℝ) : 
  (∀ x y : ℝ, 4 * y - 3 * x - 2 = 0 ↔ y = (3/4) * x + 1/2) →
  (∀ x y : ℝ, 6 * y + b * x + 1 = 0 ↔ y = (-b/6) * x - 1/6) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    (4 * y₁ - 3 * x₁ - 2 = 0 ∧ 6 * y₂ + b * x₂ + 1 = 0) → 
    (y₂ - y₁) / (x₂ - x₁) = (3/4)) →
  b = -4.5 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_b_value_l2145_214519


namespace NUMINAMATH_CALUDE_algebraic_simplification_l2145_214537

theorem algebraic_simplification (y : ℝ) (h : y ≠ 0) :
  (20 * y^3) * (8 * y^2) * (1 / (4*y)^3) = (5/2) * y^2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l2145_214537


namespace NUMINAMATH_CALUDE_triangle_area_equals_sqrt_semiperimeter_l2145_214584

theorem triangle_area_equals_sqrt_semiperimeter 
  (x y z : ℝ) (a b c s Δ : ℝ) 
  (ha : a = x / y + y / z)
  (hb : b = y / z + z / x)
  (hc : c = z / x + x / y)
  (hs : s = (a + b + c) / 2) :
  Δ = Real.sqrt s := by sorry

end NUMINAMATH_CALUDE_triangle_area_equals_sqrt_semiperimeter_l2145_214584


namespace NUMINAMATH_CALUDE_total_books_l2145_214546

theorem total_books (joan_books tom_books : ℕ) 
  (h1 : joan_books = 10) 
  (h2 : tom_books = 38) : 
  joan_books + tom_books = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l2145_214546


namespace NUMINAMATH_CALUDE_extrema_relations_l2145_214589

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x^2 + 1)

theorem extrema_relations (a b : ℝ) 
  (h1 : ∀ x, f x ≥ a) 
  (h2 : ∃ x, f x = a)
  (h3 : ∀ x, f x ≤ b) 
  (h4 : ∃ x, f x = b) :
  (∀ x, (x^3 - 1) / (x^6 + 1) ≥ a) ∧
  (∃ x, (x^3 - 1) / (x^6 + 1) = a) ∧
  (∀ x, (x^3 - 1) / (x^6 + 1) ≤ b) ∧
  (∃ x, (x^3 - 1) / (x^6 + 1) = b) ∧
  (∀ x, (x + 1) / (x^2 + 1) ≥ -b) ∧
  (∃ x, (x + 1) / (x^2 + 1) = -b) ∧
  (∀ x, (x + 1) / (x^2 + 1) ≤ -a) ∧
  (∃ x, (x + 1) / (x^2 + 1) = -a) :=
by sorry

end NUMINAMATH_CALUDE_extrema_relations_l2145_214589


namespace NUMINAMATH_CALUDE_routes_4_by_3_l2145_214547

/-- The number of routes in a rectangular grid --/
def num_routes (m n : ℕ) : ℕ := Nat.choose (m + n) m

/-- Theorem: The number of routes in a 4 by 3 grid is 35 --/
theorem routes_4_by_3 : num_routes 3 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_routes_4_by_3_l2145_214547


namespace NUMINAMATH_CALUDE_number_puzzle_l2145_214501

theorem number_puzzle : ∃ x : ℝ, (x / 5 + 6 = 65) ∧ (x = 295) := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2145_214501


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2145_214524

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides -/
theorem regular_polygon_sides (n : ℕ) : n > 0 → (360 : ℝ) / n = 18 → n = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2145_214524


namespace NUMINAMATH_CALUDE_sum_equals_negative_six_l2145_214575

theorem sum_equals_negative_six (a b c d : ℤ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 8) : 
  a + b + c + d = -6 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_negative_six_l2145_214575


namespace NUMINAMATH_CALUDE_range_of_p_l2145_214552

def h (x : ℝ) : ℝ := 4 * x - 3

def p (x : ℝ) : ℝ := h (h (h x))

theorem range_of_p :
  ∀ y ∈ Set.range (fun x => p x),
  (1 ≤ x ∧ x ≤ 3) → (1 ≤ y ∧ y ≤ 129) :=
by sorry

end NUMINAMATH_CALUDE_range_of_p_l2145_214552


namespace NUMINAMATH_CALUDE_cash_drawer_value_l2145_214510

/-- Calculates the total value of bills in a cash drawer given the total number of bills,
    the number of 5-dollar bills, and assuming the rest are 20-dollar bills. -/
def total_value (total_bills : ℕ) (five_dollar_bills : ℕ) : ℕ :=
  let twenty_dollar_bills := total_bills - five_dollar_bills
  5 * five_dollar_bills + 20 * twenty_dollar_bills

/-- Theorem stating that given 54 bills in total with 20 5-dollar bills,
    the total value is $780. -/
theorem cash_drawer_value :
  total_value 54 20 = 780 := by
  sorry

#eval total_value 54 20  -- Should output 780

end NUMINAMATH_CALUDE_cash_drawer_value_l2145_214510


namespace NUMINAMATH_CALUDE_pat_kate_ratio_l2145_214587

/-- Represents the hours charged by each person -/
structure ProjectHours where
  pat : ℝ
  kate : ℝ
  mark : ℝ

/-- Defines the conditions of the problem -/
def satisfiesConditions (h : ProjectHours) : Prop :=
  h.pat + h.kate + h.mark = 135 ∧
  ∃ r : ℝ, h.pat = r * h.kate ∧
  h.pat = (1/3) * h.mark ∧
  h.mark = h.kate + 75

/-- The main theorem to prove -/
theorem pat_kate_ratio (h : ProjectHours) 
  (hcond : satisfiesConditions h) : h.pat / h.kate = 2 := by
  sorry

end NUMINAMATH_CALUDE_pat_kate_ratio_l2145_214587


namespace NUMINAMATH_CALUDE_smallest_two_digit_multiple_plus_one_l2145_214558

theorem smallest_two_digit_multiple_plus_one : ∃ (n : ℕ), 
  (n ≥ 10 ∧ n < 100) ∧ 
  (∃ (k : ℕ), n = 2 * k + 1) ∧
  (∃ (k : ℕ), n = 3 * k + 1) ∧
  (∃ (k : ℕ), n = 4 * k + 1) ∧
  (∃ (k : ℕ), n = 5 * k + 1) ∧
  (∃ (k : ℕ), n = 6 * k + 1) ∧
  (∀ (m : ℕ), m < n → 
    (m < 10 ∨ m ≥ 100 ∨
    (∀ (k : ℕ), m ≠ 2 * k + 1) ∨
    (∀ (k : ℕ), m ≠ 3 * k + 1) ∨
    (∀ (k : ℕ), m ≠ 4 * k + 1) ∨
    (∀ (k : ℕ), m ≠ 5 * k + 1) ∨
    (∀ (k : ℕ), m ≠ 6 * k + 1))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_multiple_plus_one_l2145_214558


namespace NUMINAMATH_CALUDE_reflection_matrix_iff_l2145_214582

def reflection_matrix (a b : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![a, b],
    ![-3/4, 1/4]]

theorem reflection_matrix_iff (a b : ℚ) :
  (reflection_matrix a b)^2 = 1 ↔ a = -1/4 ∧ b = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_reflection_matrix_iff_l2145_214582


namespace NUMINAMATH_CALUDE_random_walk_properties_l2145_214556

/-- Represents a random walk on a line -/
structure RandomWalk where
  a : ℕ  -- number of steps to the right
  b : ℕ  -- number of steps to the left
  h : a > b

/-- The maximum possible range of the random walk -/
def max_range (w : RandomWalk) : ℕ := w.a

/-- The minimum possible range of the random walk -/
def min_range (w : RandomWalk) : ℕ := w.a - w.b

/-- The number of sequences achieving the maximum range -/
def max_range_sequences (w : RandomWalk) : ℕ := w.b + 1

theorem random_walk_properties (w : RandomWalk) :
  (max_range w = w.a) ∧
  (min_range w = w.a - w.b) ∧
  (max_range_sequences w = w.b + 1) := by
  sorry

end NUMINAMATH_CALUDE_random_walk_properties_l2145_214556


namespace NUMINAMATH_CALUDE_sphere_cylinder_volume_l2145_214540

/-- The volume of space inside a sphere and outside an inscribed right cylinder -/
theorem sphere_cylinder_volume (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 7) (h_cylinder : r_cylinder = 4) :
  let v_sphere := (4 / 3) * π * r_sphere^3
  let h_cylinder := Real.sqrt (r_sphere^2 - r_cylinder^2)
  let v_cylinder := π * r_cylinder^2 * h_cylinder
  v_sphere - v_cylinder = ((1372 - 48 * Real.sqrt 33) / 3) * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volume_l2145_214540


namespace NUMINAMATH_CALUDE_function_inequality_l2145_214544

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x : ℝ, f x > (deriv f) x) : 
  2023 * f (Real.log 2022) > 2022 * f (Real.log 2023) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2145_214544


namespace NUMINAMATH_CALUDE_job_size_ratio_l2145_214593

/-- Given two jobs with different numbers of workers and days, 
    prove that the ratio of work done in the new job to the original job is 3. -/
theorem job_size_ratio (original_workers original_days new_workers new_days : ℕ) 
    (h1 : original_workers = 250)
    (h2 : original_days = 16)
    (h3 : new_workers = 600)
    (h4 : new_days = 20) : 
    (new_workers * new_days) / (original_workers * original_days) = 3 := by
  sorry


end NUMINAMATH_CALUDE_job_size_ratio_l2145_214593


namespace NUMINAMATH_CALUDE_equation_transformation_correctness_l2145_214526

theorem equation_transformation_correctness :
  -- Option A is incorrect
  (∀ x : ℝ, 3 + x = 7 → x ≠ 7 + 3) ∧
  -- Option B is incorrect
  (∀ x : ℝ, 5 * x = -4 → x ≠ -5/4) ∧
  -- Option C is incorrect
  (∀ x : ℝ, 7/4 * x = 3 → x ≠ 3 * 7/4) ∧
  -- Option D is correct
  (∀ x : ℝ, -(x - 2) / 4 = 1 → -(x - 2) = 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_transformation_correctness_l2145_214526


namespace NUMINAMATH_CALUDE_equation_solution_l2145_214581

theorem equation_solution : 
  ∃! x : ℝ, x ≠ 1 ∧ x ≠ -6 ∧ (3*x + 6)/((x^2 + 5*x - 6)) = (3 - x)/(x - 1) ∧ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2145_214581


namespace NUMINAMATH_CALUDE_first_player_win_prob_correct_l2145_214551

/-- Represents the probability of winning for the first player in a three-player sequential game -/
def first_player_win_probability : ℚ :=
  729 / 5985

/-- The probability of a successful hit on any turn -/
def hit_probability : ℚ := 1 / 3

/-- The number of players in the game -/
def num_players : ℕ := 3

/-- Theorem stating the probability of the first player winning the game -/
theorem first_player_win_prob_correct :
  let p := hit_probability
  let n := num_players
  (p^2 * (1 - p^(2*n))⁻¹ : ℚ) = first_player_win_probability :=
by sorry

end NUMINAMATH_CALUDE_first_player_win_prob_correct_l2145_214551


namespace NUMINAMATH_CALUDE_fishing_line_section_length_l2145_214509

theorem fishing_line_section_length 
  (num_reels : ℕ) 
  (reel_length : ℝ) 
  (num_sections : ℕ) 
  (h1 : num_reels = 3) 
  (h2 : reel_length = 100) 
  (h3 : num_sections = 30) : 
  (num_reels * reel_length) / num_sections = 10 := by
  sorry

end NUMINAMATH_CALUDE_fishing_line_section_length_l2145_214509


namespace NUMINAMATH_CALUDE_vector_difference_l2145_214518

/-- Given two 2D vectors a and b, prove that their difference is (5, -3) -/
theorem vector_difference (a b : ℝ × ℝ) 
  (ha : a = (2, 1)) (hb : b = (-3, 4)) : 
  a - b = (5, -3) := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_l2145_214518


namespace NUMINAMATH_CALUDE_inequality_proof_l2145_214576

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (c/a)*(8*b+c) + (d/b)*(8*c+d) + (a/c)*(8*d+a) + (b/d)*(8*a+b) ≥ 9*(a+b+c+d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2145_214576


namespace NUMINAMATH_CALUDE_fried_chicken_cost_l2145_214513

/-- Calculates the cost of fried chicken given the total spent and other expenses at a club. -/
theorem fried_chicken_cost
  (entry_fee : ℚ)
  (drink_cost : ℚ)
  (friends : ℕ)
  (rounds : ℕ)
  (james_drinks : ℕ)
  (tip_rate : ℚ)
  (total_spent : ℚ)
  (h_entry_fee : entry_fee = 20)
  (h_drink_cost : drink_cost = 6)
  (h_friends : friends = 5)
  (h_rounds : rounds = 2)
  (h_james_drinks : james_drinks = 6)
  (h_tip_rate : tip_rate = 0.3)
  (h_total_spent : total_spent = 163)
  : ∃ (chicken_cost : ℚ),
    chicken_cost = 14 ∧
    total_spent = entry_fee +
                  (friends * rounds + james_drinks) * drink_cost +
                  chicken_cost +
                  ((friends * rounds + james_drinks) * drink_cost + chicken_cost) * tip_rate :=
by sorry


end NUMINAMATH_CALUDE_fried_chicken_cost_l2145_214513


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l2145_214542

-- Define a parallelogram
structure Parallelogram :=
  (A B C D : ℝ × ℝ)

-- Define the length function
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem parallelogram_side_length 
  (ABCD : Parallelogram) 
  (x : ℝ) 
  (h1 : length ABCD.A ABCD.B = x + 3)
  (h2 : length ABCD.B ABCD.C = x - 4)
  (h3 : length ABCD.C ABCD.D = 16) :
  length ABCD.A ABCD.D = 9 := by sorry

end NUMINAMATH_CALUDE_parallelogram_side_length_l2145_214542


namespace NUMINAMATH_CALUDE_area_scientific_notation_l2145_214555

/-- Represents the area in square meters -/
def area : ℝ := 216000

/-- Represents the coefficient in scientific notation -/
def coefficient : ℝ := 2.16

/-- Represents the exponent in scientific notation -/
def exponent : ℤ := 5

/-- Theorem stating that the area is equal to its scientific notation representation -/
theorem area_scientific_notation : area = coefficient * (10 : ℝ) ^ exponent := by sorry

end NUMINAMATH_CALUDE_area_scientific_notation_l2145_214555


namespace NUMINAMATH_CALUDE_jenny_research_time_l2145_214522

/-- Represents the time allocation for Jenny's school project -/
structure ProjectTime where
  total : ℕ
  proposal : ℕ
  report : ℕ

/-- Calculates the time spent on research given the project time allocation -/
def researchTime (pt : ProjectTime) : ℕ :=
  pt.total - pt.proposal - pt.report

/-- Theorem stating that Jenny spent 10 hours on research -/
theorem jenny_research_time :
  ∀ (pt : ProjectTime),
  pt.total = 20 ∧ pt.proposal = 2 ∧ pt.report = 8 →
  researchTime pt = 10 := by
  sorry

end NUMINAMATH_CALUDE_jenny_research_time_l2145_214522


namespace NUMINAMATH_CALUDE_expected_value_of_sum_l2145_214586

def marbles : Finset ℕ := {1, 2, 3, 4, 5, 6}

def combinations : Finset (Finset ℕ) :=
  marbles.powerset.filter (λ s => s.card = 3)

def sum_of_combination (c : Finset ℕ) : ℕ := c.sum id

def total_sum : ℕ := combinations.sum sum_of_combination

def num_combinations : ℕ := combinations.card

theorem expected_value_of_sum :
  (total_sum : ℚ) / num_combinations = 21/2 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_sum_l2145_214586


namespace NUMINAMATH_CALUDE_jordan_novels_count_l2145_214580

theorem jordan_novels_count :
  ∀ (j a : ℕ),
  a = j / 10 →
  j = a + 108 →
  j = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_jordan_novels_count_l2145_214580


namespace NUMINAMATH_CALUDE_cone_height_relationship_l2145_214504

/-- Represents the properties of a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Given two cones with equal volume and the second cone's radius 10% larger than the first,
    prove that the height of the first cone is 21% larger than the second -/
theorem cone_height_relationship (cone1 cone2 : Cone) 
  (h_volume : (1/3) * π * cone1.radius^2 * cone1.height = (1/3) * π * cone2.radius^2 * cone2.height)
  (h_radius : cone2.radius = 1.1 * cone1.radius) : 
  cone1.height = 1.21 * cone2.height := by
  sorry

end NUMINAMATH_CALUDE_cone_height_relationship_l2145_214504


namespace NUMINAMATH_CALUDE_balance_scale_l2145_214594

/-- The weight of the book that balances the scale -/
def book_weight : ℝ := 1.1

/-- The weight of the first item on the scale -/
def weight1 : ℝ := 0.5

/-- The weight of each of the two identical items on the scale -/
def weight2 : ℝ := 0.3

/-- The number of identical items with weight2 -/
def count2 : ℕ := 2

theorem balance_scale :
  book_weight = weight1 + count2 * weight2 := by sorry

end NUMINAMATH_CALUDE_balance_scale_l2145_214594


namespace NUMINAMATH_CALUDE_selene_total_cost_l2145_214525

/-- Calculate the total cost of Selene's purchase --/
def calculate_total_cost (camera_price : ℚ) (camera_count : ℕ) (frame_price : ℚ) (frame_count : ℕ)
  (card_price : ℚ) (card_count : ℕ) (camera_discount : ℚ) (frame_discount : ℚ) (card_discount : ℚ)
  (camera_frame_tax : ℚ) (card_tax : ℚ) : ℚ :=
  let camera_total := camera_price * camera_count
  let frame_total := frame_price * frame_count
  let card_total := card_price * card_count
  let camera_discounted := camera_total * (1 - camera_discount)
  let frame_discounted := frame_total * (1 - frame_discount)
  let card_discounted := card_total * (1 - card_discount)
  let camera_frame_subtotal := camera_discounted + frame_discounted
  let camera_frame_taxed := camera_frame_subtotal * (1 + camera_frame_tax)
  let card_taxed := card_discounted * (1 + card_tax)
  camera_frame_taxed + card_taxed

/-- Theorem stating that Selene's total cost is $691.72 --/
theorem selene_total_cost :
  calculate_total_cost 110 2 120 3 30 4 (7/100) (5/100) (10/100) (6/100) (4/100) = 69172/100 := by
  sorry

end NUMINAMATH_CALUDE_selene_total_cost_l2145_214525


namespace NUMINAMATH_CALUDE_tangent_function_property_l2145_214583

theorem tangent_function_property (ω : ℝ) (h1 : ω > 0) : 
  (∀ x : ℝ, ∃ y : ℝ, y > x ∧ Real.tan (ω * y) = Real.tan (ω * x) ∧ y - x = π / 4) → 
  Real.tan (ω * (π / 4)) = 0 := by
sorry

end NUMINAMATH_CALUDE_tangent_function_property_l2145_214583


namespace NUMINAMATH_CALUDE_function_and_range_l2145_214508

def f (a c : ℝ) (x : ℝ) : ℝ := a * x^3 + c * x

theorem function_and_range (a c : ℝ) (h1 : a > 0) :
  (∃ k, (3 * a + c) * k = -1 ∧ k ≠ 0) →
  (∀ x, 3 * a * x^2 + c ≥ -12) →
  (∃ x, 3 * a * x^2 + c = -12) →
  (f a c = fun x ↦ 2 * x^3 - 12 * x) ∧
  (∀ y ∈ Set.Icc (-8 * Real.sqrt 2) (8 * Real.sqrt 2), 
    ∃ x ∈ Set.Icc (-2) 2, f a c x = y) ∧
  (∀ x ∈ Set.Icc (-2) 2, f a c x ∈ Set.Icc (-8 * Real.sqrt 2) (8 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_function_and_range_l2145_214508


namespace NUMINAMATH_CALUDE_remainder_5031_div_28_l2145_214535

theorem remainder_5031_div_28 : 5031 % 28 = 19 := by
  sorry

end NUMINAMATH_CALUDE_remainder_5031_div_28_l2145_214535


namespace NUMINAMATH_CALUDE_unique_decreasing_term_l2145_214514

def a (n : ℕ+) : ℚ := 4 / (11 - 2 * n)

theorem unique_decreasing_term :
  ∃! (n : ℕ+), a (n + 1) < a n :=
by
  sorry

end NUMINAMATH_CALUDE_unique_decreasing_term_l2145_214514


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_p_necessary_not_sufficient_l2145_214538

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1
theorem range_of_x_when_a_is_one :
  ∀ x : ℝ, (p x 1 ∧ q x) ↔ x ∈ Set.Ioo 2 3 :=
sorry

-- Part 2
theorem range_of_a_when_p_necessary_not_sufficient :
  (∀ x : ℝ, q x → p x 1) ∧ 
  (∃ x : ℝ, p x 1 ∧ ¬q x) ↔
  1 ∈ Set.Ioo 1 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_p_necessary_not_sufficient_l2145_214538


namespace NUMINAMATH_CALUDE_log_78903_between_consecutive_integers_l2145_214568

theorem log_78903_between_consecutive_integers :
  ∃ (c d : ℤ), c + 1 = d ∧ (c : ℝ) < Real.log 78903 / Real.log 10 ∧ Real.log 78903 / Real.log 10 < (d : ℝ) → c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_78903_between_consecutive_integers_l2145_214568


namespace NUMINAMATH_CALUDE_three_digit_number_proof_l2145_214533

theorem three_digit_number_proof :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n / 100 = 1 ∧
  (n % 100 * 10 + 1) - n = 9 * (10 : ℝ) ∧ n = 121 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_number_proof_l2145_214533


namespace NUMINAMATH_CALUDE_orchard_tree_difference_l2145_214534

theorem orchard_tree_difference : 
  let ahmed_orange : ℕ := 8
  let hassan_apple : ℕ := 1
  let hassan_orange : ℕ := 2
  let ahmed_apple : ℕ := 4 * hassan_apple
  let ahmed_total : ℕ := ahmed_orange + ahmed_apple
  let hassan_total : ℕ := hassan_apple + hassan_orange
  ahmed_total - hassan_total = 9 := by
sorry

end NUMINAMATH_CALUDE_orchard_tree_difference_l2145_214534


namespace NUMINAMATH_CALUDE_sqrt_plus_one_iff_ax_plus_x_over_x_minus_one_l2145_214591

theorem sqrt_plus_one_iff_ax_plus_x_over_x_minus_one 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt a + 1 > b ↔ ∀ x > 1, a * x + x / (x - 1) > b := by
sorry

end NUMINAMATH_CALUDE_sqrt_plus_one_iff_ax_plus_x_over_x_minus_one_l2145_214591


namespace NUMINAMATH_CALUDE_undefined_slopes_parallel_l2145_214541

-- Define a type for lines
structure Line where
  slope : Option ℝ
  -- Other properties of a line could be added here if needed

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop :=
  (l1.slope = none ∧ l2.slope = none) ∨ (l1.slope ≠ none ∧ l2.slope ≠ none ∧ l1.slope = l2.slope)

-- Define what it means for two lines to be distinct
def distinct (l1 l2 : Line) : Prop :=
  l1 ≠ l2

-- Theorem statement
theorem undefined_slopes_parallel (l1 l2 : Line) :
  distinct l1 l2 → l1.slope = none → l2.slope = none → parallel l1 l2 :=
by
  sorry


end NUMINAMATH_CALUDE_undefined_slopes_parallel_l2145_214541


namespace NUMINAMATH_CALUDE_inverse_abs_is_geometric_sequence_preserving_l2145_214557

/-- A function is geometric sequence-preserving if it transforms any non-constant
    geometric sequence into another geometric sequence. -/
def IsGeometricSequencePreserving (f : ℝ → ℝ) : Prop :=
  ∀ (a : ℕ → ℝ) (q : ℝ),
    (∀ n, a n ≠ 0) →
    (∀ n, a (n + 1) = q * a n) →
    q ≠ 1 →
    ∃ r : ℝ, r ≠ 1 ∧ ∀ n, f (a (n + 1)) = r * f (a n)

/-- The function f(x) = 1/|x| is geometric sequence-preserving. -/
theorem inverse_abs_is_geometric_sequence_preserving :
    IsGeometricSequencePreserving (fun x ↦ 1 / |x|) := by
  sorry


end NUMINAMATH_CALUDE_inverse_abs_is_geometric_sequence_preserving_l2145_214557


namespace NUMINAMATH_CALUDE_pencil_length_l2145_214599

/-- Prove that given the conditions of the pen, rubber, and pencil lengths, the pencil is 12 cm long -/
theorem pencil_length (rubber pen pencil : ℝ) 
  (pen_rubber : pen = rubber + 3)
  (pencil_pen : pencil = pen + 2)
  (total_length : rubber + pen + pencil = 29) :
  pencil = 12 := by
  sorry

end NUMINAMATH_CALUDE_pencil_length_l2145_214599


namespace NUMINAMATH_CALUDE_parabola_chord_intersection_l2145_214536

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 16*x

-- Define a point on the parabola
def point_on_parabola (p : ℝ × ℝ) : Prop := parabola p.1 p.2

-- Define perpendicular vectors
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Theorem statement
theorem parabola_chord_intersection :
  ∀ (A B : ℝ × ℝ),
  point_on_parabola A →
  point_on_parabola B →
  perpendicular A B →
  ∃ (t : ℝ), A.1 = t * A.2 + 16 ∧ B.1 = t * B.2 + 16 :=
sorry

end NUMINAMATH_CALUDE_parabola_chord_intersection_l2145_214536


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2145_214548

/-- The minimum distance from the origin to a point on the line 3x + 4y - 20 = 0 is 4 -/
theorem min_distance_to_line : 
  ∀ a b : ℝ, (3 * a + 4 * b = 20) → (∀ x y : ℝ, (3 * x + 4 * y = 20) → (a^2 + b^2 ≤ x^2 + y^2)) → 
  Real.sqrt (a^2 + b^2) = 4 := by
sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l2145_214548


namespace NUMINAMATH_CALUDE_five_card_selection_with_constraints_l2145_214574

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- The number of suits in a standard deck -/
def number_of_suits : ℕ := 4

/-- The number of cards in each suit -/
def cards_per_suit : ℕ := 13

/-- The number of cards to be chosen -/
def cards_to_choose : ℕ := 5

/-- The number of cards that must share a suit -/
def cards_sharing_suit : ℕ := 2

/-- 
  The number of ways to choose 5 cards from a standard deck of 52 cards, 
  where exactly two cards share a suit and the remaining three are of different suits.
-/
theorem five_card_selection_with_constraints : 
  (number_of_suits) * 
  (Nat.choose cards_per_suit cards_sharing_suit) * 
  (Nat.choose (number_of_suits - 1) (cards_to_choose - cards_sharing_suit)) * 
  (cards_per_suit ^ (cards_to_choose - cards_sharing_suit)) = 684684 := by
  sorry

end NUMINAMATH_CALUDE_five_card_selection_with_constraints_l2145_214574


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2145_214512

theorem solution_set_inequality (x : ℝ) :
  (1/2 - x) * (x - 1/3) > 0 ↔ 1/3 < x ∧ x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2145_214512


namespace NUMINAMATH_CALUDE_shipping_cost_per_unit_l2145_214559

/-- A computer manufacturer produces electronic components with the following parameters:
  * Production cost per component: $80
  * Fixed monthly costs: $16,200
  * Monthly production and sales: 150 components
  * Lowest break-even selling price: $190 per component
  This theorem proves that the shipping cost per unit is $2. -/
theorem shipping_cost_per_unit (production_cost : ℝ) (fixed_costs : ℝ) (units : ℝ) (selling_price : ℝ)
  (h1 : production_cost = 80)
  (h2 : fixed_costs = 16200)
  (h3 : units = 150)
  (h4 : selling_price = 190) :
  ∃ (shipping_cost : ℝ), 
    units * (production_cost + shipping_cost) + fixed_costs = units * selling_price ∧ 
    shipping_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_shipping_cost_per_unit_l2145_214559


namespace NUMINAMATH_CALUDE_robert_coin_arrangements_l2145_214545

/-- Represents the number of distinguishable arrangements of coins -/
def coin_arrangements (gold_coins silver_coins : Nat) : Nat :=
  let total_coins := gold_coins + silver_coins
  let positions := Nat.choose total_coins gold_coins
  let orientations := 30  -- Simplified representation of valid orientations
  positions * orientations

/-- Theorem stating the number of distinguishable arrangements for the given problem -/
theorem robert_coin_arrangements :
  coin_arrangements 5 3 = 1680 := by
  sorry

end NUMINAMATH_CALUDE_robert_coin_arrangements_l2145_214545


namespace NUMINAMATH_CALUDE_eighteen_horses_walking_legs_l2145_214553

/-- Calculates the number of legs walking on the ground given the number of horses --/
def legsWalking (numHorses : ℕ) : ℕ :=
  let numMen := numHorses
  let numWalkingMen := numMen / 2
  let numWalkingHorses := numWalkingMen
  2 * numWalkingMen + 4 * numWalkingHorses

theorem eighteen_horses_walking_legs :
  legsWalking 18 = 54 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_horses_walking_legs_l2145_214553


namespace NUMINAMATH_CALUDE_y_is_75_percent_of_x_l2145_214531

-- Define variables
variable (x y z p : ℝ)

-- Define the theorem
theorem y_is_75_percent_of_x
  (h1 : 0.45 * z = 0.9 * y)
  (h2 : z = 1.5 * x)
  (h3 : y = p * x)
  : y = 0.75 * x :=
by sorry

end NUMINAMATH_CALUDE_y_is_75_percent_of_x_l2145_214531


namespace NUMINAMATH_CALUDE_ticket_cost_correct_l2145_214590

/-- The cost of one ticket for Sebastian's art exhibit -/
def ticket_cost : ℝ := 44

/-- The number of tickets Sebastian bought -/
def num_tickets : ℕ := 3

/-- The service fee for the online transaction -/
def service_fee : ℝ := 18

/-- The total amount Sebastian paid -/
def total_paid : ℝ := 150

/-- Theorem stating that the ticket cost is correct given the conditions -/
theorem ticket_cost_correct : 
  ticket_cost * num_tickets + service_fee = total_paid :=
by sorry

end NUMINAMATH_CALUDE_ticket_cost_correct_l2145_214590


namespace NUMINAMATH_CALUDE_function_inequality_l2145_214597

theorem function_inequality (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_cond : ∀ x : ℝ, (deriv^[2] f) x < 2 * f x) : 
  (Real.exp 4034 * f (-2017) > f 0) ∧ (f 2017 < Real.exp 4034 * f 0) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2145_214597


namespace NUMINAMATH_CALUDE_hat_cost_l2145_214567

/-- Given a sale of clothes where shirts cost $5 each, jeans cost $10 per pair,
    and the total cost for 3 shirts, 2 pairs of jeans, and 4 hats is $51,
    prove that each hat costs $4. -/
theorem hat_cost (shirt_cost jeans_cost total_cost : ℕ) (hat_cost : ℕ) :
  shirt_cost = 5 →
  jeans_cost = 10 →
  total_cost = 51 →
  3 * shirt_cost + 2 * jeans_cost + 4 * hat_cost = total_cost →
  hat_cost = 4 := by
  sorry

end NUMINAMATH_CALUDE_hat_cost_l2145_214567


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2145_214565

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  (1 : ℝ) * x^2 + 5*y - 4*x^2 - 3*y = -3*x^2 + 2*y := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  7*a + 3*(a - 3*b) - 2*(b - a) = 12*a - 11*b := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2145_214565


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2145_214527

def polynomial (x : ℝ) : ℝ :=
  -3 * (x^8 - 2*x^5 + 4*x^3 - 6) + 5 * (x^4 + 3*x^2 - 2*x + 1) - 4 * (x^6 - 5*x + 7)

theorem sum_of_coefficients :
  polynomial 1 = 12 :=
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2145_214527


namespace NUMINAMATH_CALUDE_smallest_of_three_consecutive_sum_90_l2145_214521

theorem smallest_of_three_consecutive_sum_90 (x y z : ℤ) :
  y = x + 1 ∧ z = y + 1 ∧ x + y + z = 90 → x = 29 := by
  sorry

end NUMINAMATH_CALUDE_smallest_of_three_consecutive_sum_90_l2145_214521


namespace NUMINAMATH_CALUDE_orange_stack_theorem_l2145_214560

/-- Calculates the number of oranges in a trapezoidal layer -/
def trapezoidalLayer (a b h : ℕ) : ℕ := (a + b) * h / 2

/-- Calculates the total number of oranges in the stack -/
def orangeStack (baseA baseB height : ℕ) : ℕ :=
  let rec stackLayers (a b h : ℕ) : ℕ :=
    if h = 0 then 0
    else trapezoidalLayer a b h + stackLayers (a - 1) (b - 1) (h - 1)
  stackLayers baseA baseB height

theorem orange_stack_theorem :
  orangeStack 7 5 6 = 90 := by sorry

end NUMINAMATH_CALUDE_orange_stack_theorem_l2145_214560


namespace NUMINAMATH_CALUDE_base7_divisibility_l2145_214577

/-- Converts a base 7 number to decimal --/
def base7ToDecimal (a b c d : ℕ) : ℕ := a * 7^3 + b * 7^2 + c * 7 + d

/-- Checks if a number is divisible by 29 --/
def isDivisibleBy29 (n : ℕ) : Prop := ∃ k : ℕ, n = 29 * k

theorem base7_divisibility :
  ∃! y : ℕ, y ≤ 6 ∧ isDivisibleBy29 (base7ToDecimal 2 y 6 3) :=
sorry

end NUMINAMATH_CALUDE_base7_divisibility_l2145_214577


namespace NUMINAMATH_CALUDE_only_seven_satisfies_inequality_l2145_214500

theorem only_seven_satisfies_inequality :
  ∃! (n : ℤ), (3 : ℚ) / 10 < (n : ℚ) / 20 ∧ (n : ℚ) / 20 < 2 / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_only_seven_satisfies_inequality_l2145_214500


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2145_214566

/-- Given an arithmetic sequence {a_n} with first term a₁ = -1 and common difference d = 2,
    prove that if a_{n-1} = 15, then n = 10. -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) (n : ℕ) :
  (∀ k, a (k + 1) = a k + 2) →  -- Common difference is 2
  a 1 = -1 →                    -- First term is -1
  a (n - 1) = 15 →              -- a_{n-1} = 15
  n = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2145_214566


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l2145_214561

theorem smallest_k_no_real_roots : ∃ k : ℤ, 
  (∀ x : ℝ, 3 * x * (k * x - 5) - x^2 + 8 - x^3 ≠ 0) ∧ 
  (∀ m : ℤ, m < k → ∃ x : ℝ, 3 * x * (m * x - 5) - x^2 + 8 - x^3 = 0) ∧
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l2145_214561


namespace NUMINAMATH_CALUDE_students_studying_both_subjects_difference_l2145_214532

theorem students_studying_both_subjects_difference (total : ℕ) 
  (math_min math_max science_min science_max : ℕ) : 
  total = 2500 →
  math_min = 1875 →
  math_max = 2000 →
  science_min = 875 →
  science_max = 1125 →
  let max_both := math_min + science_min - total
  let min_both := total - math_max - science_max
  max_both - min_both = 625 := by sorry

end NUMINAMATH_CALUDE_students_studying_both_subjects_difference_l2145_214532


namespace NUMINAMATH_CALUDE_congruence_solution_count_l2145_214563

theorem congruence_solution_count : 
  ∃! (x : ℕ), x > 0 ∧ x < 50 ∧ (x + 20) % 43 = 75 % 43 := by sorry

end NUMINAMATH_CALUDE_congruence_solution_count_l2145_214563


namespace NUMINAMATH_CALUDE_four_student_committees_from_six_l2145_214596

theorem four_student_committees_from_six (n k : ℕ) : n = 6 ∧ k = 4 → Nat.choose n k = 15 := by
  sorry

end NUMINAMATH_CALUDE_four_student_committees_from_six_l2145_214596


namespace NUMINAMATH_CALUDE_nina_jerome_age_ratio_l2145_214585

-- Define the ages as natural numbers
def leonard : ℕ := 6
def nina : ℕ := leonard + 4
def jerome : ℕ := 36 - nina - leonard

-- Theorem statement
theorem nina_jerome_age_ratio :
  nina * 2 = jerome :=
sorry

end NUMINAMATH_CALUDE_nina_jerome_age_ratio_l2145_214585


namespace NUMINAMATH_CALUDE_expression_evaluation_l2145_214528

theorem expression_evaluation : 2 - (-3)^2 - 4 - (-5) - 6^2 - (-7) = -35 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2145_214528


namespace NUMINAMATH_CALUDE_vodka_mixture_profit_l2145_214573

/-- Profit percentage of a mixture of two vodkas -/
def mixture_profit_percentage (profit1 profit2 : ℚ) (increase1 increase2 : ℚ) : ℚ :=
  ((profit1 * increase1 + profit2 * increase2) / 2)

theorem vodka_mixture_profit :
  let initial_profit1 : ℚ := 40 / 100
  let initial_profit2 : ℚ := 20 / 100
  let increase1 : ℚ := 4 / 3
  let increase2 : ℚ := 5 / 3
  mixture_profit_percentage initial_profit1 initial_profit2 increase1 increase2 = 13 / 30 := by
  sorry

#eval (13 / 30 : ℚ)

end NUMINAMATH_CALUDE_vodka_mixture_profit_l2145_214573


namespace NUMINAMATH_CALUDE_minimum_value_of_a_l2145_214562

theorem minimum_value_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/2), x^2 + a*x + 1 ≥ 0) → a ≥ -5/2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_a_l2145_214562


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l2145_214595

/-- A geometric sequence with positive terms and common ratio not equal to 1 -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ q > 0 ∧ q ≠ 1 ∧ ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_sum_inequality
  (a : ℕ → ℝ) (q : ℝ) (h : GeometricSequence a q) :
  a 1 + a 8 > a 4 + a 5 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l2145_214595


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2145_214569

def M : Set ℝ := {x | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {x | x < -5 ∨ x > 5}

theorem union_of_M_and_N : M ∪ N = {x | x < -5 ∨ x > -3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2145_214569


namespace NUMINAMATH_CALUDE_roses_per_set_l2145_214516

theorem roses_per_set (days_in_week : ℕ) (sets_per_day : ℕ) (total_roses : ℕ) :
  days_in_week = 7 →
  sets_per_day = 2 →
  total_roses = 168 →
  total_roses / (days_in_week * sets_per_day) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_roses_per_set_l2145_214516


namespace NUMINAMATH_CALUDE_equation_graph_is_two_parallel_lines_l2145_214502

-- Define the equation
def equation (x y : ℝ) : Prop := x^3 * (x + y + 2) = y^3 * (x + y + 2)

-- Define what it means for two lines to be parallel
def parallel (l₁ l₂ : ℝ → ℝ) : Prop := 
  ∃ (k : ℝ), ∀ x, l₂ x = l₁ x + k

-- Theorem statement
theorem equation_graph_is_two_parallel_lines :
  ∃ (l₁ l₂ : ℝ → ℝ), 
    (∀ x y, equation x y ↔ (y = l₁ x ∨ y = l₂ x)) ∧
    parallel l₁ l₂ :=
sorry

end NUMINAMATH_CALUDE_equation_graph_is_two_parallel_lines_l2145_214502


namespace NUMINAMATH_CALUDE_trapezoid_bases_count_l2145_214515

theorem trapezoid_bases_count : ∃! n : ℕ, 
  n = (Finset.filter (fun p : ℕ × ℕ => 
    10 ∣ p.1 ∧ 10 ∣ p.2 ∧ 
    (p.1 + p.2) * 30 = 1800 ∧ 
    0 < p.1 ∧ 0 < p.2) (Finset.product (Finset.range 181) (Finset.range 181))).card ∧
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_bases_count_l2145_214515


namespace NUMINAMATH_CALUDE_parallelogram_area_l2145_214529

/-- The area of a parallelogram with given side lengths and included angle -/
theorem parallelogram_area (a b : ℝ) (θ : Real) (ha : a = 32) (hb : b = 18) (hθ : θ = 75 * π / 180) :
  abs (a * b * Real.sin θ - 556.36) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2145_214529


namespace NUMINAMATH_CALUDE_price_per_working_game_l2145_214571

def total_games : ℕ := 10
def non_working_games : ℕ := 8
def total_earnings : ℕ := 12

theorem price_per_working_game :
  (total_earnings : ℚ) / (total_games - non_working_games) = 6 := by
  sorry

end NUMINAMATH_CALUDE_price_per_working_game_l2145_214571


namespace NUMINAMATH_CALUDE_simplify_polynomial_simplify_expression_l2145_214554

-- Problem 1
theorem simplify_polynomial (x : ℝ) :
  2*x^3 - 4*x^2 - 3*x - 2*x^2 - x^3 + 5*x - 7 = x^3 - 6*x^2 + 2*x - 7 := by
  sorry

-- Problem 2
theorem simplify_expression (m n : ℝ) :
  let A := 2*m^2 - m*n
  let B := m^2 + 2*m*n - 5
  4*A - 2*B = 6*m^2 - 8*m*n + 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_simplify_expression_l2145_214554


namespace NUMINAMATH_CALUDE_teacher_age_l2145_214517

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) :
  num_students = 20 →
  student_avg_age = 21 →
  new_avg_age = student_avg_age + 1 →
  (num_students + 1) * new_avg_age - num_students * student_avg_age = 42 :=
by sorry

end NUMINAMATH_CALUDE_teacher_age_l2145_214517


namespace NUMINAMATH_CALUDE_parabola_latus_rectum_p_l2145_214506

/-- A parabola with equation y^2 = 2px and latus rectum line x = -2 has p = 4 -/
theorem parabola_latus_rectum_p (y x p : ℝ) : 
  (y^2 = 2*p*x) →  -- Parabola equation
  (x = -2)      →  -- Latus rectum line equation
  p = 4         :=  -- Conclusion: p equals 4
by sorry

end NUMINAMATH_CALUDE_parabola_latus_rectum_p_l2145_214506


namespace NUMINAMATH_CALUDE_simplify_expression_l2145_214511

theorem simplify_expression (y : ℝ) : 3*y + 9*y^2 + 15 - (6 - 3*y - 9*y^2) = 18*y^2 + 6*y + 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2145_214511


namespace NUMINAMATH_CALUDE_smallest_n_less_than_one_hundredth_l2145_214530

/-- The probability of stopping after drawing exactly n marbles -/
def Q (n : ℕ+) : ℚ := 1 / (n * (n + 1))

/-- The number of boxes -/
def num_boxes : ℕ := 100

theorem smallest_n_less_than_one_hundredth :
  (∀ k : ℕ+, k < 10 → Q k ≥ 1/100) ∧
  (Q 10 < 1/100) ∧
  (∀ n : ℕ+, n ≤ num_boxes → Q n < 1/100 → n ≥ 10) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_less_than_one_hundredth_l2145_214530


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_11_l2145_214564

theorem smallest_four_digit_mod_11 :
  ∀ n : ℕ, 
    1000 ≤ n ∧ n < 10000 ∧ n % 11 = 2 → n ≥ 1003 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_11_l2145_214564


namespace NUMINAMATH_CALUDE_right_prism_surface_area_l2145_214550

/-- A right prism with an isosceles trapezoid base -/
structure RightPrism where
  /-- Length of parallel sides AB and CD -/
  ab_cd : ℝ
  /-- Length of side BC -/
  bc : ℝ
  /-- Length of side AD -/
  ad : ℝ
  /-- Area of the diagonal cross-section -/
  diagonal_area : ℝ
  /-- Condition: AB and CD are equal -/
  ab_eq_cd : ab_cd > 0
  /-- Condition: BC is positive -/
  bc_pos : bc > 0
  /-- Condition: AD is positive -/
  ad_pos : ad > 0
  /-- Condition: AD > BC (trapezoid property) -/
  ad_gt_bc : ad > bc
  /-- Condition: Diagonal area is positive -/
  diagonal_area_pos : diagonal_area > 0

/-- Total surface area of the right prism -/
def totalSurfaceArea (p : RightPrism) : ℝ :=
  sorry

/-- Theorem: The total surface area of the specified right prism is 906 -/
theorem right_prism_surface_area :
  ∀ (p : RightPrism),
    p.ab_cd = 13 ∧ p.bc = 11 ∧ p.ad = 21 ∧ p.diagonal_area = 180 →
    totalSurfaceArea p = 906 := by
  sorry

end NUMINAMATH_CALUDE_right_prism_surface_area_l2145_214550


namespace NUMINAMATH_CALUDE_activity_popularity_order_l2145_214543

-- Define the activities
inductive Activity
  | dodgeball
  | natureWalk
  | painting

-- Define the popularity fraction for each activity
def popularity (a : Activity) : Rat :=
  match a with
  | Activity.dodgeball => 13/40
  | Activity.natureWalk => 8/25
  | Activity.painting => 9/20

-- Define a function to compare two activities based on their popularity
def morePopular (a b : Activity) : Prop :=
  popularity a > popularity b

-- Theorem stating the correct order of activities
theorem activity_popularity_order :
  morePopular Activity.painting Activity.dodgeball ∧
  morePopular Activity.dodgeball Activity.natureWalk :=
by
  sorry

#check activity_popularity_order

end NUMINAMATH_CALUDE_activity_popularity_order_l2145_214543


namespace NUMINAMATH_CALUDE_fraction_sum_equation_l2145_214592

theorem fraction_sum_equation (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = 2 / 7 → 
  ((x = 4 ∧ y = 28) ∨ (x = 28 ∧ y = 4)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_equation_l2145_214592


namespace NUMINAMATH_CALUDE_expression_simplification_l2145_214539

theorem expression_simplification (d : ℝ) (h : d ≠ 0) :
  ∃ (a b c : ℤ),
    (15 : ℝ) * d + 16 + 17 * d^2 + (3 : ℝ) * d + 2 = (a : ℝ) * d + b + (c : ℝ) * d^2 ∧
    a + b + c = 53 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2145_214539


namespace NUMINAMATH_CALUDE_second_car_departure_time_l2145_214523

/-- Proves that the second car left 45 minutes after the first car --/
theorem second_car_departure_time (first_car_speed : ℝ) (trip_distance : ℝ) 
  (second_car_speed : ℝ) (time_difference : ℝ) : 
  first_car_speed = 30 →
  trip_distance = 80 →
  second_car_speed = 60 →
  time_difference = 1.5 →
  (time_difference - (first_car_speed * time_difference / second_car_speed)) * 60 = 45 := by
  sorry

#check second_car_departure_time

end NUMINAMATH_CALUDE_second_car_departure_time_l2145_214523


namespace NUMINAMATH_CALUDE_adam_ate_three_more_than_bill_l2145_214520

-- Define the number of pies eaten by each person
def sierra_pies : ℕ := 12
def total_pies : ℕ := 27

-- Define the relationships between the number of pies eaten
def bill_pies : ℕ := sierra_pies / 2
def adam_pies : ℕ := total_pies - sierra_pies - bill_pies

-- Theorem to prove
theorem adam_ate_three_more_than_bill :
  adam_pies = bill_pies + 3 := by
  sorry

end NUMINAMATH_CALUDE_adam_ate_three_more_than_bill_l2145_214520


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l2145_214578

-- Define the repeating decimals
def repeating_137 : ℚ := 137 / 999
def repeating_6 : ℚ := 2 / 3

-- Theorem statement
theorem product_of_repeating_decimals : 
  repeating_137 * repeating_6 = 274 / 2997 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l2145_214578


namespace NUMINAMATH_CALUDE_intersection_M_N_l2145_214549

def M : Set ℝ := {x | x ≥ -2}
def N : Set ℝ := {x | 1 < x ∧ x < 2}

theorem intersection_M_N : M ∩ N = {x | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2145_214549


namespace NUMINAMATH_CALUDE_square_inequality_negative_l2145_214570

theorem square_inequality_negative (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_negative_l2145_214570


namespace NUMINAMATH_CALUDE_fraction_equality_l2145_214507

theorem fraction_equality (m n s u : ℚ) 
  (h1 : m / n = 5 / 4) 
  (h2 : s / u = 8 / 15) : 
  (5 * m * s - 2 * n * u) / (7 * n * u - 10 * m * s) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2145_214507
