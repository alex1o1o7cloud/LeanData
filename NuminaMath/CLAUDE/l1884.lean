import Mathlib

namespace negation_of_exponential_inequality_l1884_188431

theorem negation_of_exponential_inequality :
  (¬ ∀ x : ℝ, x ≤ 0 → Real.exp x ≤ 1) ↔ (∃ x₀ : ℝ, x₀ ≤ 0 ∧ Real.exp x₀ > 1) := by
  sorry

end negation_of_exponential_inequality_l1884_188431


namespace profit_increase_l1884_188482

theorem profit_increase (initial_profit : ℝ) (x : ℝ) : 
  -- Conditions
  (initial_profit * (1 + x / 100) * 0.8 * 1.5 = initial_profit * 1.6200000000000001) →
  -- Conclusion
  x = 35 := by
sorry

end profit_increase_l1884_188482


namespace arithmetic_sequence_sum_twelve_l1884_188479

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def arithmetic_sum (a₁ d : ℚ) (n : ℕ) : ℚ := (n : ℚ) / 2 * (2 * a₁ + (n - 1) * d)

theorem arithmetic_sequence_sum_twelve (a₁ d : ℚ) :
  arithmetic_sequence a₁ d 5 = 1 →
  arithmetic_sequence a₁ d 17 = 18 →
  arithmetic_sum a₁ d 12 = 37.5 := by
  sorry


end arithmetic_sequence_sum_twelve_l1884_188479


namespace equation_is_union_of_twisted_cubics_twisted_cubic_is_parabola_like_equation_represents_two_parabolas_l1884_188421

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the equation y^6 - 9x^6 = 3y^3 - 1 -/
def equation (p : Point3D) : Prop :=
  p.y^6 - 9*p.x^6 = 3*p.y^3 - 1

/-- Represents a twisted cubic curve -/
def twistedCubic (a b c : ℝ) (p : Point3D) : Prop :=
  p.y^3 = a*p.x^3 + b*p.x + c

/-- The equation represents the union of two twisted cubic curves -/
theorem equation_is_union_of_twisted_cubics :
  ∀ p : Point3D, equation p ↔ (twistedCubic 3 0 1 p ∨ twistedCubic (-3) 0 1 p) :=
sorry

/-- Twisted cubic curves behave like parabolas -/
theorem twisted_cubic_is_parabola_like (a b c : ℝ) :
  ∀ p : Point3D, twistedCubic a b c p → (∃ q : Point3D, twistedCubic a b c q ∧ q ≠ p) :=
sorry

/-- The equation represents two parabola-like curves -/
theorem equation_represents_two_parabolas :
  ∃ (curve1 curve2 : Point3D → Prop),
    (∀ p : Point3D, equation p ↔ (curve1 p ∨ curve2 p)) ∧
    (∀ p : Point3D, curve1 p → (∃ q : Point3D, curve1 q ∧ q ≠ p)) ∧
    (∀ p : Point3D, curve2 p → (∃ q : Point3D, curve2 q ∧ q ≠ p)) :=
sorry

end equation_is_union_of_twisted_cubics_twisted_cubic_is_parabola_like_equation_represents_two_parabolas_l1884_188421


namespace coin_toss_problem_l1884_188406

theorem coin_toss_problem (p : ℝ) (n : ℕ) : 
  p = 1 / 2 →  -- Condition 1: Fair coin
  (1 / 2 : ℝ) ^ n = (1 / 8 : ℝ) →  -- Condition 2: Probability of same side is 0.125 (1/8)
  n = 3 := by  -- Question: Prove n = 3
sorry

end coin_toss_problem_l1884_188406


namespace inequality_proof_l1884_188477

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b * c * (a + b + c) = 3) :
  (a + b) * (b + c) * (c + a) ≥ 8 := by sorry

end inequality_proof_l1884_188477


namespace inequality_proof_l1884_188456

theorem inequality_proof (a b c x y z : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > 0) 
  (h4 : x > y) (h5 : y > z) (h6 : z > 0) : 
  (a^2 * x^2) / ((b*y + c*z) * (b*z + c*y)) + 
  (b^2 * y^2) / ((c*z + a*x) * (c*x + a*z)) + 
  (c^2 * z^2) / ((a*x + b*y) * (a*y + b*x)) ≥ 3/4 := by
sorry

end inequality_proof_l1884_188456


namespace wall_width_calculation_l1884_188478

/-- Calculates the width of a wall given brick dimensions and wall specifications -/
theorem wall_width_calculation (brick_length brick_width brick_height : ℝ)
  (wall_length wall_height : ℝ) (num_bricks : ℕ) :
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_height = 0.075 →
  wall_length = 29 →
  wall_height = 2 →
  num_bricks = 29000 →
  (brick_length * brick_width * brick_height * num_bricks) / (wall_length * wall_height) = 7.5 := by
  sorry

#check wall_width_calculation

end wall_width_calculation_l1884_188478


namespace restaurant_order_l1884_188426

theorem restaurant_order (b h p s : ℕ) : 
  b = 30 → b = 2 * h → p = h + 5 → s = 3 * p → b + h + p + s = 125 := by
  sorry

end restaurant_order_l1884_188426


namespace homework_completion_difference_l1884_188487

-- Define the efficiency rates and Tim's completion time
def samuel_efficiency : ℝ := 0.90
def sarah_efficiency : ℝ := 0.75
def tim_efficiency : ℝ := 0.80
def tim_completion_time : ℝ := 45

-- Define the theorem
theorem homework_completion_difference :
  let base_time := tim_completion_time / tim_efficiency
  let samuel_time := base_time / samuel_efficiency
  let sarah_time := base_time / sarah_efficiency
  sarah_time - samuel_time = 12.5 := by
sorry

end homework_completion_difference_l1884_188487


namespace parallelogram_area_28_32_l1884_188459

/-- The area of a parallelogram with given base and height -/
def parallelogramArea (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 28 cm and height 32 cm is 896 square centimeters -/
theorem parallelogram_area_28_32 : parallelogramArea 28 32 = 896 := by
  sorry

end parallelogram_area_28_32_l1884_188459


namespace train_length_l1884_188401

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 54 → time_s = 9 → speed_kmh * (1000 / 3600) * time_s = 135 := by
  sorry

#check train_length

end train_length_l1884_188401


namespace simplest_radical_form_among_options_l1884_188470

def is_simplest_radical_form (x : ℝ) : Prop :=
  ∃ n : ℕ, x = Real.sqrt n ∧ 
  (∀ m : ℕ, m ^ 2 ∣ n → m = 1) ∧
  (∀ a b : ℕ, n ≠ a / b)

theorem simplest_radical_form_among_options : 
  is_simplest_radical_form (Real.sqrt 10) ∧
  ¬is_simplest_radical_form (Real.sqrt 9) ∧
  ¬is_simplest_radical_form (Real.sqrt 20) ∧
  ¬is_simplest_radical_form (Real.sqrt (1/3)) :=
by sorry

end simplest_radical_form_among_options_l1884_188470


namespace candy_distribution_l1884_188441

def initial_candy : ℕ := 648
def sister_candy : ℕ := 48
def num_people : ℕ := 4
def bags_per_person : ℕ := 8

theorem candy_distribution (initial_candy sister_candy num_people bags_per_person : ℕ) :
  initial_candy = 648 →
  sister_candy = 48 →
  num_people = 4 →
  bags_per_person = 8 →
  (((initial_candy - sister_candy) / num_people) / bags_per_person : ℚ).floor = 18 := by
  sorry

end candy_distribution_l1884_188441


namespace secret_spread_day_l1884_188463

/-- The number of people who know the secret after n days -/
def secret_spread (n : ℕ) : ℕ :=
  (3^(n+1) - 1) / 2

/-- The day of the week, represented as a number from 0 (Sunday) to 6 (Saturday) -/
def day_of_week (n : ℕ) : Fin 7 :=
  n % 7

theorem secret_spread_day : ∃ n : ℕ, secret_spread n ≥ 3280 ∧ day_of_week n = 6 :=
sorry

end secret_spread_day_l1884_188463


namespace postage_for_420g_book_l1884_188418

/-- Calculates the postage cost for mailing a book in China. -/
def postage_cost (weight : ℕ) : ℚ :=
  let base_rate : ℚ := 7/10
  let additional_rate : ℚ := 4/10
  let base_weight : ℕ := 100
  let additional_weight := (weight - 1) / base_weight + 1
  base_rate + additional_rate * additional_weight

/-- Theorem stating that the postage cost for a 420g book is 2.3 yuan. -/
theorem postage_for_420g_book :
  postage_cost 420 = 23/10 := by sorry

end postage_for_420g_book_l1884_188418


namespace no_integer_solution_l1884_188472

theorem no_integer_solution : ¬ ∃ (a k : ℤ), 2 * a^2 - 7 * k + 2 = 0 := by
  sorry

end no_integer_solution_l1884_188472


namespace lcm_gcd_product_24_36_l1884_188424

theorem lcm_gcd_product_24_36 : Nat.lcm 24 36 * Nat.gcd 24 36 = 864 := by
  sorry

end lcm_gcd_product_24_36_l1884_188424


namespace reciprocal_opposite_equation_l1884_188497

theorem reciprocal_opposite_equation (m : ℝ) : (1 / (-0.5) = -(m + 4)) → m = 2 := by
  sorry

end reciprocal_opposite_equation_l1884_188497


namespace prob_ace_hearts_king_spades_l1884_188422

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards of each suit in a standard deck -/
def CardsPerSuit : ℕ := 13

/-- The probability of drawing a specific card from a standard deck -/
def ProbFirstCard : ℚ := 1 / StandardDeck

/-- The probability of drawing a specific card from the remaining deck after one card is drawn -/
def ProbSecondCard : ℚ := 1 / (StandardDeck - 1)

/-- The probability of drawing two specific cards in order from a standard deck -/
def ProbTwoSpecificCards : ℚ := ProbFirstCard * ProbSecondCard

theorem prob_ace_hearts_king_spades : 
  ProbTwoSpecificCards = 1 / 2652 := by sorry

end prob_ace_hearts_king_spades_l1884_188422


namespace intersection_of_M_and_N_l1884_188402

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {y | ∃ x, y = 2^x}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x | 0 < x ∧ x ≤ 2} :=
sorry

end intersection_of_M_and_N_l1884_188402


namespace floor_sqrt_80_l1884_188466

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end floor_sqrt_80_l1884_188466


namespace problem_statement_l1884_188444

def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem problem_statement (a : ℝ) :
  (p a ↔ a ≤ 1) ∧
  ((p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a > 1 ∨ (-2 < a ∧ a < 1)) :=
sorry

end problem_statement_l1884_188444


namespace total_watermelon_seeds_l1884_188490

theorem total_watermelon_seeds (bom gwi yeon : ℕ) : 
  bom = 300 →
  gwi = bom + 40 →
  yeon = 3 * gwi →
  bom + gwi + yeon = 1660 := by
sorry

end total_watermelon_seeds_l1884_188490


namespace zoo_population_is_90_l1884_188400

/-- Calculates the final number of animals in a zoo after a series of events --/
def final_zoo_population (initial_animals : ℕ) 
                         (gorillas_sent : ℕ) 
                         (hippo_adopted : ℕ) 
                         (rhinos_added : ℕ) 
                         (lion_cubs_born : ℕ) : ℕ :=
  initial_animals - gorillas_sent + hippo_adopted + rhinos_added + lion_cubs_born + 2 * lion_cubs_born

/-- Theorem stating that the final zoo population is 90 given the specific events --/
theorem zoo_population_is_90 : 
  final_zoo_population 68 6 1 3 8 = 90 := by
  sorry


end zoo_population_is_90_l1884_188400


namespace emily_walks_farther_l1884_188473

def troy_base_distance : ℕ := 75
def emily_base_distance : ℕ := 98

def troy_detours : List ℕ := [15, 20, 10, 0, 5]
def emily_detours : List ℕ := [10, 25, 10, 15, 10]

def total_distance (base : ℕ) (detours : List ℕ) : ℕ :=
  (base * 10 + detours.sum) * 2

theorem emily_walks_farther :
  total_distance emily_base_distance emily_detours -
  total_distance troy_base_distance troy_detours = 270 := by
  sorry

end emily_walks_farther_l1884_188473


namespace product_of_squares_l1884_188493

theorem product_of_squares (x : ℝ) :
  (2024 - x)^2 + (2022 - x)^2 = 4038 →
  (2024 - x) * (2022 - x) = 2017 := by
sorry

end product_of_squares_l1884_188493


namespace max_value_of_f_l1884_188415

theorem max_value_of_f (x : ℝ) (h : x < 3) : 
  (4 / (x - 3) + x) ≤ -1 ∧ ∃ y < 3, 4 / (y - 3) + y = -1 :=
sorry

end max_value_of_f_l1884_188415


namespace sum_abc_equals_51_l1884_188486

theorem sum_abc_equals_51 (a b c : ℕ+) 
  (h1 : a * b + c = 50)
  (h2 : a * c + b = 50)
  (h3 : b * c + a = 50) : 
  a + b + c = 51 := by
  sorry

end sum_abc_equals_51_l1884_188486


namespace tangent_slope_at_zero_l1884_188438

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp x

theorem tangent_slope_at_zero : 
  (deriv f) 0 = 2 :=
sorry

end tangent_slope_at_zero_l1884_188438


namespace nates_dropped_matches_l1884_188412

/-- Proves that Nate dropped 10 matches in the creek given the initial conditions. -/
theorem nates_dropped_matches (initial_matches : ℕ) (remaining_matches : ℕ) (dropped_matches : ℕ) :
  initial_matches = 70 →
  remaining_matches = 40 →
  initial_matches - remaining_matches = dropped_matches + 2 * dropped_matches →
  dropped_matches = 10 := by
sorry

end nates_dropped_matches_l1884_188412


namespace alvin_marbles_l1884_188439

theorem alvin_marbles (initial : ℕ) (lost : ℕ) (won : ℕ) (final : ℕ) : 
  initial = 57 → lost = 18 → won = 25 → final = 64 →
  final = initial - lost + won :=
by sorry

end alvin_marbles_l1884_188439


namespace tank_m_height_is_10_l1884_188440

/-- Tank M is a right circular cylinder with circumference 8 meters -/
def tank_m_circumference : ℝ := 8

/-- Tank B is a right circular cylinder with height 8 meters and circumference 10 meters -/
def tank_b_height : ℝ := 8
def tank_b_circumference : ℝ := 10

/-- The capacity of tank M is 80% of the capacity of tank B -/
def capacity_ratio : ℝ := 0.8

/-- The height of tank M -/
def tank_m_height : ℝ := 10

theorem tank_m_height_is_10 :
  tank_m_height = 10 := by sorry

end tank_m_height_is_10_l1884_188440


namespace solve_equation_l1884_188474

theorem solve_equation : ∃! x : ℝ, (x - 4)^4 = (1/16)⁻¹ := by
  use 6
  sorry

end solve_equation_l1884_188474


namespace power_function_monotonicity_l1884_188496

/-- A power function is monotonically increasing on (0, +∞) -/
def is_monotone_increasing (m : ℝ) : Prop :=
  ∀ x > 0, ∀ y > 0, x < y → (m^2 - m - 1) * x^m < (m^2 - m - 1) * y^m

/-- The condition |m-2| < 1 -/
def condition_q (m : ℝ) : Prop := |m - 2| < 1

theorem power_function_monotonicity (m : ℝ) :
  (is_monotone_increasing m → condition_q m) ∧
  ¬(condition_q m → is_monotone_increasing m) :=
sorry

end power_function_monotonicity_l1884_188496


namespace winning_strategy_l1884_188425

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Predicate to check if a number is a Fibonacci number -/
def isFibonacci (n : ℕ) : Prop :=
  ∃ k, fib k = n

/-- Game rules -/
structure GameRules where
  n : ℕ
  n_gt_one : n > 1
  first_turn_not_all : ∀ (first_pick : ℕ), first_pick < n
  subsequent_turns : ∀ (prev_pick current_pick : ℕ), current_pick ≤ 2 * prev_pick

/-- Winning strategy for Player A -/
def playerAWins (rules : GameRules) : Prop :=
  ¬(isFibonacci rules.n)

/-- Main theorem: Player A has a winning strategy iff n is not a Fibonacci number -/
theorem winning_strategy (rules : GameRules) :
  playerAWins rules ↔ ¬(isFibonacci rules.n) := by sorry

end winning_strategy_l1884_188425


namespace equal_area_rectangles_l1884_188483

/-- Given two rectangles with equal area, where one rectangle has dimensions 12 inches by 15 inches,
    and the other rectangle has a length of 6 inches, prove that the width of the second rectangle is 30 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_length jordan_width : ℝ)
    (h1 : carol_length = 12)
    (h2 : carol_width = 15)
    (h3 : jordan_length = 6)
    (h4 : carol_length * carol_width = jordan_length * jordan_width) :
  jordan_width = 30 := by
  sorry

end equal_area_rectangles_l1884_188483


namespace line_intersection_plane_intersection_l1884_188457

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the properties
variable (lies_in : Line → Plane → Prop)
variable (intersects_line : Line → Line → Prop)
variable (intersects_plane : Plane → Plane → Prop)

-- State the theorem
theorem line_intersection_plane_intersection 
  (a b : Line) (α β : Plane) 
  (ha : lies_in a α) (hb : lies_in b β) :
  (∀ (a b : Line) (α β : Plane), lies_in a α → lies_in b β → 
    intersects_line a b → intersects_plane α β) ∧ 
  (∃ (a b : Line) (α β : Plane), lies_in a α ∧ lies_in b β ∧ 
    intersects_plane α β ∧ ¬intersects_line a b) :=
sorry

end line_intersection_plane_intersection_l1884_188457


namespace sector_arc_length_l1884_188452

/-- The length of an arc in a circular sector with given central angle and radius -/
def arc_length (central_angle : Real) (radius : Real) : Real :=
  central_angle * radius

theorem sector_arc_length :
  let central_angle : Real := π / 3
  let radius : Real := 2
  arc_length central_angle radius = 2 * π / 3 := by
  sorry

end sector_arc_length_l1884_188452


namespace stamps_per_binder_l1884_188489

-- Define the number of notebooks and stamps per notebook
def num_notebooks : ℕ := 4
def stamps_per_notebook : ℕ := 20

-- Define the number of binders
def num_binders : ℕ := 2

-- Define the fraction of stamps kept
def fraction_kept : ℚ := 1/4

-- Define the number of stamps given away
def stamps_given_away : ℕ := 135

-- Theorem to prove
theorem stamps_per_binder :
  ∃ (x : ℕ), 
    (3/4 : ℚ) * (num_notebooks * stamps_per_notebook + num_binders * x) = stamps_given_away ∧
    x = 50 := by
  sorry

end stamps_per_binder_l1884_188489


namespace vector_properties_l1884_188480

/-- Given vectors a and b, prove that the projection of a onto b is equal to b,
    and that (a - b) is perpendicular to b. -/
theorem vector_properties (a b : ℝ × ℝ) 
    (ha : a = (2, 0)) (hb : b = (1, 1)) : 
    (((a • b) / (b • b)) • b = b) ∧ ((a - b) • b = 0) := by
  sorry

end vector_properties_l1884_188480


namespace mother_younger_by_two_l1884_188491

/-- A family consisting of a father, mother, brother, sister, and Kaydence. -/
structure Family where
  total_age : ℕ
  father_age : ℕ
  brother_age : ℕ
  sister_age : ℕ
  kaydence_age : ℕ

/-- The age difference between the father and mother in the family. -/
def age_difference (f : Family) : ℕ :=
  f.father_age - (f.total_age - (f.father_age + f.brother_age + f.sister_age + f.kaydence_age))

/-- Theorem stating the age difference between the father and mother is 2 years. -/
theorem mother_younger_by_two (f : Family) 
    (h1 : f.total_age = 200)
    (h2 : f.father_age = 60)
    (h3 : f.brother_age = f.father_age / 2)
    (h4 : f.sister_age = 40)
    (h5 : f.kaydence_age = 12) :
    age_difference f = 2 := by
  sorry

end mother_younger_by_two_l1884_188491


namespace levi_initial_score_proof_l1884_188495

/-- Levi's initial score in a basketball game with his brother -/
def levi_initial_score : ℕ := 8

/-- Levi's brother's initial score -/
def brother_initial_score : ℕ := 12

/-- The minimum difference in scores Levi wants to achieve -/
def score_difference : ℕ := 5

/-- Additional scores by Levi's brother -/
def brother_additional_score : ℕ := 3

/-- Additional scores Levi needs to reach his goal -/
def levi_additional_score : ℕ := 12

theorem levi_initial_score_proof :
  levi_initial_score = 8 ∧
  brother_initial_score = 12 ∧
  score_difference = 5 ∧
  brother_additional_score = 3 ∧
  levi_additional_score = 12 ∧
  levi_initial_score + levi_additional_score = 
    brother_initial_score + brother_additional_score + score_difference :=
by sorry

end levi_initial_score_proof_l1884_188495


namespace square_plot_area_l1884_188494

/-- Proves that a square plot with a fence costing Rs. 58 per foot and Rs. 3944 in total has an area of 289 square feet. -/
theorem square_plot_area (price_per_foot : ℝ) (total_cost : ℝ) :
  price_per_foot = 58 →
  total_cost = 3944 →
  ∃ (side_length : ℝ),
    4 * side_length * price_per_foot = total_cost ∧
    side_length^2 = 289 :=
by sorry


end square_plot_area_l1884_188494


namespace university_application_options_l1884_188433

theorem university_application_options : 
  let total_universities : ℕ := 6
  let applications_needed : ℕ := 3
  let universities_with_coinciding_exams : ℕ := 2
  
  (Nat.choose (total_universities - universities_with_coinciding_exams) applications_needed) +
  (universities_with_coinciding_exams * Nat.choose (total_universities - universities_with_coinciding_exams) (applications_needed - 1)) = 16 := by
  sorry

end university_application_options_l1884_188433


namespace multiply_subtract_divide_l1884_188417

theorem multiply_subtract_divide : 4 * 6 * 8 - 24 / 4 = 186 := by
  sorry

end multiply_subtract_divide_l1884_188417


namespace germs_left_percentage_l1884_188423

/-- Represents the effectiveness of four sanitizer sprays and their overlaps -/
structure SanitizerSprays where
  /-- Kill rates for each spray -/
  spray1 : ℝ
  spray2 : ℝ
  spray3 : ℝ
  spray4 : ℝ
  /-- Two-way overlaps between sprays -/
  overlap12 : ℝ
  overlap23 : ℝ
  overlap34 : ℝ
  overlap13 : ℝ
  overlap14 : ℝ
  overlap24 : ℝ
  /-- Three-way overlaps between sprays -/
  overlap123 : ℝ
  overlap234 : ℝ

/-- Calculates the percentage of germs left after applying all sprays -/
def germsLeft (s : SanitizerSprays) : ℝ :=
  100 - (s.spray1 + s.spray2 + s.spray3 + s.spray4 - 
         (s.overlap12 + s.overlap23 + s.overlap34 + s.overlap13 + s.overlap14 + s.overlap24) -
         (s.overlap123 + s.overlap234))

/-- Theorem stating that for the given spray effectiveness and overlaps, 13.8% of germs are left -/
theorem germs_left_percentage (s : SanitizerSprays) 
  (h1 : s.spray1 = 50) (h2 : s.spray2 = 35) (h3 : s.spray3 = 20) (h4 : s.spray4 = 10)
  (h5 : s.overlap12 = 10) (h6 : s.overlap23 = 7) (h7 : s.overlap34 = 5)
  (h8 : s.overlap13 = 3) (h9 : s.overlap14 = 2) (h10 : s.overlap24 = 1)
  (h11 : s.overlap123 = 0.5) (h12 : s.overlap234 = 0.3) :
  germsLeft s = 13.8 := by
  sorry


end germs_left_percentage_l1884_188423


namespace quadratic_inequality_solution_l1884_188455

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 + 8 * x + 5

-- Define the solution set
def solution_set : Set ℝ := {x | x < -1 ∨ x > 5/3}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x < 0} = solution_set :=
by sorry

end quadratic_inequality_solution_l1884_188455


namespace supermarket_discount_items_l1884_188419

/-- Represents the supermarket's inventory and pricing --/
structure Supermarket where
  total_cost : ℝ
  items_a : ℕ
  items_b : ℕ
  cost_a : ℝ
  cost_b : ℝ
  price_a : ℝ
  price_b : ℝ

/-- Represents the second purchase scenario --/
structure SecondPurchase where
  sm : Supermarket
  items_b_new : ℕ
  discount_price_b : ℝ
  total_profit : ℝ

/-- The main theorem to prove --/
theorem supermarket_discount_items (sm : Supermarket) (sp : SecondPurchase) :
  sm.total_cost = 6000 ∧
  sm.items_a = 2 * sm.items_b - 30 ∧
  sm.cost_a = 22 ∧
  sm.cost_b = 30 ∧
  sm.price_a = 29 ∧
  sm.price_b = 40 ∧
  sp.sm = sm ∧
  sp.items_b_new = 3 * sm.items_b ∧
  sp.discount_price_b = sm.price_b / 2 ∧
  sp.total_profit = 2350 →
  ∃ (discount_items : ℕ), 
    discount_items = 70 ∧
    (sm.price_a - sm.cost_a) * sm.items_a + 
    (sm.price_b - sm.cost_b) * (sp.items_b_new - discount_items) +
    (sp.discount_price_b - sm.cost_b) * discount_items = sp.total_profit :=
by sorry


end supermarket_discount_items_l1884_188419


namespace unique_solution_sum_in_base7_l1884_188448

/-- Represents a digit in base 7 --/
def Digit7 := Fin 7

/-- Addition in base 7 --/
def add7 (a b : Digit7) : Digit7 × Bool :=
  let sum := a.val + b.val
  (⟨sum % 7, by sorry⟩, sum ≥ 7)

/-- Represents the equation in base 7 --/
def equation (A B C : Digit7) : Prop :=
  ∃ (carry1 carry2 : Bool),
    let (units, carry1) := add7 B C
    let (tens, carry2) := add7 A B
    units = A ∧
    (if carry1 then add7 (⟨1, by sorry⟩) tens else (tens, false)).1 = C ∧
    (if carry2 then add7 (⟨1, by sorry⟩) A else (A, false)).1 = A

theorem unique_solution :
  ∃! (A B C : Digit7),
    A.val ≠ 0 ∧ B.val ≠ 0 ∧ C.val ≠ 0 ∧
    A.val ≠ B.val ∧ A.val ≠ C.val ∧ B.val ≠ C.val ∧
    equation A B C ∧
    A.val = 6 ∧ B.val = 3 ∧ C.val = 5 :=
  sorry

theorem sum_in_base7 (A B C : Digit7) 
  (h : A.val = 6 ∧ B.val = 3 ∧ C.val = 5) :
  (A.val + B.val + C.val : ℕ) % 49 = 20 :=
  sorry

end unique_solution_sum_in_base7_l1884_188448


namespace apps_left_l1884_188458

/-- 
Given that Dave had 23 apps initially and deleted 18 apps, 
prove that he has 5 apps left.
-/
theorem apps_left (initial_apps : ℕ) (deleted_apps : ℕ) (apps_left : ℕ) : 
  initial_apps = 23 → deleted_apps = 18 → apps_left = initial_apps - deleted_apps → apps_left = 5 := by
  sorry

end apps_left_l1884_188458


namespace wax_left_after_detailing_l1884_188453

/-- The amount of wax needed to detail Kellan's car in ounces. -/
def car_wax : ℕ := 3

/-- The amount of wax needed to detail Kellan's SUV in ounces. -/
def suv_wax : ℕ := 4

/-- The amount of wax in the bottle Kellan bought in ounces. -/
def bought_wax : ℕ := 11

/-- The amount of wax Kellan spilled in ounces. -/
def spilled_wax : ℕ := 2

/-- The theorem states that given the above conditions, 
    the amount of wax Kellan has left after waxing his car and SUV is 2 ounces. -/
theorem wax_left_after_detailing : 
  bought_wax - spilled_wax - (car_wax + suv_wax) = 2 := by
  sorry

end wax_left_after_detailing_l1884_188453


namespace smallest_upper_bound_l1884_188492

-- Define the set of natural numbers
def N : Set ℕ := Set.univ

-- Define the set of real numbers
def R : Set ℝ := Set.univ

-- Define the set S of functions f: N → R satisfying the given conditions
def S : Set (ℕ → ℝ) := {f | f 1 = 2 ∧ ∀ n, f (n + 1) ≥ f n ∧ f n ≥ (n / (n + 1 : ℝ)) * f (2 * n)}

-- State the theorem
theorem smallest_upper_bound :
  ∃ M : ℕ, (∀ f ∈ S, ∀ n : ℕ, f n < M) ∧
  (∀ M' : ℕ, M' < M → ∃ f ∈ S, ∃ n : ℕ, f n ≥ M') :=
sorry

end smallest_upper_bound_l1884_188492


namespace trig_expression_equals_one_l1884_188447

theorem trig_expression_equals_one : 
  (2 * Real.sin (46 * π / 180) - Real.sqrt 3 * Real.cos (74 * π / 180)) / Real.cos (16 * π / 180) = 1 := by
  sorry

end trig_expression_equals_one_l1884_188447


namespace triangle_problem_l1884_188498

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem to be proved -/
theorem triangle_problem (t : Triangle) 
  (h1 : cos (2 * t.A) - 3 * cos (t.B + t.C) = 1)
  (h2 : 1/2 * t.b * t.c * sin t.A = 5 * sqrt 3)
  (h3 : t.b = 5) :
  t.A = π/3 ∧ t.a = sqrt 21 := by
  sorry


end triangle_problem_l1884_188498


namespace writable_13121_not_writable_12131_l1884_188476

/-- A number that can be written on the blackboard -/
def Writable (n : ℕ) : Prop :=
  ∃ x y : ℕ, n + 1 = 2^x * 3^y

/-- The rule for writing new numbers on the blackboard -/
axiom write_rule {a b : ℕ} (ha : Writable a) (hb : Writable b) : Writable (a * b + a + b)

/-- 1 is initially on the blackboard -/
axiom writable_one : Writable 1

/-- 2 is initially on the blackboard -/
axiom writable_two : Writable 2

/-- Theorem: 13121 can be written on the blackboard -/
theorem writable_13121 : Writable 13121 :=
  sorry

/-- Theorem: 12131 cannot be written on the blackboard -/
theorem not_writable_12131 : ¬ Writable 12131 :=
  sorry

end writable_13121_not_writable_12131_l1884_188476


namespace probability_three_red_cards_l1884_188467

/-- The probability of drawing three red cards in succession from a shuffled standard deck --/
theorem probability_three_red_cards (total_cards : ℕ) (red_cards : ℕ) 
  (h1 : total_cards = 52)
  (h2 : red_cards = 26) : 
  (red_cards * (red_cards - 1) * (red_cards - 2)) / 
  (total_cards * (total_cards - 1) * (total_cards - 2)) = 4 / 17 := by
sorry

end probability_three_red_cards_l1884_188467


namespace factorization_of_a_squared_plus_3a_l1884_188409

theorem factorization_of_a_squared_plus_3a (a : ℝ) : a^2 + 3*a = a*(a+3) := by
  sorry

end factorization_of_a_squared_plus_3a_l1884_188409


namespace equation_solution_l1884_188428

theorem equation_solution : 
  ∃! x : ℚ, 7 * (2 * x + 3) - 5 = -3 * (2 - 5 * x) + 2 * x ∧ x = 22 / 3 := by
  sorry

end equation_solution_l1884_188428


namespace jessie_weight_loss_l1884_188485

/-- Jessie's weight loss problem -/
theorem jessie_weight_loss (current_weight lost_weight : ℕ) 
  (h1 : current_weight = 34)
  (h2 : lost_weight = 35) : 
  current_weight + lost_weight = 69 := by
  sorry

end jessie_weight_loss_l1884_188485


namespace monkey_peaches_l1884_188449

theorem monkey_peaches : ∃ (n : ℕ) (m : ℕ), 
  n > 0 ∧ 
  n % 3 = 0 ∧ 
  m % n = 27 ∧ 
  (m - 27) / n = 5 ∧ 
  ∃ (x : ℕ), 0 < x ∧ x < 7 ∧ m = 7 * n - x ∧
  m = 102 := by
  sorry

end monkey_peaches_l1884_188449


namespace length_of_BC_l1884_188443

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def validTriangle (t : Triangle) : Prop :=
  -- A is at the origin
  t.A = (0, 0) ∧
  -- All vertices lie on the parabola
  t.A.2 = parabola t.A.1 ∧
  t.B.2 = parabola t.B.1 ∧
  t.C.2 = parabola t.C.1 ∧
  -- BC is parallel to x-axis
  t.B.2 = t.C.2 ∧
  -- Area of the triangle is 128
  abs ((t.B.1 - t.A.1) * (t.C.2 - t.A.2) - (t.C.1 - t.A.1) * (t.B.2 - t.A.2)) / 2 = 128

-- Theorem statement
theorem length_of_BC (t : Triangle) (h : validTriangle t) : 
  Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2) = 8 := by
  sorry

end length_of_BC_l1884_188443


namespace correct_group_formations_l1884_188432

/-- The number of ways to form n groups of 2 from 2n soldiers -/
def groupFormations (n : ℕ) : ℕ × ℕ :=
  (Nat.factorial (2*n) / Nat.factorial n,
   Nat.factorial (2*n) / (2^n * Nat.factorial n))

/-- Theorem stating the correct number of group formations for both cases -/
theorem correct_group_formations (n : ℕ) :
  groupFormations n = (Nat.factorial (2*n) / Nat.factorial n,
                       Nat.factorial (2*n) / (2^n * Nat.factorial n)) :=
by sorry

end correct_group_formations_l1884_188432


namespace combine_like_terms_l1884_188499

theorem combine_like_terms (a b : ℝ) :
  4 * (a - b)^2 - 6 * (a - b)^2 + 8 * (a - b)^2 = 6 * (a - b)^2 := by sorry

end combine_like_terms_l1884_188499


namespace gcd_105_90_l1884_188462

theorem gcd_105_90 : Nat.gcd 105 90 = 15 := by
  sorry

end gcd_105_90_l1884_188462


namespace tourists_distribution_eight_l1884_188481

/-- The number of ways to distribute n tourists between 2 guides,
    where each guide must have at least one tourist -/
def distribute_tourists (n : ℕ) : ℕ :=
  2^n - 2

theorem tourists_distribution_eight :
  distribute_tourists 8 = 254 :=
sorry

end tourists_distribution_eight_l1884_188481


namespace fifth_number_in_row_51_l1884_188408

/-- Pascal's triangle binomial coefficient -/
def pascal (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- The number of elements in a row of Pascal's triangle -/
def row_size (row : ℕ) : ℕ :=
  row + 1

theorem fifth_number_in_row_51 :
  pascal 50 4 = 22050 :=
by
  sorry

end fifth_number_in_row_51_l1884_188408


namespace complex_number_in_fourth_quadrant_l1884_188461

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 - I) / (3 + 4*I)
  (z.re > 0 ∧ z.im < 0) := by sorry

end complex_number_in_fourth_quadrant_l1884_188461


namespace cone_height_increase_l1884_188404

theorem cone_height_increase (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let V := (1/3) * Real.pi * r^2 * h
  let V' := 2.3 * V
  ∃ x : ℝ, V' = (1/3) * Real.pi * r^2 * (h * (1 + x/100)) → x = 130 := by
  sorry

end cone_height_increase_l1884_188404


namespace arithmetic_simplification_l1884_188436

theorem arithmetic_simplification :
  -3 + (-9) + 10 - (-18) = 16 := by
  sorry

end arithmetic_simplification_l1884_188436


namespace derivative_of_f_l1884_188484

-- Define the function f
def f (x : ℝ) : ℝ := (x + 1) * (x^2 - x + 1)

-- State the theorem
theorem derivative_of_f : 
  deriv f = λ x => 3 * x^2 := by sorry

end derivative_of_f_l1884_188484


namespace equation_solution_l1884_188442

theorem equation_solution : ∃ x : ℚ, (3/x + (1/x) / (5/x) + 1/(2*x) = 5/4) ∧ (x = 10/3) := by
  sorry

end equation_solution_l1884_188442


namespace arithmetic_sequence_middle_term_l1884_188430

theorem arithmetic_sequence_middle_term (a₁ a₃ z : ℤ) : 
  a₁ = 2^3 → a₃ = 2^5 → z = (a₁ + a₃) / 2 → z = 20 := by sorry

end arithmetic_sequence_middle_term_l1884_188430


namespace complex_modulus_l1884_188413

theorem complex_modulus (z : ℂ) (h : z + Complex.I = 3) : Complex.abs z = Real.sqrt 10 := by
  sorry

end complex_modulus_l1884_188413


namespace choose_officers_count_l1884_188420

/-- Represents the club with its member composition -/
structure Club where
  total_members : Nat
  boys : Nat
  girls : Nat
  senior_boys : Nat
  senior_girls : Nat

/-- Calculates the number of ways to choose a president and vice-president -/
def choose_officers (club : Club) : Nat :=
  (club.senior_boys * (club.boys - 1)) + (club.senior_girls * (club.girls - 1))

/-- The specific club instance from the problem -/
def our_club : Club :=
  { total_members := 30
  , boys := 16
  , girls := 14
  , senior_boys := 3
  , senior_girls := 3 
  }

/-- Theorem stating that the number of ways to choose officers for our club is 84 -/
theorem choose_officers_count : choose_officers our_club = 84 := by
  sorry

#eval choose_officers our_club

end choose_officers_count_l1884_188420


namespace expression_simplification_l1884_188437

theorem expression_simplification (a b : ℝ) (h1 : a = 1) (h2 : b = 2) :
  (a - b)^2 - a*(a - b) + (a + b)*(a - b) = -1 := by
  sorry

end expression_simplification_l1884_188437


namespace third_circle_radius_l1884_188410

/-- Given two externally tangent circles and a third circle tangent to both and their common external tangent, prove the radius of the third circle --/
theorem third_circle_radius (r1 r2 r3 : ℝ) : 
  r1 = 1 →                            -- radius of circle A
  r2 = 4 →                            -- radius of circle B
  (r1 + r2)^2 = r1^2 + r2^2 + 6*r1*r2 → -- circles A and B are externally tangent
  (r1 + r3)^2 = (r1 - r3)^2 + 4*r3 →    -- circle with radius r3 is tangent to circle A
  (r2 + r3)^2 = (r2 - r3)^2 + 16*r3 →   -- circle with radius r3 is tangent to circle B
  r3 = 4/9 :=                           -- radius of the third circle
by sorry

end third_circle_radius_l1884_188410


namespace smallest_solution_of_equation_l1884_188450

theorem smallest_solution_of_equation (x : ℝ) :
  x > 0 ∧ x / 4 + 2 / (3 * x) = 5 / 6 →
  x ≥ 4 / 3 :=
by sorry

end smallest_solution_of_equation_l1884_188450


namespace hank_carwash_earnings_l1884_188451

/-- Proves that Hank made $100 in the carwash given the donation information -/
theorem hank_carwash_earnings :
  ∀ (carwash_earnings : ℝ),
    -- Conditions
    (carwash_earnings * 0.9 + 80 * 0.75 + 50 = 200) →
    -- Conclusion
    carwash_earnings = 100 :=
by
  sorry


end hank_carwash_earnings_l1884_188451


namespace contacts_in_second_box_l1884_188435

/-- The number of contacts in the first box -/
def first_box_contacts : ℕ := 50

/-- The price of the first box in cents -/
def first_box_price : ℕ := 2500

/-- The price of the second box in cents -/
def second_box_price : ℕ := 3300

/-- The number of contacts that equal $1 worth in the chosen box -/
def contacts_per_dollar : ℕ := 3

/-- The number of contacts in the second box -/
def second_box_contacts : ℕ := 99

theorem contacts_in_second_box :
  (first_box_price / first_box_contacts > second_box_price / second_box_contacts) ∧
  (second_box_price / second_box_contacts = 100 / contacts_per_dollar) →
  second_box_contacts = 99 := by
  sorry

end contacts_in_second_box_l1884_188435


namespace triangle_angle_c_l1884_188405

theorem triangle_angle_c (A B C : ℝ) : 
  A - B = 10 → B = A / 2 → A + B + C = 180 → C = 150 := by sorry

end triangle_angle_c_l1884_188405


namespace moremom_arrangements_count_l1884_188454

/-- The number of unique arrangements of letters in MOREMOM -/
def moremom_arrangements : ℕ := 420

/-- The total number of letters in MOREMOM -/
def total_letters : ℕ := 7

/-- The number of M's in MOREMOM -/
def m_count : ℕ := 3

/-- The number of O's in MOREMOM -/
def o_count : ℕ := 2

/-- Theorem stating that the number of unique arrangements of letters in MOREMOM is 420 -/
theorem moremom_arrangements_count :
  moremom_arrangements = Nat.factorial total_letters /(Nat.factorial m_count * Nat.factorial o_count) :=
by sorry

end moremom_arrangements_count_l1884_188454


namespace simplify_radical_expression_l1884_188414

theorem simplify_radical_expression : 
  Real.sqrt 80 - 4 * Real.sqrt 5 + 3 * Real.sqrt (180 / 3) = Real.sqrt 540 := by
  sorry

end simplify_radical_expression_l1884_188414


namespace sum_of_solutions_quadratic_l1884_188434

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 + 3*x - 20 = 7*x + 8) → 
  (∃ x₁ x₂ : ℝ, (x₁^2 + 3*x₁ - 20 = 7*x₁ + 8) ∧ 
                (x₂^2 + 3*x₂ - 20 = 7*x₂ + 8) ∧ 
                (x₁ + x₂ = 4)) := by
  sorry

end sum_of_solutions_quadratic_l1884_188434


namespace population_change_l1884_188475

theorem population_change (P : ℝ) : 
  (P * 1.05 * 0.95 = 9975) → P = 10000 := by
  sorry

end population_change_l1884_188475


namespace expression_value_l1884_188471

theorem expression_value : 5^3 - 3 * 5^2 + 3 * 5 - 1 = 64 := by
  sorry

end expression_value_l1884_188471


namespace mixed_beads_cost_l1884_188411

/-- The cost per box of mixed beads -/
def cost_per_box_mixed (red_cost yellow_cost : ℚ) (total_boxes red_boxes yellow_boxes : ℕ) : ℚ :=
  (red_cost * red_boxes + yellow_cost * yellow_boxes) / total_boxes

/-- Theorem stating the cost per box of mixed beads is $1.32 -/
theorem mixed_beads_cost :
  cost_per_box_mixed (13/10) 2 10 4 4 = 132/100 := by
  sorry

end mixed_beads_cost_l1884_188411


namespace probability_at_least_one_even_is_65_81_l1884_188429

def valid_digits : Finset ℕ := {0, 3, 5, 7, 8, 9}
def code_length : ℕ := 4

def probability_at_least_one_even : ℚ :=
  1 - (Finset.filter (λ x => ¬ Even x) valid_digits).card ^ code_length /
      valid_digits.card ^ code_length

theorem probability_at_least_one_even_is_65_81 :
  probability_at_least_one_even = 65 / 81 := by
  sorry

end probability_at_least_one_even_is_65_81_l1884_188429


namespace determinant_zero_l1884_188465

def matrix1 : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 2, 3],
    ![4, 5, 6],
    ![7, 8, 9]]

def matrix2 : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 4, 9],
    ![16, 25, 36],
    ![49, 64, 81]]

theorem determinant_zero (h : Matrix.det matrix1 = 0) :
  Matrix.det matrix2 = 0 := by
  sorry

end determinant_zero_l1884_188465


namespace van_speed_ratio_l1884_188460

theorem van_speed_ratio (distance : ℝ) (original_time : ℝ) (new_speed : ℝ)
  (h1 : distance = 465)
  (h2 : original_time = 5)
  (h3 : new_speed = 62)
  : (distance / new_speed) / original_time = 1.5 := by
  sorry

end van_speed_ratio_l1884_188460


namespace pentagon_divisible_hexagon_divisible_heptagon_not_divisible_l1884_188488

/-- A polygon is a closed shape with a certain number of sides and vertices. -/
structure Polygon where
  sides : ℕ
  vertices : ℕ

/-- A triangle is a polygon with 3 sides and 3 vertices. -/
def Triangle : Polygon := ⟨3, 3⟩

/-- A pentagon is a polygon with 5 sides and 5 vertices. -/
def Pentagon : Polygon := ⟨5, 5⟩

/-- A hexagon is a polygon with 6 sides and 6 vertices. -/
def Hexagon : Polygon := ⟨6, 6⟩

/-- A heptagon is a polygon with 7 sides and 7 vertices. -/
def Heptagon : Polygon := ⟨7, 7⟩

/-- A polygon can be divided into two triangles if there exists a way to combine two triangles to form that polygon. -/
def CanBeDividedIntoTwoTriangles (p : Polygon) : Prop :=
  ∃ (t1 t2 : Polygon), t1 = Triangle ∧ t2 = Triangle ∧ p.sides = t1.sides + t2.sides - 2 ∧ p.vertices = t1.vertices + t2.vertices - 2

theorem pentagon_divisible : CanBeDividedIntoTwoTriangles Pentagon := by sorry

theorem hexagon_divisible : CanBeDividedIntoTwoTriangles Hexagon := by sorry

theorem heptagon_not_divisible : ¬CanBeDividedIntoTwoTriangles Heptagon := by sorry

end pentagon_divisible_hexagon_divisible_heptagon_not_divisible_l1884_188488


namespace number_greater_than_one_sixth_l1884_188407

theorem number_greater_than_one_sixth (x : ℝ) : x = 1/6 + 0.33333333333333337 → x = 0.5 := by
  sorry

end number_greater_than_one_sixth_l1884_188407


namespace fa_f_product_zero_l1884_188446

/-- Given a point F, a line l, and a circle C, prove that |FA| · |F| = 0 --/
theorem fa_f_product_zero (F : ℝ × ℝ) (l : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) : 
  F.1 = 0 →
  l = {(x, y) : ℝ × ℝ | -Real.sqrt 3 * y = 0} →
  C = {(x, y) : ℝ × ℝ | (x - 2)^2 + y^2 = 22} →
  ∃ (A : ℝ × ℝ), A ∈ l ∧ (‖A - F‖ * ‖F‖ = 0) := by
  sorry

#check fa_f_product_zero

end fa_f_product_zero_l1884_188446


namespace complex_equation_solution_l1884_188445

/-- Given a, b ∈ ℝ and a - bi = (1 + i)i³, prove that a = 1 and b = -1 -/
theorem complex_equation_solution (a b : ℝ) : 
  (Complex.mk a (-b) = Complex.I * Complex.I * Complex.I * (1 + Complex.I)) → 
  (a = 1 ∧ b = -1) :=
by sorry

end complex_equation_solution_l1884_188445


namespace second_to_first_rocket_height_ratio_l1884_188468

def first_rocket_height : ℝ := 500
def combined_height : ℝ := 1500

theorem second_to_first_rocket_height_ratio :
  (combined_height - first_rocket_height) / first_rocket_height = 2 := by
  sorry

end second_to_first_rocket_height_ratio_l1884_188468


namespace women_no_traits_l1884_188469

/-- Represents the number of women in the population -/
def total_population : ℕ := 200

/-- Probability of having only one specific trait -/
def prob_one_trait : ℚ := 1/20

/-- Probability of having precisely two specific traits -/
def prob_two_traits : ℚ := 2/25

/-- Probability of having all three traits, given a woman has X and Y -/
def prob_all_given_xy : ℚ := 1/4

/-- Number of women with only one trait -/
def women_one_trait : ℕ := 10

/-- Number of women with exactly two traits -/
def women_two_traits : ℕ := 16

/-- Number of women with all three traits -/
def women_all_traits : ℕ := 5

/-- Theorem stating the number of women with none of the three traits -/
theorem women_no_traits : 
  total_population - 3 * women_one_trait - 3 * women_two_traits - women_all_traits = 117 := by
  sorry

end women_no_traits_l1884_188469


namespace goldfish_count_l1884_188427

/-- The number of goldfish in the aquarium -/
def total_goldfish : ℕ := 100

/-- The number of goldfish Maggie was allowed to take home -/
def allowed_goldfish : ℕ := total_goldfish / 2

/-- The number of goldfish Maggie caught -/
def caught_goldfish : ℕ := (3 * allowed_goldfish) / 5

/-- The number of goldfish Maggie still needs to catch -/
def remaining_goldfish : ℕ := 20

theorem goldfish_count : 
  total_goldfish = 100 ∧
  allowed_goldfish = total_goldfish / 2 ∧
  caught_goldfish = (3 * allowed_goldfish) / 5 ∧
  remaining_goldfish = allowed_goldfish - caught_goldfish ∧
  remaining_goldfish = 20 := by
  sorry

end goldfish_count_l1884_188427


namespace no_integer_solutions_l1884_188403

theorem no_integer_solutions : ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 := by
  sorry

end no_integer_solutions_l1884_188403


namespace count_eight_digit_integers_l1884_188464

/-- The number of different 8-digit positive integers -/
def eight_digit_integers : ℕ := 9 * (10^7)

/-- Theorem stating that the number of different 8-digit positive integers is 90,000,000 -/
theorem count_eight_digit_integers : eight_digit_integers = 90000000 := by
  sorry

end count_eight_digit_integers_l1884_188464


namespace cats_in_academy_l1884_188416

/-- The number of cats that can jump -/
def jump : ℕ := 40

/-- The number of cats that can fetch -/
def fetch : ℕ := 25

/-- The number of cats that can spin -/
def spin : ℕ := 30

/-- The number of cats that can jump and fetch -/
def jump_fetch : ℕ := 20

/-- The number of cats that can fetch and spin -/
def fetch_spin : ℕ := 10

/-- The number of cats that can jump and spin -/
def jump_spin : ℕ := 15

/-- The number of cats that can do all three tricks -/
def all_tricks : ℕ := 7

/-- The number of cats that can do none of the tricks -/
def no_tricks : ℕ := 5

/-- The total number of cats in the academy -/
def total_cats : ℕ := 62

theorem cats_in_academy :
  total_cats = 
    (jump - jump_fetch - jump_spin + all_tricks) +
    (jump_fetch - all_tricks) +
    (fetch - jump_fetch - fetch_spin + all_tricks) +
    (fetch_spin - all_tricks) +
    (jump_spin - all_tricks) +
    (spin - jump_spin - fetch_spin + all_tricks) +
    all_tricks +
    no_tricks := by sorry

end cats_in_academy_l1884_188416
