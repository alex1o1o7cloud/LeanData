import Mathlib

namespace max_value_on_circle_l3867_386763

theorem max_value_on_circle (x y : ℝ) : 
  Complex.abs (x - 2 + y * Complex.I) = 1 →
  (∃ (x' y' : ℝ), Complex.abs (x' - 2 + y' * Complex.I) = 1 ∧ 
    |3 * x' - y'| ≥ |3 * x - y|) →
  |3 * x - y| ≤ 6 + Real.sqrt 10 :=
sorry

end max_value_on_circle_l3867_386763


namespace coin_collection_problem_l3867_386757

theorem coin_collection_problem (n d q : ℕ) : 
  n + d + q = 23 →
  5 * n + 10 * d + 25 * q = 320 →
  d = n + 3 →
  q - n = 2 :=
by sorry

end coin_collection_problem_l3867_386757


namespace triangle_inequality_l3867_386728

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b) : 
  (a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥ 
   (b + c - a) / a + (c + a - b) / b + (a + b - c) / c) ∧
  ((b + c - a) / a + (c + a - b) / b + (a + b - c) / c ≥ 3) := by
  sorry

end triangle_inequality_l3867_386728


namespace inverse_f_at_3_l3867_386787

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2

-- State the theorem
theorem inverse_f_at_3 :
  ∃ (f_inv : ℝ → ℝ),
    (∀ x ≤ 0, f_inv (f x) = x) ∧
    (∀ y ≥ 2, f (f_inv y) = y) ∧
    f_inv 3 = -1 :=
sorry

end inverse_f_at_3_l3867_386787


namespace mlb_game_misses_l3867_386781

theorem mlb_game_misses (hits misses : ℕ) : 
  misses = 3 * hits → 
  hits + misses = 200 → 
  misses = 150 := by
sorry

end mlb_game_misses_l3867_386781


namespace inequality_proof_l3867_386707

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = a*b) :
  (a + 2*b ≥ 8) ∧ (2*a + b ≥ 9) ∧ (a^2 + 4*b^2 + 5*a*b ≥ 72) := by
  sorry

end inequality_proof_l3867_386707


namespace equilateral_triangle_condition_l3867_386727

/-- A function that checks if a natural number n satisfies the conditions for forming an equilateral triangle with sticks of lengths 1 to n. -/
def can_form_equilateral_triangle (n : ℕ) : Prop :=
  n ≥ 5 ∧ (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5)

/-- The sum of the first n natural numbers. -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem stating that sticks of lengths 1 to n can form an equilateral triangle
    if and only if n satisfies the specific conditions. -/
theorem equilateral_triangle_condition (n : ℕ) :
  (sum_first_n n % 3 = 0 ∧ ∀ k < n, k > 0) ↔ can_form_equilateral_triangle n := by
  sorry


end equilateral_triangle_condition_l3867_386727


namespace sandwiches_per_student_l3867_386751

theorem sandwiches_per_student
  (students_per_group : ℕ)
  (total_groups : ℕ)
  (total_bread_pieces : ℕ)
  (bread_per_sandwich : ℕ)
  (h1 : students_per_group = 6)
  (h2 : total_groups = 5)
  (h3 : total_bread_pieces = 120)
  (h4 : bread_per_sandwich = 2) :
  total_bread_pieces / (bread_per_sandwich * (students_per_group * total_groups)) = 2 :=
by sorry

end sandwiches_per_student_l3867_386751


namespace average_speed_theorem_l3867_386799

theorem average_speed_theorem (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 ∧ 
  first_half_speed = 80 ∧ 
  second_half_time_factor = 3 →
  (total_distance / (total_distance / (2 * first_half_speed) * (1 + second_half_time_factor))) = 40 := by
  sorry

#check average_speed_theorem

end average_speed_theorem_l3867_386799


namespace product_from_sum_and_difference_l3867_386724

theorem product_from_sum_and_difference :
  ∀ x y : ℝ, x + y = 72 ∧ x - y = 20 → x * y = 1196 := by
sorry

end product_from_sum_and_difference_l3867_386724


namespace school_principal_election_l3867_386720

/-- Given that Emma received 45 votes in a school principal election,
    and these votes represent 3/7 of the total votes,
    prove that the total number of votes cast is 105. -/
theorem school_principal_election (emma_votes : ℕ) (total_votes : ℕ)
    (h1 : emma_votes = 45)
    (h2 : emma_votes = 3 * total_votes / 7) :
    total_votes = 105 := by
  sorry

end school_principal_election_l3867_386720


namespace star_equation_solution_l3867_386771

/-- Custom binary operation ⭐ -/
def star (a b : ℝ) : ℝ := a * b + 3 * b - a

/-- Theorem stating that if 5 ⭐ x = 40, then x = 45/8 -/
theorem star_equation_solution :
  star 5 x = 40 → x = 45 / 8 := by
  sorry

end star_equation_solution_l3867_386771


namespace larger_integer_value_l3867_386701

theorem larger_integer_value (a b : ℕ+) (h1 : (a : ℝ) / (b : ℝ) = 7 / 3) (h2 : (a : ℕ) * b = 441) :
  max a b = ⌊7 * Real.sqrt 21⌋ := by
  sorry

end larger_integer_value_l3867_386701


namespace mayor_approval_probability_l3867_386737

def probability_two_successes_in_four_trials (p : ℝ) : ℝ :=
  6 * p^2 * (1 - p)^2

theorem mayor_approval_probability : 
  probability_two_successes_in_four_trials 0.6 = 0.3456 := by
  sorry

end mayor_approval_probability_l3867_386737


namespace f_condition_equivalent_to_a_range_l3867_386711

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x + (1/2) * a * x^2 + a * x

theorem f_condition_equivalent_to_a_range :
  ∀ a : ℝ, (∀ x : ℝ, 2 * Real.exp 1 * f a x + Real.exp 1 + 2 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
by sorry

end f_condition_equivalent_to_a_range_l3867_386711


namespace fifth_term_is_nine_l3867_386717

/-- An arithmetic sequence with first term 1 and common difference 2 -/
def arithmetic_sequence (n : ℕ) : ℝ :=
  1 + (n - 1) * 2

/-- The fifth term of the arithmetic sequence is 9 -/
theorem fifth_term_is_nine : arithmetic_sequence 5 = 9 := by
  sorry

end fifth_term_is_nine_l3867_386717


namespace cos_squared_alpha_minus_pi_fourth_l3867_386792

theorem cos_squared_alpha_minus_pi_fourth (α : Real) 
  (h : Real.sin (2 * α) = 1 / 3) : 
  Real.cos (α - π / 4) ^ 2 = 2 / 3 := by
sorry

end cos_squared_alpha_minus_pi_fourth_l3867_386792


namespace no_integer_solution_l3867_386705

theorem no_integer_solution : ¬∃ (x y : ℤ), 
  (x + 2019) * (x + 2020) + (x + 2020) * (x + 2021) + (x + 2019) * (x + 2021) = y^2 := by
  sorry

end no_integer_solution_l3867_386705


namespace vieta_cubic_formulas_l3867_386793

theorem vieta_cubic_formulas (a b c d x₁ x₂ x₃ : ℝ) (ha : a ≠ 0) :
  (∀ x, a * x^3 + b * x^2 + c * x + d = a * (x - x₁) * (x - x₂) * (x - x₃)) →
  (x₁ + x₂ + x₃ = -b / a) ∧ 
  (x₁ * x₂ + x₁ * x₃ + x₂ * x₃ = c / a) ∧ 
  (x₁ * x₂ * x₃ = -d / a) := by
  sorry

end vieta_cubic_formulas_l3867_386793


namespace problem_solution_l3867_386779

/-- The graph of y = x + m - 2 does not pass through the second quadrant -/
def p (m : ℝ) : Prop := ∀ x y : ℝ, y = x + m - 2 → ¬(x < 0 ∧ y > 0)

/-- The equation x^2 + y^2 / (1-m) = 1 represents an ellipse with its focus on the x-axis -/
def q (m : ℝ) : Prop := 0 < 1 - m ∧ 1 - m < 1

theorem problem_solution (m : ℝ) :
  (∀ m, q m → p m) ∧ ¬(∀ m, p m → q m) ∧
  (¬(p m ∧ q m) ∧ (p m ∨ q m) ↔ m ≤ 0 ∨ (1 ≤ m ∧ m ≤ 2)) :=
sorry

end problem_solution_l3867_386779


namespace marble_ratio_l3867_386795

theorem marble_ratio (blue red : ℕ) (h1 : blue = red + 24) (h2 : red = 6) :
  blue / red = 5 := by
  sorry

end marble_ratio_l3867_386795


namespace age_problem_l3867_386704

/-- Mr. Li's current age -/
def mr_li_age : ℕ := 23

/-- Xiao Ming's current age -/
def xiao_ming_age : ℕ := 10

/-- The age difference between Mr. Li and Xiao Ming -/
def age_difference : ℕ := 13

theorem age_problem :
  (mr_li_age - 6 = xiao_ming_age + 7) ∧
  (mr_li_age + 4 + xiao_ming_age - 5 = 32) ∧
  (mr_li_age = xiao_ming_age + age_difference) :=
by sorry

end age_problem_l3867_386704


namespace digit_sum_2017_power_l3867_386770

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The theorem to prove -/
theorem digit_sum_2017_power : S (S (S (S (2017^2017)))) = 1 := by sorry

end digit_sum_2017_power_l3867_386770


namespace angle_DEB_is_165_l3867_386758

-- Define the geometric configuration
structure GeometricConfiguration where
  -- Triangle ABC
  angleACB : ℝ
  angleABC : ℝ
  -- Other angles
  angleADE : ℝ
  angleCDE : ℝ
  -- AEB is a straight angle
  angleAEB : ℝ

-- Define the theorem
theorem angle_DEB_is_165 (config : GeometricConfiguration) 
  (h1 : config.angleACB = 90)
  (h2 : config.angleABC = 55)
  (h3 : config.angleADE = 130)
  (h4 : config.angleCDE = 50)
  (h5 : config.angleAEB = 180) :
  ∃ (angleDEB : ℝ), angleDEB = 165 := by
    sorry

end angle_DEB_is_165_l3867_386758


namespace area_of_region_l3867_386754

-- Define the curve
def curve (x y : ℝ) : Prop := 2 * x^2 - 4 * x - x * y + 2 * y = 0

-- Define the region
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve p.1 p.2 ∧ 0 ≤ p.2}

-- State the theorem
theorem area_of_region : MeasureTheory.volume region = 6 := by sorry

end area_of_region_l3867_386754


namespace number_tower_pattern_l3867_386798

theorem number_tower_pattern (n : ℕ) : (10^n - 1) * 9 + (n + 1) = 10^(n+1) - 1 := by
  sorry

end number_tower_pattern_l3867_386798


namespace equation_solution_l3867_386777

theorem equation_solution (x : ℝ) : 3 - 1 / (2 - x) = 2 * (1 / (2 - x)) → x = 1 := by
  sorry

end equation_solution_l3867_386777


namespace binomial_coefficient_1999000_l3867_386738

theorem binomial_coefficient_1999000 :
  ∀ x : ℕ+, (∃ y : ℕ+, Nat.choose x.val y.val = 1999000) ↔ (x.val = 1999000 ∨ x.val = 2000) := by
  sorry

end binomial_coefficient_1999000_l3867_386738


namespace trig_simplification_l3867_386764

theorem trig_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.sin x ^ 2 := by
  sorry

end trig_simplification_l3867_386764


namespace five_digit_multiple_of_9_l3867_386740

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def five_digit_number (d : ℕ) : ℕ := 56780 + d

theorem five_digit_multiple_of_9 (d : ℕ) : 
  d < 10 → (is_multiple_of_9 (five_digit_number d) ↔ d = 1) := by
  sorry

end five_digit_multiple_of_9_l3867_386740


namespace six_seat_colorings_eq_66_l3867_386786

/-- Represents the number of ways to paint n seats in a circular arrangement
    with the first seat fixed as red, using three colors (red, blue, green)
    such that adjacent seats have different colors. -/
def S : ℕ → ℕ
| 0 => 0
| 1 => 0
| 2 => 2
| 3 => 2
| (n + 2) => S (n + 1) + 2 * S n

/-- The number of ways to paint six seats in a circular arrangement
    using three colors (red, blue, green) such that adjacent seats
    have different colors. -/
def six_seat_colorings : ℕ := 3 * S 6

theorem six_seat_colorings_eq_66 : six_seat_colorings = 66 := by
  sorry

end six_seat_colorings_eq_66_l3867_386786


namespace ratio_problem_l3867_386765

theorem ratio_problem (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + 3 * y) = 3 / 4) :
  x / y = 17 / 6 := by
  sorry

end ratio_problem_l3867_386765


namespace partial_fraction_decomposition_l3867_386780

/-- Given a rational function decomposition, prove the value of B -/
theorem partial_fraction_decomposition (x A B C : ℝ) : 
  (2 : ℝ) / (x^3 + 5*x^2 - 13*x - 35) = A / (x-7) + B / (x+1) + C / (x+1)^2 →
  x^3 + 5*x^2 - 13*x - 35 = (x-7)*(x+1)^2 →
  B = (1 : ℝ) / 16 := by
  sorry

end partial_fraction_decomposition_l3867_386780


namespace book_reading_percentage_l3867_386743

theorem book_reading_percentage (total_pages : ℕ) (remaining_pages : ℕ) : 
  total_pages = 400 → remaining_pages = 320 → 
  (((total_pages - remaining_pages) : ℚ) / total_pages) * 100 = 20 := by
sorry

end book_reading_percentage_l3867_386743


namespace number_of_best_friends_l3867_386762

theorem number_of_best_friends (total_cards : ℕ) (cards_per_friend : ℕ) 
  (h1 : total_cards = 455) 
  (h2 : cards_per_friend = 91) : 
  total_cards / cards_per_friend = 5 := by
  sorry

end number_of_best_friends_l3867_386762


namespace line_plane_relationship_l3867_386766

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perpLine : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpPlane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (a b : Line) (α : Plane) 
  (h1 : perpLine a b) 
  (h2 : perpPlane a α) : 
  subset b α ∨ parallel b α :=
sorry

end line_plane_relationship_l3867_386766


namespace sum_x_y_equals_2700_l3867_386729

theorem sum_x_y_equals_2700 (x y : ℝ) : 
  (0.9 * 600 = 0.5 * x) → 
  (0.6 * x = 0.4 * y) → 
  x + y = 2700 := by
sorry

end sum_x_y_equals_2700_l3867_386729


namespace max_marks_proof_l3867_386773

def math_pass_percentage : ℚ := 45/100
def science_pass_percentage : ℚ := 1/2
def math_score : ℕ := 267
def math_shortfall : ℕ := 45
def science_score : ℕ := 292
def science_shortfall : ℕ := 38

def total_marks : ℕ := 1354

theorem max_marks_proof :
  let math_total := (math_score + math_shortfall) / math_pass_percentage
  let science_total := (science_score + science_shortfall) / science_pass_percentage
  ⌈math_total⌉ + science_total = total_marks := by
  sorry

end max_marks_proof_l3867_386773


namespace lamp_probability_l3867_386708

/-- Represents the total number of outlets available -/
def total_outlets : Nat := 7

/-- Represents the number of plugs to be connected -/
def num_plugs : Nat := 3

/-- Represents the number of ways to plug 3 plugs into 7 outlets -/
def total_ways : Nat := total_outlets * (total_outlets - 1) * (total_outlets - 2)

/-- Represents the number of favorable outcomes where the lamp lights up -/
def favorable_outcomes : Nat := 78

/-- Theorem stating that the probability of the lamp lighting up is 13/35 -/
theorem lamp_probability : 
  (favorable_outcomes : ℚ) / total_ways = 13 / 35 := by sorry

end lamp_probability_l3867_386708


namespace student_card_distribution_l3867_386700

/-- Given n students (n ≥ 3) and m = (n * (n-1)) / 2 cards, prove that if m is odd
    and there exists a distribution of m distinct integers from 1 to m among n students
    such that the pairwise sums of these integers give different remainders modulo m,
    then n - 2 is a perfect square. -/
theorem student_card_distribution (n : ℕ) (h1 : n ≥ 3) :
  let m : ℕ := n * (n - 1) / 2
  ∃ (distribution : Fin n → Fin m),
    Function.Injective distribution ∧
    (∀ i j : Fin n, i ≠ j →
      ∀ k l : Fin n, k ≠ l →
        (distribution i + distribution j : ℕ) % m ≠
        (distribution k + distribution l : ℕ) % m) →
    Odd m →
    ∃ k : ℕ, n - 2 = k^2 := by
  sorry

end student_card_distribution_l3867_386700


namespace shaded_area_in_square_with_semicircles_l3867_386752

/-- Given a square with side length 4 and four semicircles with centers at the midpoints of the square's sides, 
    prove that the area not covered by the semicircles is 8 - 2π. -/
theorem shaded_area_in_square_with_semicircles (square_side : ℝ) (semicircle_radius : ℝ) : 
  square_side = 4 → 
  semicircle_radius = Real.sqrt 2 →
  (4 : ℝ) * (π / 2 * semicircle_radius^2) = 2 * π →
  square_side^2 - (4 : ℝ) * (π / 2 * semicircle_radius^2) = 8 - 2 * π := by
  sorry

#align shaded_area_in_square_with_semicircles shaded_area_in_square_with_semicircles

end shaded_area_in_square_with_semicircles_l3867_386752


namespace leigh_has_16_shells_l3867_386742

-- Define the number of seashells each person has
def mimi_shells : ℕ := 24  -- 2 dozen seashells
def kyle_shells : ℕ := 2 * mimi_shells
def leigh_shells : ℕ := kyle_shells / 3

-- Theorem to prove
theorem leigh_has_16_shells : leigh_shells = 16 := by
  sorry

end leigh_has_16_shells_l3867_386742


namespace jake_fewer_peaches_l3867_386721

theorem jake_fewer_peaches (steven_peaches jill_peaches : ℕ) 
  (h1 : steven_peaches = 14)
  (h2 : jill_peaches = 5)
  (jake_peaches : ℕ)
  (h3 : jake_peaches = jill_peaches + 3)
  (h4 : jake_peaches < steven_peaches) :
  steven_peaches - jake_peaches = 6 := by
  sorry

end jake_fewer_peaches_l3867_386721


namespace land_area_increase_l3867_386730

theorem land_area_increase :
  let initial_side : ℝ := 6
  let increase : ℝ := 1
  let new_side := initial_side + increase
  let initial_area := initial_side ^ 2
  let new_area := new_side ^ 2
  new_area - initial_area = 13 := by
sorry

end land_area_increase_l3867_386730


namespace largest_t_value_l3867_386710

theorem largest_t_value : 
  let f (t : ℝ) := (15 * t^2 - 38 * t + 14) / (4 * t - 3) + 6 * t
  ∃ (t_max : ℝ), t_max = 1 ∧ 
    (∀ (t : ℝ), f t = 7 * t - 2 → t ≤ t_max) ∧
    (f t_max = 7 * t_max - 2) :=
by sorry

end largest_t_value_l3867_386710


namespace complex_number_equality_l3867_386778

theorem complex_number_equality : (1 + Complex.I)^2 * (1 - Complex.I) = 2 - 2 * Complex.I := by
  sorry

end complex_number_equality_l3867_386778


namespace race_time_difference_l3867_386718

/-- Proves the time difference between Malcolm and Joshua finishing a race --/
theorem race_time_difference 
  (malcolm_speed : ℝ) 
  (joshua_speed : ℝ) 
  (race_distance : ℝ) 
  (h1 : malcolm_speed = 5)
  (h2 : joshua_speed = 7)
  (h3 : race_distance = 12) :
  joshua_speed * race_distance - malcolm_speed * race_distance = 24 := by
  sorry

end race_time_difference_l3867_386718


namespace triangle_tangent_equality_l3867_386744

theorem triangle_tangent_equality (A B : ℝ) (a b : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : A + B < π) :
  a * Real.tan A + b * Real.tan B = (a + b) * Real.tan ((A + B) / 2) ↔ a = b :=
by sorry

end triangle_tangent_equality_l3867_386744


namespace angle_measure_in_regular_octagon_l3867_386796

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- Measure of an angle in radians -/
def angle_measure (a b c : ℝ × ℝ) : ℝ := sorry

theorem angle_measure_in_regular_octagon 
  (ABCDEFGH : RegularOctagon) 
  (A E C : ℝ × ℝ) 
  (hA : A = ABCDEFGH.vertices 0)
  (hE : E = ABCDEFGH.vertices 4)
  (hC : C = ABCDEFGH.vertices 2) :
  angle_measure A E C = 112.5 * π / 180 := by
  sorry

end angle_measure_in_regular_octagon_l3867_386796


namespace f_properties_l3867_386794

-- Define the function f(x) = x^3 - 6x + 5
def f (x : ℝ) : ℝ := x^3 - 6*x + 5

-- Define the theorem for the extreme points and the range of k
theorem f_properties :
  -- Part I: Extreme points
  (∃ (x_max x_min : ℝ), x_max = -Real.sqrt 2 ∧ x_min = Real.sqrt 2 ∧
    (∀ (x : ℝ), f x ≤ f x_max) ∧
    (∀ (x : ℝ), f x ≥ f x_min)) ∧
  -- Part II: Range of k
  (∀ (k : ℝ), (∀ (x : ℝ), x > 1 → f x ≥ k * (x - 1)) ↔ k ≤ -3) :=
by sorry

end f_properties_l3867_386794


namespace square_increase_l3867_386706

theorem square_increase (a : ℕ) : (a + 1)^2 - a^2 = 1001 → a = 500 := by
  sorry

end square_increase_l3867_386706


namespace triangle_midpoint_vector_l3867_386784

/-- Given a triangle ABC with vertices A(-1, 0), B(0, 2), and C(2, 0),
    and D is the midpoint of BC, prove that vector AD equals (2, 1) -/
theorem triangle_midpoint_vector (A B C D : ℝ × ℝ) : 
  A = (-1, 0) → B = (0, 2) → C = (2, 0) → D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  (D.1 - A.1, D.2 - A.2) = (2, 1) := by
sorry

end triangle_midpoint_vector_l3867_386784


namespace evaluate_expression_l3867_386702

theorem evaluate_expression : 6 - 8 * (9 - 4^2) / 2 = 20 := by
  sorry

end evaluate_expression_l3867_386702


namespace solve_system_l3867_386703

theorem solve_system (p q : ℚ) (eq1 : 5 * p + 3 * q = 7) (eq2 : 2 * p + 5 * q = 8) : p = 11 / 19 := by
  sorry

end solve_system_l3867_386703


namespace tan_30_plus_4cos_30_l3867_386756

theorem tan_30_plus_4cos_30 :
  Real.tan (30 * π / 180) + 4 * Real.cos (30 * π / 180) = 7 * Real.sqrt 3 / 3 :=
by sorry

end tan_30_plus_4cos_30_l3867_386756


namespace number_relationships_l3867_386768

theorem number_relationships : 
  (10 * 10000 = 100000) ∧
  (10 * 1000000 = 10000000) ∧
  (10 * 10000000 = 100000000) ∧
  (100000000 / 10000 = 10000) := by
  sorry

end number_relationships_l3867_386768


namespace square_sum_plus_quadruple_product_l3867_386723

theorem square_sum_plus_quadruple_product (x y : ℝ) 
  (h1 : x + y = 8) (h2 : x * y = 15) : 
  x^2 + 6*x*y + y^2 = 124 := by
  sorry

end square_sum_plus_quadruple_product_l3867_386723


namespace derivative_zero_at_negative_one_l3867_386739

theorem derivative_zero_at_negative_one (t : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (x^2 - 4) * (x - t)
  let f' : ℝ → ℝ := λ x ↦ 2*x*(x - t) + (x^2 - 4)
  f' (-1) = 0 → t = 1/2 := by
  sorry

end derivative_zero_at_negative_one_l3867_386739


namespace isosceles_right_triangle_roots_l3867_386747

theorem isosceles_right_triangle_roots (a b z₁ z₂ : ℂ) : 
  z₁^2 + a*z₁ + b = 0 → 
  z₂^2 + a*z₂ + b = 0 → 
  z₁ ≠ z₂ →
  (z₂ - 0) • (z₁ - 0) = 0 →  -- Perpendicular condition
  Complex.abs (z₂ - 0) = Complex.abs (z₁ - z₂) →  -- Isosceles condition
  a^2 / b = 2*Real.sqrt 2 + 2*Complex.I*Real.sqrt 2 :=
by sorry

end isosceles_right_triangle_roots_l3867_386747


namespace range_of_m_l3867_386785

/-- Condition p: |1 - (x-1)/3| < 2 -/
def p (x : ℝ) : Prop := |1 - (x-1)/3| < 2

/-- Condition q: (x-1)^2 < m^2 -/
def q (x m : ℝ) : Prop := (x-1)^2 < m^2

/-- q is a sufficient condition for p -/
def q_sufficient (m : ℝ) : Prop := ∀ x, q x m → p x

/-- q is not a necessary condition for p -/
def q_not_necessary (m : ℝ) : Prop := ∃ x, p x ∧ ¬q x m

theorem range_of_m :
  (∀ m, q_sufficient m ∧ q_not_necessary m) →
  (∀ m, m ∈ Set.Icc (-3 : ℝ) 3) ∧ 
  (∃ m₁ m₂, m₁ ∈ Set.Ioo (-3 : ℝ) 3 ∧ m₂ ∈ Set.Ioo (-3 : ℝ) 3 ∧ m₁ ≠ m₂) :=
sorry

end range_of_m_l3867_386785


namespace log_equation_solution_l3867_386769

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  4 * Real.log x / Real.log 3 = Real.log (4 * x) / Real.log 3 → x = (4 : ℝ) ^ (1/3) := by
  sorry

end log_equation_solution_l3867_386769


namespace right_triangle_arithmetic_progression_l3867_386716

theorem right_triangle_arithmetic_progression (a b c : ℝ) : 
  -- The triangle is right-angled
  a^2 + b^2 = c^2 →
  -- The lengths form an arithmetic progression
  b - a = c - b →
  -- The common difference is 1
  b - a = 1 →
  -- The hypotenuse is 5
  c = 5 := by
sorry

end right_triangle_arithmetic_progression_l3867_386716


namespace min_value_fraction_l3867_386753

theorem min_value_fraction (x y : ℝ) : 
  x ≥ 0 → y ≥ 0 → x + y = 2 → 
  (∀ a b : ℝ, a ≥ 0 → b ≥ 0 → a + b = 2 → 8 / ((x + 2) * (y + 4)) ≤ 8 / ((a + 2) * (b + 4))) →
  8 / ((x + 2) * (y + 4)) = 1 / 2 :=
by sorry

end min_value_fraction_l3867_386753


namespace ivan_petrovich_savings_l3867_386790

/-- Simple interest calculation --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proof of Ivan Petrovich's retirement savings --/
theorem ivan_petrovich_savings : 
  let principal : ℝ := 750000
  let rate : ℝ := 0.08
  let time : ℝ := 12
  simple_interest principal rate time = 1470000 := by
  sorry

end ivan_petrovich_savings_l3867_386790


namespace sum_x_y_z_l3867_386726

theorem sum_x_y_z (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 3 * y + x) : 
  x + y + z = 14 * x := by
sorry

end sum_x_y_z_l3867_386726


namespace vector_operation_proof_l3867_386776

theorem vector_operation_proof :
  let v1 : Fin 2 → ℝ := ![3, -5]
  let v2 : Fin 2 → ℝ := ![-1, 6]
  let v3 : Fin 2 → ℝ := ![2, -1]
  5 • v1 - 3 • v2 + v3 = ![20, -44] := by
  sorry

end vector_operation_proof_l3867_386776


namespace quadratic_completing_square_l3867_386788

/-- The quadratic equation x^2 + 2x - 1 = 0 is equivalent to (x+1)^2 = 2 -/
theorem quadratic_completing_square :
  ∀ x : ℝ, x^2 + 2*x - 1 = 0 ↔ (x + 1)^2 = 2 := by sorry

end quadratic_completing_square_l3867_386788


namespace x_intercept_distance_l3867_386745

/-- Given two lines intersecting at (8, 20), one with slope 4 and the other with slope -3,
    the distance between their x-intercepts is 35/3. -/
theorem x_intercept_distance (line1 line2 : (ℝ → ℝ)) : 
  (∀ x, line1 x = 4 * x - 12) →
  (∀ x, line2 x = -3 * x + 44) →
  line1 8 = 20 →
  line2 8 = 20 →
  |((0 - (-12)) / 4) - ((0 - 44) / (-3))| = 35/3 := by
sorry

end x_intercept_distance_l3867_386745


namespace max_gcd_lcm_value_l3867_386755

theorem max_gcd_lcm_value (a b c : ℕ) 
  (h : Nat.gcd (Nat.lcm a b) c * Nat.lcm (Nat.gcd a b) c = 200) : 
  Nat.gcd (Nat.lcm a b) c ≤ 10 ∧ 
  ∃ (a' b' c' : ℕ), Nat.gcd (Nat.lcm a' b') c' = 10 ∧ 
    Nat.gcd (Nat.lcm a' b') c' * Nat.lcm (Nat.gcd a' b') c' = 200 :=
by sorry

end max_gcd_lcm_value_l3867_386755


namespace probability_same_length_segments_l3867_386774

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of segments (sides + diagonals) in a regular hexagon -/
def total_segments : ℕ := num_sides + num_diagonals

/-- The number of diagonals of the first length in a regular hexagon -/
def num_diagonals_length1 : ℕ := 3

/-- The number of diagonals of the second length in a regular hexagon -/
def num_diagonals_length2 : ℕ := 6

/-- The probability of selecting two segments of the same length from a regular hexagon -/
theorem probability_same_length_segments :
  (Nat.choose num_sides 2 + Nat.choose num_diagonals_length1 2 + Nat.choose num_diagonals_length2 2) /
  Nat.choose total_segments 2 = 11 / 35 := by
  sorry

end probability_same_length_segments_l3867_386774


namespace common_number_in_overlapping_lists_l3867_386714

theorem common_number_in_overlapping_lists (l : List ℚ) : 
  l.length = 7 ∧ 
  (l.take 4).sum / 4 = 7 ∧ 
  (l.drop 3).sum / 4 = 11 ∧ 
  l.sum / 7 = 66 / 7 → 
  ∃ x ∈ l.take 4 ∩ l.drop 3, x = 6 := by
sorry

end common_number_in_overlapping_lists_l3867_386714


namespace simplify_nested_roots_l3867_386732

theorem simplify_nested_roots (b : ℝ) (hb : b > 0) :
  (((b^16)^(1/3))^(1/4))^3 * (((b^16)^(1/4))^(1/3))^3 = b^8 := by
  sorry

end simplify_nested_roots_l3867_386732


namespace larger_number_problem_l3867_386733

theorem larger_number_problem (x y : ℝ) (h1 : x - y = 5) (h2 : x + y = 27) : x = 16 := by
  sorry

end larger_number_problem_l3867_386733


namespace sqrt_equation_solution_l3867_386722

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (4 + Real.sqrt (3 * y - 7)) = 3 → y = 32 / 3 := by
  sorry

end sqrt_equation_solution_l3867_386722


namespace fixed_point_on_graph_l3867_386715

theorem fixed_point_on_graph (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := fun x ↦ a^(x + 2) - 3
  f (-2) = -2 := by
  sorry

end fixed_point_on_graph_l3867_386715


namespace starting_player_wins_l3867_386761

/-- A game state representing the cards held by each player -/
structure GameState :=
  (player_cards : List Nat)
  (opponent_cards : List Nat)

/-- Check if a list of digits can form a number divisible by 17 -/
def can_form_divisible_by_17 (digits : List Nat) : Bool :=
  sorry

/-- The optimal strategy for the starting player -/
def optimal_strategy (state : GameState) : Option Nat :=
  sorry

/-- Theorem stating that the starting player wins with optimal play -/
theorem starting_player_wins :
  ∀ (initial_cards : List Nat),
    initial_cards.length = 7 ∧
    (∀ n, n ∈ initial_cards → n ≥ 0 ∧ n ≤ 6) →
    ∃ (final_state : GameState),
      final_state.player_cards ⊆ initial_cards ∧
      final_state.opponent_cards ⊆ initial_cards ∧
      final_state.player_cards.length + final_state.opponent_cards.length = 7 ∧
      can_form_divisible_by_17 final_state.player_cards ∧
      ¬can_form_divisible_by_17 final_state.opponent_cards :=
  sorry

end starting_player_wins_l3867_386761


namespace food_product_range_l3867_386749

/-- Represents the net content of a food product -/
structure NetContent where
  nominal : ℝ
  tolerance : ℝ

/-- Represents a range of values -/
structure Range where
  lower : ℝ
  upper : ℝ

/-- Calculates the qualified net content range for a given net content -/
def qualifiedRange (nc : NetContent) : Range :=
  { lower := nc.nominal - nc.tolerance,
    upper := nc.nominal + nc.tolerance }

/-- Theorem: The qualified net content range for a product labeled "500g ± 5g" is 495g to 505g -/
theorem food_product_range :
  let nc : NetContent := { nominal := 500, tolerance := 5 }
  let range := qualifiedRange nc
  range.lower = 495 ∧ range.upper = 505 := by
  sorry

end food_product_range_l3867_386749


namespace cube_minus_reciprocal_cube_l3867_386789

theorem cube_minus_reciprocal_cube (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 := by
  sorry

end cube_minus_reciprocal_cube_l3867_386789


namespace geometric_sequence_sum_l3867_386783

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry

end geometric_sequence_sum_l3867_386783


namespace find_b_l3867_386736

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * b) : b = 49 := by
  sorry

end find_b_l3867_386736


namespace carrot_count_l3867_386782

theorem carrot_count (initial picked_later thrown_out : ℕ) :
  initial ≥ thrown_out →
  initial - thrown_out + picked_later = initial + picked_later - thrown_out :=
by sorry

end carrot_count_l3867_386782


namespace symmetry_propositions_l3867_386735

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Proposition ①
def prop1 (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x - 1) = f (x + 1)) →
  (∀ x y : ℝ, f (1 + (x - 1)) = f (1 - (x - 1)) ∧ y = f (x - 1) ↔ y = f (2 - x))

-- Proposition ②
def prop2 (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f x = -f (-x)) →
  (∀ x y : ℝ, y = f (x - 1) ↔ -y = f (2 - x))

-- Proposition ③
def prop3 (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 1) + f (1 - x) = 0) →
  (∀ x y : ℝ, y = f x ↔ -y = f (2 - x))

-- Proposition ④
def prop4 (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, y = f (x - 1) ↔ y = f (1 - x)

-- Theorem stating which propositions are correct
theorem symmetry_propositions :
  ¬ (∀ f : ℝ → ℝ, prop1 f) ∧
  (∀ f : ℝ → ℝ, prop2 f) ∧
  (∀ f : ℝ → ℝ, prop3 f) ∧
  ¬ (∀ f : ℝ → ℝ, prop4 f) :=
sorry

end symmetry_propositions_l3867_386735


namespace quadratic_equations_solutions_l3867_386746

theorem quadratic_equations_solutions :
  (∃ x : ℝ, 2*x^2 - 4*x - 1 = 0) ∧
  (∃ x : ℝ, 4*(x+2)^2 - 9*(x-3)^2 = 0) ∧
  (∀ x : ℝ, 2*x^2 - 4*x - 1 = 0 → x = (2 + Real.sqrt 6) / 2 ∨ x = (2 - Real.sqrt 6) / 2) ∧
  (∀ x : ℝ, 4*(x+2)^2 - 9*(x-3)^2 = 0 → x = 1 ∨ x = 13) :=
by sorry

end quadratic_equations_solutions_l3867_386746


namespace min_chord_length_proof_l3867_386712

/-- The circle equation x^2 + y^2 - 6x = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

/-- The point through which the chord passes -/
def point : ℝ × ℝ := (1, 2)

/-- The minimum length of the chord intercepted by the circle passing through the point -/
def min_chord_length : ℝ := 2

theorem min_chord_length_proof :
  ∀ (x y : ℝ), circle_equation x y →
  min_chord_length = (2 : ℝ) :=
sorry

end min_chord_length_proof_l3867_386712


namespace playground_route_combinations_l3867_386797

theorem playground_route_combinations : 
  ∀ (n : ℕ) (k : ℕ), n = 2 ∧ k = 3 → n ^ k = 8 := by
  sorry

end playground_route_combinations_l3867_386797


namespace polar_to_rectangular_coordinates_l3867_386734

theorem polar_to_rectangular_coordinates (r : ℝ) (θ : ℝ) :
  r = 2 ∧ θ = π / 6 →
  ∃ x y : ℝ, x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ x = Real.sqrt 3 ∧ y = 1 := by
  sorry

end polar_to_rectangular_coordinates_l3867_386734


namespace fourth_section_area_l3867_386750

/-- Represents a regular hexagon divided into four sections by three line segments -/
structure DividedHexagon where
  total_area : ℝ
  section1_area : ℝ
  section2_area : ℝ
  section3_area : ℝ
  section4_area : ℝ
  is_regular : total_area = 6 * (section1_area + section2_area + section3_area + section4_area) / 6
  sum_of_parts : total_area = section1_area + section2_area + section3_area + section4_area

/-- The theorem stating that if three sections of a divided regular hexagon have areas 2, 3, and 4,
    then the fourth section has an area of 11 -/
theorem fourth_section_area (h : DividedHexagon) 
    (h2 : h.section1_area = 2) 
    (h3 : h.section2_area = 3) 
    (h4 : h.section3_area = 4) : 
    h.section4_area = 11 := by
  sorry

end fourth_section_area_l3867_386750


namespace quadratic_function_value_l3867_386775

/-- A quadratic function with specific properties -/
def QuadraticFunction (d e f : ℝ) : ℝ → ℝ := fun x ↦ d * x^2 + e * x + f

theorem quadratic_function_value (d e f : ℝ) :
  (∀ x, QuadraticFunction d e f x = d * x^2 + e * x + f) →
  QuadraticFunction d e f 0 = 2 →
  (∀ x, QuadraticFunction d e f (3.5 + x) = QuadraticFunction d e f (3.5 - x)) →
  ∃ n : ℤ, QuadraticFunction d e f 10 = n →
  QuadraticFunction d e f 10 = 2 := by
  sorry

end quadratic_function_value_l3867_386775


namespace ellipse_slope_theorem_l3867_386731

-- Define the ellipse
def is_on_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the foci property
def foci_property (a b : ℝ) (F₁ F₂ : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, c^2 = a^2 - b^2 ∧ F₁ = (c, 0) ∧ F₂ = (-c, 0)

-- Define the complementary angle property
def complementary_angles (k : ℝ) : Prop :=
  ∃ (xC yC xD yD : ℝ),
    yC - 1 = k * (xC - 1) ∧
    yD - 1 = -k * (xD - 1)

theorem ellipse_slope_theorem (a b : ℝ) (F₁ F₂ : ℝ × ℝ) (k : ℝ) :
  a > b ∧ b > 0 →
  is_on_ellipse 1 1 a b →
  foci_property a b F₁ F₂ →
  Real.sqrt ((1 - F₁.1)^2 + (1 - F₁.2)^2) + Real.sqrt ((1 - F₂.1)^2 + (1 - F₂.2)^2) = 4 →
  complementary_angles k →
  ∃ (xC yC xD yD : ℝ),
    is_on_ellipse xC yC a b ∧
    is_on_ellipse xD yD a b ∧
    (yD - yC) / (xD - xC) = 1/3 :=
by sorry

end ellipse_slope_theorem_l3867_386731


namespace lei_lei_sheep_count_l3867_386767

/-- The number of sheep Lei Lei bought -/
def num_sheep : ℕ := 10

/-- The initial average price per sheep in yuan -/
def initial_avg_price : ℚ := sorry

/-- The total price of all sheep and goats -/
def total_price : ℚ := sorry

/-- The number of goats Lei Lei bought -/
def num_goats : ℕ := sorry

theorem lei_lei_sheep_count :
  (total_price + 2 * (initial_avg_price + 60) = (num_sheep + 2) * (initial_avg_price + 60)) ∧
  (total_price - 2 * (initial_avg_price - 90) = (num_sheep - 2) * (initial_avg_price - 90)) →
  num_sheep = 10 := by sorry

end lei_lei_sheep_count_l3867_386767


namespace sum_of_squares_l3867_386713

theorem sum_of_squares (a b : ℝ) : (a^2 + b^2) * (a^2 + b^2 + 4) = 12 → a^2 + b^2 = 2 := by
  sorry

end sum_of_squares_l3867_386713


namespace jessica_red_marbles_l3867_386709

theorem jessica_red_marbles (sandy_marbles : ℕ) (sandy_multiple : ℕ) :
  sandy_marbles = 144 →
  sandy_multiple = 4 →
  (sandy_marbles / sandy_multiple) / 12 = 3 := by
  sorry

end jessica_red_marbles_l3867_386709


namespace expression_value_l3867_386741

theorem expression_value (y d : ℝ) (h1 : y > 0) 
  (h2 : (8 * y) / 20 + (3 * y) / d = 0.7 * y) : d = 10 := by
  sorry

end expression_value_l3867_386741


namespace max_min_sum_of_f_l3867_386791

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.log (Real.sqrt (x^2 + 1) + x)) / (x^2 + 1)

theorem max_min_sum_of_f :
  ∃ (M N : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧
                (∀ x, N ≤ f x) ∧ (∃ x, f x = N) ∧
                (M + N = 2) := by
  sorry

end max_min_sum_of_f_l3867_386791


namespace unique_solution_l3867_386748

def system_solution (x y : ℝ) : Prop :=
  2 * x + y = 3 ∧ x - 2 * y = -1

theorem unique_solution : 
  {p : ℝ × ℝ | system_solution p.1 p.2} = {(1, 1)} := by sorry

end unique_solution_l3867_386748


namespace sum_even_integers_402_to_500_l3867_386725

/-- Sum of first n positive even integers -/
def sumFirstEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of even integers from a to b inclusive -/
def sumEvenIntegers (a b : ℕ) : ℕ :=
  let n := (b - a) / 2 + 1
  n * (a + b) / 2

theorem sum_even_integers_402_to_500 :
  sumFirstEvenIntegers 50 = 2550 →
  sumEvenIntegers 402 500 = 22550 := by
  sorry

end sum_even_integers_402_to_500_l3867_386725


namespace function_value_at_2012_l3867_386772

theorem function_value_at_2012 (f : ℝ → ℝ) 
  (h1 : f 0 = 2012)
  (h2 : ∀ x : ℝ, f (x + 2) - f x ≤ 3 * 2^x)
  (h3 : ∀ x : ℝ, f (x + 6) - f x ≥ 63 * 2^x) :
  f 2012 = 2^2012 + 2011 := by
sorry

end function_value_at_2012_l3867_386772


namespace club_members_proof_l3867_386719

theorem club_members_proof (total : ℕ) (left_handed : ℕ) (jazz_lovers : ℕ) (right_handed_jazz_dislikers : ℕ) 
  (h1 : total = 25)
  (h2 : left_handed = 10)
  (h3 : jazz_lovers = 18)
  (h4 : right_handed_jazz_dislikers = 2)
  (h5 : left_handed + (total - left_handed) = total) :
  ∃ x : ℕ, x = 5 ∧ 
    x + (left_handed - x) + (jazz_lovers - x) + right_handed_jazz_dislikers = total :=
by sorry

end club_members_proof_l3867_386719


namespace jessica_cut_two_roses_l3867_386760

/-- The number of roses Jessica cut from her garden -/
def roses_cut (initial_roses initial_orchids final_roses final_orchids : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Theorem stating that Jessica cut 2 roses given the initial and final flower counts -/
theorem jessica_cut_two_roses :
  roses_cut 15 62 17 96 = 2 := by
  sorry

end jessica_cut_two_roses_l3867_386760


namespace cyclist_travel_time_is_40_l3867_386759

/-- Represents the tram schedule and cyclist's journey -/
structure TramSchedule where
  /-- Interval between tram departures from Station A (in minutes) -/
  departure_interval : ℕ
  /-- Time for a tram to travel from Station A to Station B (in minutes) -/
  journey_time : ℕ
  /-- Number of trams encountered by the cyclist -/
  trams_encountered : ℕ

/-- Calculates the cyclist's travel time -/
def cyclist_travel_time (schedule : TramSchedule) : ℕ :=
  (schedule.trams_encountered + 2) * schedule.departure_interval

/-- Theorem stating the cyclist's travel time is 40 minutes -/
theorem cyclist_travel_time_is_40 (schedule : TramSchedule)
  (h1 : schedule.departure_interval = 5)
  (h2 : schedule.journey_time = 15)
  (h3 : schedule.trams_encountered = 10) :
  cyclist_travel_time schedule = 40 := by
  sorry

#eval cyclist_travel_time { departure_interval := 5, journey_time := 15, trams_encountered := 10 }

end cyclist_travel_time_is_40_l3867_386759
