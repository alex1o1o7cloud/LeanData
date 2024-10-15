import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_squares_l767_76733

theorem sum_of_squares (x y : ℕ+) 
  (h1 : x * y + x + y = 83)
  (h2 : x * x * y + x * y * y = 1056) :
  x * x + y * y = 458 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l767_76733


namespace NUMINAMATH_CALUDE_pencil_distribution_l767_76780

theorem pencil_distribution (boxes : Real) (pencils_per_box : Real) (students : Nat) :
  boxes = 4.0 →
  pencils_per_box = 648.0 →
  students = 36 →
  (boxes * pencils_per_box) / students = 72 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l767_76780


namespace NUMINAMATH_CALUDE_existential_and_true_proposition_l767_76765

theorem existential_and_true_proposition :
  (∃ a : ℕ, a^2 + a ≤ 0) ∧
  (∃ a : ℕ, a^2 + a ≤ 0) = True :=
by sorry

end NUMINAMATH_CALUDE_existential_and_true_proposition_l767_76765


namespace NUMINAMATH_CALUDE_ab_nonpositive_l767_76786

theorem ab_nonpositive (a b : ℝ) (h : 2011 * a + 2012 * b = 0) : a * b ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_nonpositive_l767_76786


namespace NUMINAMATH_CALUDE_crossnumber_puzzle_l767_76722

/-- A number is a two-digit number if it's between 10 and 99 inclusive. -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The tens digit of a natural number -/
def tensDigit (n : ℕ) : ℕ := n / 10

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The statement of the crossnumber puzzle -/
theorem crossnumber_puzzle :
  ∃! (a b c d : ℕ),
    isTwoDigit a ∧ isTwoDigit b ∧ isTwoDigit c ∧ isTwoDigit d ∧
    Nat.Prime a ∧
    ∃ (m n p : ℕ), b = m^2 ∧ c = n^2 ∧ d = p^2 ∧
    tensDigit a = unitsDigit b ∧
    unitsDigit a = tensDigit d ∧
    c = d :=
sorry

end NUMINAMATH_CALUDE_crossnumber_puzzle_l767_76722


namespace NUMINAMATH_CALUDE_line_through_A_and_B_line_through_C_and_D_line_through_E_and_K_line_through_M_and_P_l767_76738

-- Define points
def A : ℝ × ℝ := (3, 0)
def B : ℝ × ℝ := (3, 5)
def C : ℝ × ℝ := (2, 2)
def D : ℝ × ℝ := (-1, -1)
def E : ℝ × ℝ := (0, 4)
def K : ℝ × ℝ := (2, 0)
def M : ℝ × ℝ := (3, 2)
def P : ℝ × ℝ := (6, 3)

-- Define line equations
def line1 (x : ℝ) : Prop := x = 3
def line2 (x y : ℝ) : Prop := y = x
def line3 (x y : ℝ) : Prop := y = -2 * x + 4
def line4 (x y : ℝ) : Prop := y = (1/3) * x + 1

-- Theorem statements
theorem line_through_A_and_B : 
  ∀ x y : ℝ, (x, y) = A ∨ (x, y) = B → line1 x := by sorry

theorem line_through_C_and_D : 
  ∀ x y : ℝ, (x, y) = C ∨ (x, y) = D → line2 x y := by sorry

theorem line_through_E_and_K : 
  ∀ x y : ℝ, (x, y) = E ∨ (x, y) = K → line3 x y := by sorry

theorem line_through_M_and_P : 
  ∀ x y : ℝ, (x, y) = M ∨ (x, y) = P → line4 x y := by sorry

end NUMINAMATH_CALUDE_line_through_A_and_B_line_through_C_and_D_line_through_E_and_K_line_through_M_and_P_l767_76738


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_l767_76758

/-- Given a quadrilateral EFGH with the following properties:
  - m∠F = m∠G = 135°
  - EF = 4
  - FG = 6
  - GH = 8
  Prove that the area of EFGH is 18√2 -/
theorem area_of_quadrilateral (E F G H : ℝ × ℝ) : 
  let angle (A B C : ℝ × ℝ) := Real.arccos ((B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2)) / 
    (((B.1 - A.1)^2 + (B.2 - A.2)^2)^(1/2) * ((C.1 - B.1)^2 + (C.2 - B.2)^2)^(1/2))
  angle F E G = 135 * π / 180 ∧
  angle G F H = 135 * π / 180 ∧
  ((E.1 - F.1)^2 + (E.2 - F.2)^2)^(1/2) = 4 ∧
  ((F.1 - G.1)^2 + (F.2 - G.2)^2)^(1/2) = 6 ∧
  ((G.1 - H.1)^2 + (G.2 - H.2)^2)^(1/2) = 8 →
  let area := 
    1/2 * ((E.1 - F.1)^2 + (E.2 - F.2)^2)^(1/2) * ((F.1 - G.1)^2 + (F.2 - G.2)^2)^(1/2) * Real.sin (angle F E G) +
    1/2 * ((F.1 - G.1)^2 + (F.2 - G.2)^2)^(1/2) * ((G.1 - H.1)^2 + (G.2 - H.2)^2)^(1/2) * Real.sin (angle G F H)
  area = 18 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_l767_76758


namespace NUMINAMATH_CALUDE_min_skew_edge_distance_l767_76766

/-- A regular octahedron with edge length a -/
structure RegularOctahedron (a : ℝ) where
  edge_length : a > 0

/-- A point on an edge of the octahedron -/
structure EdgePoint (O : RegularOctahedron a) where
  -- Additional properties can be added if needed

/-- The distance between two points on skew edges of the octahedron -/
def skew_edge_distance (O : RegularOctahedron a) (p q : EdgePoint O) : ℝ := sorry

/-- The theorem stating the minimal distance between points on skew edges -/
theorem min_skew_edge_distance (a : ℝ) (O : RegularOctahedron a) :
  ∃ (p q : EdgePoint O), 
    skew_edge_distance O p q = a * Real.sqrt 6 / 3 ∧
    ∀ (r s : EdgePoint O), skew_edge_distance O r s ≥ a * Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_skew_edge_distance_l767_76766


namespace NUMINAMATH_CALUDE_range_of_a_l767_76701

theorem range_of_a (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) : 
  1 ≤ a ∧ a ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l767_76701


namespace NUMINAMATH_CALUDE_max_sum_abcd_l767_76702

theorem max_sum_abcd (a b c d : ℤ) 
  (b_pos : b > 0)
  (eq1 : a + b = c)
  (eq2 : b + c = d)
  (eq3 : c + d = a) :
  a + b + c + d ≤ -5 ∧ ∃ (a₀ b₀ c₀ d₀ : ℤ), 
    b₀ > 0 ∧ 
    a₀ + b₀ = c₀ ∧ 
    b₀ + c₀ = d₀ ∧ 
    c₀ + d₀ = a₀ ∧ 
    a₀ + b₀ + c₀ + d₀ = -5 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_abcd_l767_76702


namespace NUMINAMATH_CALUDE_max_lessons_is_216_l767_76712

/-- Represents the teacher's wardrobe and lesson capacity. -/
structure TeacherWardrobe where
  shirts : ℕ
  pants : ℕ
  shoes : ℕ
  jackets : ℕ
  lesson_count : ℕ

/-- Calculates the number of lessons possible with the given wardrobe. -/
def calculate_lessons (w : TeacherWardrobe) : ℕ :=
  2 * w.shirts * w.pants * w.shoes

/-- Checks if the wardrobe satisfies the given conditions. -/
def satisfies_conditions (w : TeacherWardrobe) : Prop :=
  w.jackets = 2 ∧
  calculate_lessons { w with shirts := w.shirts + 1 } = w.lesson_count + 36 ∧
  calculate_lessons { w with pants := w.pants + 1 } = w.lesson_count + 72 ∧
  calculate_lessons { w with shoes := w.shoes + 1 } = w.lesson_count + 54

/-- The theorem stating the maximum number of lessons. -/
theorem max_lessons_is_216 :
  ∃ (w : TeacherWardrobe), satisfies_conditions w ∧ w.lesson_count = 216 ∧
  ∀ (w' : TeacherWardrobe), satisfies_conditions w' → w'.lesson_count ≤ 216 :=
sorry

end NUMINAMATH_CALUDE_max_lessons_is_216_l767_76712


namespace NUMINAMATH_CALUDE_intersection_slope_l767_76792

/-- The slope of the line passing through the intersection points of two circles -/
theorem intersection_slope (x y : ℝ) : 
  (x^2 + y^2 - 6*x + 4*y - 20 = 0) ∧ 
  (x^2 + y^2 - 16*x + 8*y + 40 = 0) → 
  (∃ m : ℝ, m = -5/2 ∧ 
    ∀ (x₁ y₁ x₂ y₂ : ℝ), 
      (x₁^2 + y₁^2 - 6*x₁ + 4*y₁ - 20 = 0) ∧ 
      (x₁^2 + y₁^2 - 16*x₁ + 8*y₁ + 40 = 0) ∧
      (x₂^2 + y₂^2 - 6*x₂ + 4*y₂ - 20 = 0) ∧ 
      (x₂^2 + y₂^2 - 16*x₂ + 8*y₂ + 40 = 0) ∧
      (x₁ ≠ x₂) →
      m = (y₂ - y₁) / (x₂ - x₁)) :=
sorry

end NUMINAMATH_CALUDE_intersection_slope_l767_76792


namespace NUMINAMATH_CALUDE_kristy_baked_69_cookies_l767_76741

def cookie_problem (C : ℕ) : Prop :=
  let remaining_after_kristy := C - 3
  let remaining_after_brother := remaining_after_kristy / 2
  let remaining_after_friend1 := remaining_after_brother - 4
  let friend2_took := 2 * 4
  let friend2_returned := friend2_took / 4
  let remaining_after_friend2 := remaining_after_friend1 - (friend2_took - friend2_returned)
  let remaining_after_friend3 := remaining_after_friend2 - 8
  let remaining_after_friend4 := remaining_after_friend3 - 3
  let final_remaining := remaining_after_friend4 - 7
  2 * final_remaining = 10

theorem kristy_baked_69_cookies : ∃ C : ℕ, cookie_problem C ∧ C = 69 := by
  sorry

end NUMINAMATH_CALUDE_kristy_baked_69_cookies_l767_76741


namespace NUMINAMATH_CALUDE_balloon_purchase_theorem_l767_76762

/-- The price of a regular balloon -/
def regular_price : ℚ := 1

/-- The price of a discounted balloon -/
def discounted_price : ℚ := 1/2

/-- The total budget available -/
def budget : ℚ := 30

/-- The cost of a set of three balloons -/
def set_cost : ℚ := 2 * regular_price + discounted_price

/-- The number of balloons in a set -/
def balloons_per_set : ℕ := 3

/-- The maximum number of balloons that can be purchased -/
def max_balloons : ℕ := 36

theorem balloon_purchase_theorem : 
  (budget / set_cost : ℚ).floor * balloons_per_set = max_balloons :=
sorry

end NUMINAMATH_CALUDE_balloon_purchase_theorem_l767_76762


namespace NUMINAMATH_CALUDE_tournament_matches_l767_76745

/-- Represents a single-elimination tournament -/
structure Tournament where
  initial_teams : ℕ
  matches_played : ℕ

/-- The number of teams remaining after playing a certain number of matches -/
def teams_remaining (t : Tournament) : ℕ :=
  t.initial_teams - t.matches_played

theorem tournament_matches (t : Tournament) 
  (h1 : t.initial_teams = 128)
  (h2 : teams_remaining t = 1) : 
  t.matches_played = 127 := by
sorry

end NUMINAMATH_CALUDE_tournament_matches_l767_76745


namespace NUMINAMATH_CALUDE_log_sqrt12_1728sqrt12_eq_7_l767_76775

theorem log_sqrt12_1728sqrt12_eq_7 : Real.log (1728 * Real.sqrt 12) / Real.log (Real.sqrt 12) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sqrt12_1728sqrt12_eq_7_l767_76775


namespace NUMINAMATH_CALUDE_two_valid_numbers_l767_76736

def digits (n : ℕ) : Finset ℕ :=
  (n.digits 10).toFinset

def valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ (digits n ∪ digits (n * n) = Finset.range 9)

theorem two_valid_numbers :
  {n : ℕ | valid_number n} = {567, 854} := by sorry

end NUMINAMATH_CALUDE_two_valid_numbers_l767_76736


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l767_76783

theorem quadratic_rewrite :
  ∃ (a b c : ℤ), 
    (∀ x : ℝ, 4 * x^2 - 40 * x + 48 = (a * x + b)^2 + c) ∧
    a * b = -20 ∧
    c = -52 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l767_76783


namespace NUMINAMATH_CALUDE_central_projection_items_correct_l767_76761

-- Define the set of all items
inductive Item : Type
  | Searchlight
  | CarLight
  | Sun
  | Moon
  | DeskLamp

-- Define a predicate for items that form central projections
def FormsCentralProjection (item : Item) : Prop :=
  match item with
  | Item.Searchlight => True
  | Item.CarLight => True
  | Item.Sun => False
  | Item.Moon => False
  | Item.DeskLamp => True

-- Define the set of items that form central projections
def CentralProjectionItems : Set Item :=
  {item : Item | FormsCentralProjection item}

-- Theorem statement
theorem central_projection_items_correct :
  CentralProjectionItems = {Item.Searchlight, Item.CarLight, Item.DeskLamp} := by
  sorry


end NUMINAMATH_CALUDE_central_projection_items_correct_l767_76761


namespace NUMINAMATH_CALUDE_wengs_hourly_rate_l767_76728

/-- Weng's hourly rate given her earnings and work duration --/
theorem wengs_hourly_rate (work_duration : ℚ) (earnings : ℚ) : 
  work_duration = 50 / 60 → earnings = 10 → (earnings / work_duration) = 12 := by
  sorry

end NUMINAMATH_CALUDE_wengs_hourly_rate_l767_76728


namespace NUMINAMATH_CALUDE_sqrt_nine_minus_one_l767_76781

theorem sqrt_nine_minus_one : Real.sqrt 9 - 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_minus_one_l767_76781


namespace NUMINAMATH_CALUDE_four_distinct_three_digit_numbers_with_sum_divisibility_l767_76784

theorem four_distinct_three_digit_numbers_with_sum_divisibility :
  ∃ (a b c d : ℕ),
    100 ≤ a ∧ a < 1000 ∧
    100 ≤ b ∧ b < 1000 ∧
    100 ≤ c ∧ c < 1000 ∧
    100 ≤ d ∧ d < 1000 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (b + c + d) % a = 0 ∧
    (a + c + d) % b = 0 ∧
    (a + b + d) % c = 0 ∧
    (a + b + c) % d = 0 := by
  sorry

end NUMINAMATH_CALUDE_four_distinct_three_digit_numbers_with_sum_divisibility_l767_76784


namespace NUMINAMATH_CALUDE_polly_total_tweets_l767_76759

/-- Represents an emotional state or activity of Polly the parakeet -/
structure State where
  name : String
  tweets_per_minute : ℕ
  duration : ℕ

/-- Calculates the total number of tweets for a given state -/
def tweets_for_state (s : State) : ℕ := s.tweets_per_minute * s.duration

/-- The list of Polly's states during the day -/
def polly_states : List State := [
  { name := "Happy", tweets_per_minute := 18, duration := 50 },
  { name := "Hungry", tweets_per_minute := 4, duration := 35 },
  { name := "Watching reflection", tweets_per_minute := 45, duration := 30 },
  { name := "Sad", tweets_per_minute := 6, duration := 20 },
  { name := "Playing with toys", tweets_per_minute := 25, duration := 75 }
]

/-- Calculates the total number of tweets for all states -/
def total_tweets (states : List State) : ℕ :=
  states.map tweets_for_state |>.sum

/-- Theorem: The total number of tweets Polly makes during the day is 4385 -/
theorem polly_total_tweets : total_tweets polly_states = 4385 := by
  sorry

end NUMINAMATH_CALUDE_polly_total_tweets_l767_76759


namespace NUMINAMATH_CALUDE_total_price_theorem_l767_76767

/-- The price of a pear in dollars -/
def pear_price : ℕ := 90

/-- The total cost of an orange and a pear in dollars -/
def orange_pear_total : ℕ := 120

/-- The price of an orange in dollars -/
def orange_price : ℕ := orange_pear_total - pear_price

/-- The price of a banana in dollars -/
def banana_price : ℕ := pear_price - orange_price

/-- The number of bananas to buy -/
def num_bananas : ℕ := 200

/-- The number of oranges to buy -/
def num_oranges : ℕ := 2 * num_bananas

theorem total_price_theorem : 
  banana_price * num_bananas + orange_price * num_oranges = 24000 := by
  sorry

end NUMINAMATH_CALUDE_total_price_theorem_l767_76767


namespace NUMINAMATH_CALUDE_ellipse_equation_l767_76796

/-- Given an ellipse with foci at (-2,0) and (2,0) passing through the point (2√3, √3),
    its standard equation is x²/16 + y²/12 = 1 -/
theorem ellipse_equation (x y : ℝ) :
  let f1 : ℝ × ℝ := (-2, 0)
  let f2 : ℝ × ℝ := (2, 0)
  let p : ℝ × ℝ := (2 * Real.sqrt 3, Real.sqrt 3)
  let d1 := Real.sqrt ((x - f1.1)^2 + (y - f1.2)^2)
  let d2 := Real.sqrt ((x - f2.1)^2 + (y - f2.2)^2)
  let passing_point := d1 + d2 = Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2) +
                                 Real.sqrt ((p.1 - f2.1)^2 + (p.2 - f2.2)^2)
  passing_point → x^2 / 16 + y^2 / 12 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l767_76796


namespace NUMINAMATH_CALUDE_fifteen_percent_less_than_80_l767_76776

theorem fifteen_percent_less_than_80 : ∃ x : ℝ, x + (1/4) * x = 80 - 0.15 * 80 ∧ x = 54 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_less_than_80_l767_76776


namespace NUMINAMATH_CALUDE_toy_difference_l767_76787

/-- The number of toys each person has -/
structure ToyCount where
  mandy : ℕ
  anna : ℕ
  amanda : ℕ

/-- The conditions of the problem -/
def ProblemConditions (tc : ToyCount) : Prop :=
  tc.mandy = 20 ∧
  tc.anna = 3 * tc.mandy ∧
  tc.mandy + tc.anna + tc.amanda = 142 ∧
  tc.amanda > tc.anna

/-- The theorem to be proved -/
theorem toy_difference (tc : ToyCount) (h : ProblemConditions tc) : 
  tc.amanda - tc.anna = 2 := by
  sorry

end NUMINAMATH_CALUDE_toy_difference_l767_76787


namespace NUMINAMATH_CALUDE_cashew_price_in_mixture_l767_76729

/-- The price per pound of cashews in a mixture with peanuts -/
def cashew_price (peanut_price : ℚ) (total_weight : ℚ) (total_value : ℚ) (cashew_weight : ℚ) : ℚ :=
  (total_value - (total_weight - cashew_weight) * peanut_price) / cashew_weight

/-- Theorem stating the price of cashews in the given mixture -/
theorem cashew_price_in_mixture :
  cashew_price 2 25 92 11 = 64/11 := by
  sorry

end NUMINAMATH_CALUDE_cashew_price_in_mixture_l767_76729


namespace NUMINAMATH_CALUDE_system_solution_l767_76748

theorem system_solution : 
  let x : ℚ := 57 / 31
  let y : ℚ := 97 / 31
  (3 * x - 4 * y = -7) ∧ (4 * x + 5 * y = 23) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l767_76748


namespace NUMINAMATH_CALUDE_complex_roots_problem_l767_76755

theorem complex_roots_problem (p q r : ℂ) : 
  p + q + r = 2 ∧ 
  p * q * r = 2 ∧ 
  p * q + p * r + q * r = 0 → 
  (p = 2 ∧ q = Complex.I * Real.sqrt 2 ∧ r = -Complex.I * Real.sqrt 2) ∨
  (p = 2 ∧ q = -Complex.I * Real.sqrt 2 ∧ r = Complex.I * Real.sqrt 2) ∨
  (p = Complex.I * Real.sqrt 2 ∧ q = 2 ∧ r = -Complex.I * Real.sqrt 2) ∨
  (p = Complex.I * Real.sqrt 2 ∧ q = -Complex.I * Real.sqrt 2 ∧ r = 2) ∨
  (p = -Complex.I * Real.sqrt 2 ∧ q = 2 ∧ r = Complex.I * Real.sqrt 2) ∨
  (p = -Complex.I * Real.sqrt 2 ∧ q = Complex.I * Real.sqrt 2 ∧ r = 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_roots_problem_l767_76755


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l767_76707

theorem quadratic_factorization_sum (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 9*x + 20 = (x + a)*(x + b)) →
  (∀ x : ℝ, x^2 + 11*x - 60 = (x + b)*(x - c)) →
  a + b + c = 23 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l767_76707


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l767_76734

/-- Given a point P₀ and a plane, this theorem states that the line passing through P₀ 
    and perpendicular to the plane has a specific equation. -/
theorem line_perpendicular_to_plane 
  (P₀ : ℝ × ℝ × ℝ) 
  (plane_normal : ℝ × ℝ × ℝ) 
  (plane_constant : ℝ) :
  let (x₀, y₀, z₀) := P₀
  let (a, b, c) := plane_normal
  (P₀ = (3, 4, 2) ∧ 
   plane_normal = (8, -4, 5) ∧ 
   plane_constant = -4) →
  (∀ (x y z : ℝ), 
    ((x - x₀) / a = (y - y₀) / b ∧ (y - y₀) / b = (z - z₀) / c) ↔
    (x, y, z) ∈ {p : ℝ × ℝ × ℝ | ∃ t : ℝ, p = (x₀ + a*t, y₀ + b*t, z₀ + c*t)}) :=
by sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l767_76734


namespace NUMINAMATH_CALUDE_wheelbarrow_sale_ratio_l767_76744

def duck_price : ℕ := 10
def chicken_price : ℕ := 8
def ducks_sold : ℕ := 2
def chickens_sold : ℕ := 5
def additional_earnings : ℕ := 60

def total_earnings : ℕ := duck_price * ducks_sold + chicken_price * chickens_sold

def wheelbarrow_cost : ℕ := total_earnings / 2

theorem wheelbarrow_sale_ratio :
  (wheelbarrow_cost + additional_earnings) / wheelbarrow_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_wheelbarrow_sale_ratio_l767_76744


namespace NUMINAMATH_CALUDE_regular_octahedron_parallel_edges_l767_76763

structure RegularOctahedron where
  vertices : Finset (Fin 6)
  edges : Finset (Fin 6 × Fin 6)
  faces : Finset (Fin 3 → Fin 6)
  vertex_count : vertices.card = 6
  edge_count : edges.card = 12
  face_count : faces.card = 8

def parallel_edges (o : RegularOctahedron) : Finset (Fin 6 × Fin 6 × Fin 6 × Fin 6) :=
  sorry

theorem regular_octahedron_parallel_edges (o : RegularOctahedron) :
  (parallel_edges o).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_octahedron_parallel_edges_l767_76763


namespace NUMINAMATH_CALUDE_min_distance_complex_circle_l767_76778

theorem min_distance_complex_circle (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 3 ∧ ∀ w : ℂ, Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≥ min_val :=
by
  sorry

end NUMINAMATH_CALUDE_min_distance_complex_circle_l767_76778


namespace NUMINAMATH_CALUDE_unique_a_value_l767_76750

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 4

-- Define the condition for the function to be positive outside [2, 8]
def positive_outside (a : ℝ) : Prop :=
  ∀ x, (x < 2 ∨ x > 8) → f a x > 0

-- Theorem statement
theorem unique_a_value : ∃! a : ℝ, positive_outside a :=
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l767_76750


namespace NUMINAMATH_CALUDE_triangle_inequality_l767_76711

theorem triangle_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_sum : a + b + c = 1) : 
  5 * (a^2 + b^2 + c^2) + 18 * a * b * c ≥ 7/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l767_76711


namespace NUMINAMATH_CALUDE_orange_probability_is_two_sevenths_l767_76756

/-- Represents the contents of a fruit basket -/
structure FruitBasket where
  oranges : ℕ
  apples : ℕ
  bananas : ℕ

/-- The initial state of the fruit basket -/
def initialBasket : FruitBasket := sorry

/-- The state of the basket after removing some fruits -/
def updatedBasket : FruitBasket := sorry

/-- The total number of fruits in the initial basket -/
def totalFruits : ℕ := 28

/-- Assertion that the updated basket has 3 oranges and 7 apples -/
axiom updated_basket_state : updatedBasket.oranges = 3 ∧ updatedBasket.apples = 7

/-- Assertion that 5 oranges and 3 apples were removed -/
axiom fruits_removed : initialBasket.oranges = updatedBasket.oranges + 5 ∧
                       initialBasket.apples = updatedBasket.apples + 3

/-- Assertion that the total number of fruits in the initial basket is correct -/
axiom initial_total_correct : initialBasket.oranges + initialBasket.apples + initialBasket.bananas = totalFruits

/-- The probability of choosing an orange from the initial basket -/
def orangeProbability : ℚ := sorry

/-- Theorem stating that the probability of choosing an orange is 2/7 -/
theorem orange_probability_is_two_sevenths : orangeProbability = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_orange_probability_is_two_sevenths_l767_76756


namespace NUMINAMATH_CALUDE_probability_four_ones_in_five_rolls_prob_four_ones_in_five_rolls_l767_76779

/-- The probability of rolling exactly 4 ones in 5 rolls of a fair six-sided die -/
theorem probability_four_ones_in_five_rolls : ℚ :=
  25 / 7776

/-- A fair six-sided die -/
def fair_six_sided_die : Finset ℕ := Finset.range 6

/-- The number of rolls -/
def num_rolls : ℕ := 5

/-- The number of desired ones -/
def desired_ones : ℕ := 4

/-- The probability of rolling a specific number on a fair six-sided die -/
def prob_single_roll (n : ℕ) : ℚ :=
  if n ∈ fair_six_sided_die then 1 / 6 else 0

/-- The main theorem: The probability of rolling exactly 4 ones in 5 rolls of a fair six-sided die is 25/7776 -/
theorem prob_four_ones_in_five_rolls :
  (Nat.choose num_rolls desired_ones) *
  (prob_single_roll 1) ^ desired_ones *
  (1 - prob_single_roll 1) ^ (num_rolls - desired_ones) =
  probability_four_ones_in_five_rolls := by sorry

end NUMINAMATH_CALUDE_probability_four_ones_in_five_rolls_prob_four_ones_in_five_rolls_l767_76779


namespace NUMINAMATH_CALUDE_infinite_series_convergence_l767_76751

theorem infinite_series_convergence : 
  let f (n : ℕ) := (n^3 + 2*n^2 + 5*n + 2) / (3^n * (n^3 + 3))
  ∑' (n : ℕ), f n = 1/2 := by sorry

end NUMINAMATH_CALUDE_infinite_series_convergence_l767_76751


namespace NUMINAMATH_CALUDE_sally_box_sales_l767_76719

theorem sally_box_sales (saturday_sales : ℕ) : 
  (saturday_sales + (3 / 2 : ℚ) * saturday_sales = 150) → 
  saturday_sales = 60 := by
sorry

end NUMINAMATH_CALUDE_sally_box_sales_l767_76719


namespace NUMINAMATH_CALUDE_first_digit_must_be_odd_l767_76724

/-- Represents a permutation of digits 0 to 9 -/
def Permutation := Fin 10 → Fin 10

/-- Checks if a permutation contains each digit exactly once -/
def is_valid_permutation (p : Permutation) : Prop :=
  ∀ i j : Fin 10, p i = p j → i = j

/-- Calculates the sum A as described in the problem -/
def sum_A (p : Permutation) : ℕ :=
  (10 * p 0 + p 1) + (10 * p 2 + p 3) + (10 * p 4 + p 5) + (10 * p 6 + p 7) + (10 * p 8 + p 9)

/-- Calculates the sum B as described in the problem -/
def sum_B (p : Permutation) : ℕ :=
  (10 * p 1 + p 2) + (10 * p 3 + p 4) + (10 * p 5 + p 6) + (10 * p 7 + p 8)

theorem first_digit_must_be_odd (p : Permutation) 
  (h_valid : is_valid_permutation p) 
  (h_equal : sum_A p = sum_B p) : 
  ¬ Even (p 0) :=
by sorry

end NUMINAMATH_CALUDE_first_digit_must_be_odd_l767_76724


namespace NUMINAMATH_CALUDE_min_value_quadratic_l767_76742

theorem min_value_quadratic (a b : ℝ) : 
  let f := fun x : ℝ ↦ x^2 - a*x + b
  (∃ r₁ ∈ Set.Icc (-1) 1, f r₁ = 0) →
  (∃ r₂ ∈ Set.Icc 1 2, f r₂ = 0) →
  ∃ m : ℝ, m = -1 ∧ ∀ x : ℝ, a - 2*b ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l767_76742


namespace NUMINAMATH_CALUDE_seahawks_touchdowns_l767_76770

theorem seahawks_touchdowns 
  (total_points : ℕ)
  (field_goals : ℕ)
  (touchdown_points : ℕ)
  (field_goal_points : ℕ)
  (h1 : total_points = 37)
  (h2 : field_goals = 3)
  (h3 : touchdown_points = 7)
  (h4 : field_goal_points = 3) :
  (total_points - field_goals * field_goal_points) / touchdown_points = 4 := by
  sorry

end NUMINAMATH_CALUDE_seahawks_touchdowns_l767_76770


namespace NUMINAMATH_CALUDE_chinese_barrel_stack_l767_76713

/-- Calculates the total number of barrels in a terraced stack --/
def totalBarrels (a b n : ℕ) : ℕ :=
  let c := a + n - 1
  let d := b + n - 1
  (n * ((2 * a + c) * b + (2 * c + a) * d + (d - b))) / 6

/-- The problem statement --/
theorem chinese_barrel_stack : totalBarrels 2 1 15 = 1360 := by
  sorry

end NUMINAMATH_CALUDE_chinese_barrel_stack_l767_76713


namespace NUMINAMATH_CALUDE_symmetric_about_one_empty_solution_set_implies_a_leq_one_at_most_one_intersection_l767_76705

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define evenness for a function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Proposition 2
theorem symmetric_about_one (h : is_even (fun x ↦ f (x + 1))) :
  ∀ x, f (1 + x) = f (1 - x) := by sorry

-- Proposition 3
theorem empty_solution_set_implies_a_leq_one (a : ℝ) :
  (∀ x, |x - 4| + |x - 3| ≥ a) → a ≤ 1 := by sorry

-- Proposition 4
theorem at_most_one_intersection (a : ℝ) :
  ∃! y, f a = y := by sorry

end NUMINAMATH_CALUDE_symmetric_about_one_empty_solution_set_implies_a_leq_one_at_most_one_intersection_l767_76705


namespace NUMINAMATH_CALUDE_max_value_polynomial_l767_76790

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  (∃ (max : ℝ), ∀ (a b : ℝ), a + b = 5 → 
    a^4*b + a^3*b + a^2*b + a*b + a*b^2 + a*b^3 + a*b^4 ≤ max) ∧
  (x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ 6084/17) ∧
  (∃ (x₀ y₀ : ℝ), x₀ + y₀ = 5 ∧ 
    x₀^4*y₀ + x₀^3*y₀ + x₀^2*y₀ + x₀*y₀ + x₀*y₀^2 + x₀*y₀^3 + x₀*y₀^4 = 6084/17) :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l767_76790


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l767_76746

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (m : Line) (α β : Plane) (hα : α ≠ β) :
  perpendicular m α → perpendicular m β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l767_76746


namespace NUMINAMATH_CALUDE_original_price_after_percentage_changes_l767_76735

theorem original_price_after_percentage_changes (p : ℝ) :
  let initial_price := (10000 : ℝ) / (10000 - p^2)
  let price_after_increase := initial_price * (1 + p / 100)
  let final_price := price_after_increase * (1 - p / 100)
  final_price = 1 :=
by sorry

end NUMINAMATH_CALUDE_original_price_after_percentage_changes_l767_76735


namespace NUMINAMATH_CALUDE_equivalence_of_divisibility_conditions_l767_76771

theorem equivalence_of_divisibility_conditions (f : ℕ → ℕ) :
  (∀ m n : ℕ+, m ≤ n → (f m + n : ℕ) ∣ (f n + m : ℕ)) ↔
  (∀ m n : ℕ+, m ≥ n → (f m + n : ℕ) ∣ (f n + m : ℕ)) :=
by sorry

end NUMINAMATH_CALUDE_equivalence_of_divisibility_conditions_l767_76771


namespace NUMINAMATH_CALUDE_orange_distribution_difference_l767_76721

/-- The number of students in the class -/
def num_students : ℕ := 25

/-- The initial total number of oranges -/
def total_oranges : ℕ := 240

/-- The number of bad oranges that were removed -/
def bad_oranges : ℕ := 65

/-- The difference in oranges per student before and after removing bad oranges -/
def orange_difference : ℚ := (total_oranges : ℚ) / num_students - ((total_oranges - bad_oranges) : ℚ) / num_students

theorem orange_distribution_difference :
  orange_difference = 2.6 := by sorry

end NUMINAMATH_CALUDE_orange_distribution_difference_l767_76721


namespace NUMINAMATH_CALUDE_christmas_tree_lights_l767_76731

/-- The number of yellow lights on a Christmas tree -/
def yellow_lights (total red blue : ℕ) : ℕ := total - (red + blue)

/-- Theorem: There are 37 yellow lights on the Christmas tree -/
theorem christmas_tree_lights : yellow_lights 95 26 32 = 37 := by
  sorry

end NUMINAMATH_CALUDE_christmas_tree_lights_l767_76731


namespace NUMINAMATH_CALUDE_unique_solution_equation_l767_76709

theorem unique_solution_equation :
  ∃! x : ℝ, (x^18 + 1) * (x^16 + x^14 + x^12 + x^10 + x^8 + x^6 + x^4 + x^2 + 1) = 18 * x^9 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l767_76709


namespace NUMINAMATH_CALUDE_x_fourth_power_zero_l767_76773

theorem x_fourth_power_zero (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (1 - x^2) + Real.sqrt (1 + x^2) = 2) : x^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_power_zero_l767_76773


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l767_76710

/-- Given that x and y are inversely proportional, prove that if x = 3y when x + y = 60, then y = 45 when x = 15. -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
  (h2 : ∃ x₀ y₀ : ℝ, x₀ = 3 * y₀ ∧ x₀ + y₀ = 60 ∧ x₀ * y₀ = k) :
  x = 15 → y = 45 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l767_76710


namespace NUMINAMATH_CALUDE_lunch_cost_theorem_l767_76789

/-- The cost of the Taco Grande Plate -/
def taco_grande_cost : ℝ := 8

/-- The cost of Mike's additional items -/
def mike_additional_cost : ℝ := 2 + 4 + 2

/-- Mike's total bill -/
def mike_bill : ℝ := taco_grande_cost + mike_additional_cost

/-- John's total bill -/
def john_bill : ℝ := taco_grande_cost

/-- The combined total cost of Mike and John's lunch -/
def combined_total_cost : ℝ := mike_bill + john_bill

theorem lunch_cost_theorem :
  (mike_bill = 2 * john_bill) → combined_total_cost = 24 := by
  sorry

#eval combined_total_cost

end NUMINAMATH_CALUDE_lunch_cost_theorem_l767_76789


namespace NUMINAMATH_CALUDE_chicken_count_l767_76717

theorem chicken_count (total_chickens : ℕ) (hens : ℕ) (roosters : ℕ) (chicks : ℕ) : 
  total_chickens = 15 → 
  hens = 3 → 
  roosters = total_chickens - hens → 
  chicks = roosters - 4 → 
  chicks = 8 := by
sorry

end NUMINAMATH_CALUDE_chicken_count_l767_76717


namespace NUMINAMATH_CALUDE_perpendicular_line_through_intersection_l767_76715

/-- The intersection point of two lines -/
def intersection_point (l1 l2 : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

/-- Check if a point satisfies a line equation -/
def satisfies_line (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop := sorry

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : ℝ → ℝ → Prop) : Prop := sorry

/-- The main theorem -/
theorem perpendicular_line_through_intersection :
  let l1 : ℝ → ℝ → Prop := λ x y => 2*x - y = 0
  let l2 : ℝ → ℝ → Prop := λ x y => x + y - 6 = 0
  let l3 : ℝ → ℝ → Prop := λ x y => 2*x + y - 1 = 0
  let l4 : ℝ → ℝ → Prop := λ x y => x - 2*y + 6 = 0
  let p := intersection_point l1 l2
  satisfies_line p l4 ∧ perpendicular l3 l4 := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_intersection_l767_76715


namespace NUMINAMATH_CALUDE_ivans_initial_money_l767_76753

theorem ivans_initial_money (initial_money : ℝ) : 
  (4/5 * initial_money - 5 = 3) → initial_money = 10 := by
  sorry

end NUMINAMATH_CALUDE_ivans_initial_money_l767_76753


namespace NUMINAMATH_CALUDE_smaller_square_side_length_l767_76752

/-- A square with side length 2 -/
structure Square :=
  (side : ℝ)
  (is_two : side = 2)

/-- An equilateral triangle with vertices P, T, U where T is on RS and U is on SQ of square PQRS -/
structure EquilateralTriangle (sq : Square) :=
  (P T U : ℝ × ℝ)
  (is_equilateral : sorry)
  (T_on_RS : sorry)
  (U_on_SQ : sorry)

/-- A smaller square with vertex R and a vertex on PT -/
structure SmallerSquare (sq : Square) (tri : EquilateralTriangle sq) :=
  (side : ℝ)
  (vertex_on_PT : sorry)
  (sides_parallel : sorry)

/-- The theorem stating the properties of the smaller square's side length -/
theorem smaller_square_side_length 
  (sq : Square) 
  (tri : EquilateralTriangle sq) 
  (small_sq : SmallerSquare sq tri) :
  ∃ (d e f : ℕ), 
    d > 0 ∧ e > 0 ∧ f > 0 ∧
    ¬ (∃ (p : ℕ), Prime p ∧ p^2 ∣ e) ∧
    small_sq.side = (d - Real.sqrt e) / f ∧
    d = 4 ∧ e = 10 ∧ f = 3 ∧
    d + e + f = 17 := by
  sorry

end NUMINAMATH_CALUDE_smaller_square_side_length_l767_76752


namespace NUMINAMATH_CALUDE_proposition_p_and_q_true_l767_76799

open Real

theorem proposition_p_and_q_true : 
  (∃ φ : ℝ, (φ = π / 2 ∧ 
    (∀ x : ℝ, sin (x + φ) = sin (-x - φ)) ∧
    (∃ ψ : ℝ, ψ ≠ π / 2 ∧ ∀ x : ℝ, sin (x + ψ) = sin (-x - ψ)))) ∧
  (∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < π / 2 ∧ sin x₀ ≠ 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_and_q_true_l767_76799


namespace NUMINAMATH_CALUDE_polynomial_roots_l767_76774

theorem polynomial_roots : ∃ (a b c d : ℂ),
  (a = (1 + Real.sqrt 5) / 2) ∧
  (b = (1 - Real.sqrt 5) / 2) ∧
  (c = (3 + Real.sqrt 13) / 6) ∧
  (d = (3 - Real.sqrt 13) / 6) ∧
  (∀ x : ℂ, 3 * x^4 - 4 * x^3 - 5 * x^2 - 4 * x + 3 = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l767_76774


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l767_76798

theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_intersect : ∃ (x y : ℝ), y = 2*x ∧ x^2/a^2 - y^2/b^2 = 1) :
  let e := Real.sqrt (1 + (b/a)^2)
  e > Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l767_76798


namespace NUMINAMATH_CALUDE_sqrt_sum_equation_solutions_l767_76727

theorem sqrt_sum_equation_solutions :
  ∀ x : ℝ, (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equation_solutions_l767_76727


namespace NUMINAMATH_CALUDE_bike_ride_distance_l767_76785

/-- Calculates the total distance ridden given a constant riding rate and time, including breaks -/
def total_distance (rate : ℚ) (total_time : ℚ) (break_time : ℚ) (num_breaks : ℕ) : ℚ :=
  rate * (total_time - (break_time * num_breaks))

/-- The theorem to be proved -/
theorem bike_ride_distance :
  let rate : ℚ := 2 / 10  -- 2 miles per 10 minutes
  let total_time : ℚ := 40  -- 40 minutes total time
  let break_time : ℚ := 5  -- 5 minutes per break
  let num_breaks : ℕ := 2  -- 2 breaks
  total_distance rate total_time break_time num_breaks = 6 := by
  sorry


end NUMINAMATH_CALUDE_bike_ride_distance_l767_76785


namespace NUMINAMATH_CALUDE_meeting_organization_count_l767_76791

/-- The number of ways to organize a leadership meeting -/
def organize_meeting (total_schools : ℕ) (members_per_school : ℕ) 
  (host_representatives : ℕ) (other_representatives : ℕ) : ℕ :=
  total_schools * (members_per_school.choose host_representatives) * 
  ((members_per_school.choose other_representatives) ^ (total_schools - 1))

/-- Theorem stating the number of ways to organize the meeting -/
theorem meeting_organization_count :
  organize_meeting 4 6 3 1 = 17280 := by
  sorry

end NUMINAMATH_CALUDE_meeting_organization_count_l767_76791


namespace NUMINAMATH_CALUDE_lcm_225_624_l767_76782

theorem lcm_225_624 : Nat.lcm 225 624 = 46800 := by
  sorry

end NUMINAMATH_CALUDE_lcm_225_624_l767_76782


namespace NUMINAMATH_CALUDE_average_growth_rate_inequality_l767_76768

theorem average_growth_rate_inequality (a p q x : ℝ) 
  (h1 : a > 0) 
  (h2 : p ≥ 0) 
  (h3 : q ≥ 0) 
  (h4 : a * (1 + p) * (1 + q) = a * (1 + x)^2) : 
  x ≤ (p + q) / 2 := by
sorry

end NUMINAMATH_CALUDE_average_growth_rate_inequality_l767_76768


namespace NUMINAMATH_CALUDE_b_current_age_l767_76747

/-- Given two people A and B, proves B's current age is 39 years -/
theorem b_current_age (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) →  -- A's age in 10 years equals twice B's age 10 years ago
  (a = b + 9) →              -- A is currently 9 years older than B
  b = 39 := by               -- B's current age is 39 years
sorry


end NUMINAMATH_CALUDE_b_current_age_l767_76747


namespace NUMINAMATH_CALUDE_can_guess_majority_winners_l767_76716

/-- Represents a tennis tournament with n players -/
structure TennisTournament (n : ℕ) where
  /-- Final scores of each player -/
  scores : Fin n → ℕ
  /-- Total number of matches in the tournament -/
  total_matches : ℕ := n * (n - 1) / 2

/-- Theorem stating that it's possible to guess more than half of the match winners -/
theorem can_guess_majority_winners (n : ℕ) (tournament : TennisTournament n) :
  ∃ (guessed_matches : ℕ), guessed_matches > tournament.total_matches / 2 :=
sorry

end NUMINAMATH_CALUDE_can_guess_majority_winners_l767_76716


namespace NUMINAMATH_CALUDE_tan_alpha_value_l767_76793

theorem tan_alpha_value (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = -5) : 
  Real.tan α = -23/16 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l767_76793


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l767_76700

theorem arithmetic_simplification : 2537 + 240 * 3 / 60 - 347 = 2202 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l767_76700


namespace NUMINAMATH_CALUDE_machine_value_theorem_l767_76723

/-- Calculates the machine's value after two years and a major overhaul -/
def machine_value_after_two_years_and_overhaul (initial_value : ℝ) : ℝ :=
  let year1_depreciation_rate := 0.10
  let year2_depreciation_rate := 0.12
  let repair_rate := 0.03
  let overhaul_rate := 0.15
  
  let value_after_year1 := initial_value * (1 - year1_depreciation_rate) * (1 + repair_rate)
  let value_after_year2 := value_after_year1 * (1 - year2_depreciation_rate) * (1 + repair_rate)
  let final_value := value_after_year2 * (1 - overhaul_rate)
  
  final_value

/-- Theorem stating that the machine's value after two years and a major overhaul 
    is approximately $863.23, given an initial value of $1200 -/
theorem machine_value_theorem :
  ∃ ε > 0, abs (machine_value_after_two_years_and_overhaul 1200 - 863.23) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_machine_value_theorem_l767_76723


namespace NUMINAMATH_CALUDE_coffee_price_coffee_price_is_12_l767_76772

/-- The regular price for a half-pound of coffee, given a 25% discount and 
    quarter-pound bags sold for $4.50 after the discount. -/
theorem coffee_price : ℝ :=
  let discount_rate : ℝ := 0.25
  let discounted_price_quarter_pound : ℝ := 4.50
  let regular_price_half_pound : ℝ := 12

  have h1 : discounted_price_quarter_pound = 
    regular_price_half_pound / 2 * (1 - discount_rate) := by sorry

  regular_price_half_pound

/-- Proof that the regular price for a half-pound of coffee is $12 -/
theorem coffee_price_is_12 : coffee_price = 12 := by sorry

end NUMINAMATH_CALUDE_coffee_price_coffee_price_is_12_l767_76772


namespace NUMINAMATH_CALUDE_second_car_distance_rate_l767_76754

/-- Represents the race scenario with two cars and a motorcycle --/
structure RaceScenario where
  l : ℝ  -- Length of the race distance
  v1 : ℝ  -- Speed of the first car
  v2 : ℝ  -- Speed of the second car
  vM : ℝ  -- Speed of the motorcycle

/-- Conditions of the race --/
def race_conditions (r : RaceScenario) : Prop :=
  r.l > 0 ∧  -- The race distance is positive
  r.v1 > 0 ∧ r.v2 > 0 ∧ r.vM > 0 ∧  -- All speeds are positive
  r.l / r.v2 - r.l / r.v1 = 1/60 ∧  -- Second car takes 1 minute longer than the first car
  r.v1 = 4 * r.vM ∧  -- First car is 4 times faster than the motorcycle
  r.v2 / 60 - r.vM / 60 = r.l / 6 ∧  -- Second car covers 1/6 more distance per minute than the motorcycle
  r.l / r.vM < 10  -- Motorcycle covers the distance in less than 10 minutes

/-- The theorem to be proved --/
theorem second_car_distance_rate (r : RaceScenario) :
  race_conditions r → r.v2 / 60 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_second_car_distance_rate_l767_76754


namespace NUMINAMATH_CALUDE_sphere_cube_intersection_areas_l767_76730

/-- Given a cube with edge length a and a sphere circumscribed around it, 
    this theorem proves the areas of the sections formed by the intersection 
    of the sphere and the cube's faces. -/
theorem sphere_cube_intersection_areas (a : ℝ) (h : a > 0) :
  let R := a * Real.sqrt 3 / 2
  ∃ (bicorn_area curvilinear_quad_area : ℝ),
    bicorn_area = π * a^2 * (2 - Real.sqrt 3) / 4 ∧
    curvilinear_quad_area = π * a^2 * (Real.sqrt 3 - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sphere_cube_intersection_areas_l767_76730


namespace NUMINAMATH_CALUDE_solve_equation_l767_76737

theorem solve_equation : ∃ x : ℝ, 
  ((0.66^3 - 0.1^3) / 0.66^2) + x + 0.1^2 = 0.5599999999999999 ∧ 
  x = -0.107504 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l767_76737


namespace NUMINAMATH_CALUDE_quadratic_coefficient_positive_l767_76720

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  a_nonzero : a ≠ 0

/-- Evaluation of a quadratic polynomial at a given point -/
def evaluate (p : QuadraticPolynomial) (x : ℤ) : ℤ :=
  p.a * x * x + p.b * x + p.c

/-- The main theorem -/
theorem quadratic_coefficient_positive
  (p : QuadraticPolynomial)
  (n : ℤ)
  (h : n < evaluate p n ∧ evaluate p n < evaluate p (evaluate p n) ∧
       evaluate p (evaluate p n) < evaluate p (evaluate p (evaluate p n))) :
  0 < p.a :=
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_positive_l767_76720


namespace NUMINAMATH_CALUDE_part_one_calculation_part_two_calculation_part_three_calculation_l767_76703

-- Part 1
theorem part_one_calculation : -12 - (-18) + (-7) = -1 := by sorry

-- Part 2
theorem part_two_calculation : (4/7 - 1/9 + 2/21) * (-63) = -35 := by sorry

-- Part 3
theorem part_three_calculation : (-4)^2 / 2 + 9 * (-1/3) - |3 - 4| = 4 := by sorry

end NUMINAMATH_CALUDE_part_one_calculation_part_two_calculation_part_three_calculation_l767_76703


namespace NUMINAMATH_CALUDE_evaluate_expression_l767_76739

theorem evaluate_expression : 2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3 = 4019 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l767_76739


namespace NUMINAMATH_CALUDE_floor_of_negative_two_point_seven_l767_76743

theorem floor_of_negative_two_point_seven : ⌊(-2.7 : ℝ)⌋ = -3 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_negative_two_point_seven_l767_76743


namespace NUMINAMATH_CALUDE_cone_areas_l767_76764

/-- Represents a cone with given slant height and height -/
structure Cone where
  slantHeight : ℝ
  height : ℝ

/-- Calculates the lateral area of a cone -/
def lateralArea (c : Cone) : ℝ := sorry

/-- Calculates the area of the sector when the cone's lateral surface is unfolded -/
def sectorArea (c : Cone) : ℝ := sorry

theorem cone_areas (c : Cone) (h1 : c.slantHeight = 1) (h2 : c.height = 0.8) : 
  lateralArea c = 3/5 * Real.pi ∧ sectorArea c = 3/5 * Real.pi := by sorry

end NUMINAMATH_CALUDE_cone_areas_l767_76764


namespace NUMINAMATH_CALUDE_diameter_angle_property_l767_76732

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define a function to check if a point is inside a circle
def isInside (c : Circle) (p : Point) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

-- Define a function to check if a point is outside a circle
def isOutside (c : Circle) (p : Point) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 > c.radius^2

-- Define a function to calculate the angle between three points
noncomputable def angle (a b c : Point) : ℝ := sorry

-- Theorem statement
theorem diameter_angle_property (c : Circle) (a b : Point) :
  (∀ (x : ℝ × ℝ), x.1 = a.1 ∧ x.2 = b.2 → isInside c x) →  -- a and b are on opposite sides of the circle
  (∀ (p : Point), isInside c p → angle a p b > Real.pi / 2) ∧
  (∀ (p : Point), isOutside c p → angle a p b < Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_diameter_angle_property_l767_76732


namespace NUMINAMATH_CALUDE_area_of_non_intersecting_graphs_l767_76749

/-- The area of the set A of points (a, b) such that the graphs of 
    f(x) = x^2 - 2ax + 1 and g(x) = 2b(a-x) do not intersect is π. -/
theorem area_of_non_intersecting_graphs (a b x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - 2*a*x + 1
  let g : ℝ → ℝ := λ x => 2*b*(a-x)
  let A : Set (ℝ × ℝ) := {(a, b) | ∀ x, f x ≠ g x}
  MeasureTheory.volume A = π := by
sorry

end NUMINAMATH_CALUDE_area_of_non_intersecting_graphs_l767_76749


namespace NUMINAMATH_CALUDE_chef_apples_left_l767_76714

/-- The number of apples the chef has left after making a pie -/
def applesLeft (initialApples usedApples : ℕ) : ℕ :=
  initialApples - usedApples

theorem chef_apples_left : applesLeft 19 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_chef_apples_left_l767_76714


namespace NUMINAMATH_CALUDE_rohans_savings_l767_76718

/-- Rohan's monthly savings calculation -/
theorem rohans_savings (salary : ℝ) (food_percent : ℝ) (rent_percent : ℝ) 
  (entertainment_percent : ℝ) (conveyance_percent : ℝ) : 
  salary = 12500 ∧ 
  food_percent = 40 ∧ 
  rent_percent = 20 ∧ 
  entertainment_percent = 10 ∧ 
  conveyance_percent = 10 → 
  salary * (1 - (food_percent + rent_percent + entertainment_percent + conveyance_percent) / 100) = 2500 :=
by sorry

end NUMINAMATH_CALUDE_rohans_savings_l767_76718


namespace NUMINAMATH_CALUDE_b_finishing_time_l767_76760

/-- The number of days it takes B to finish the remaining work after A leaves -/
def days_for_B_to_finish (a_days b_days collab_days : ℚ) : ℚ :=
  let total_work := 1
  let a_rate := 1 / a_days
  let b_rate := 1 / b_days
  let combined_rate := a_rate + b_rate
  let work_done_together := combined_rate * collab_days
  let remaining_work := total_work - work_done_together
  remaining_work / b_rate

/-- Theorem stating that B will take 76/5 days to finish the remaining work -/
theorem b_finishing_time :
  days_for_B_to_finish 5 16 2 = 76 / 5 := by
  sorry

end NUMINAMATH_CALUDE_b_finishing_time_l767_76760


namespace NUMINAMATH_CALUDE_count_initials_sets_l767_76788

/-- The number of letters available for initials -/
def num_letters : ℕ := 10

/-- The number of letters in each set of initials -/
def initials_length : ℕ := 4

/-- The number of different four-letter sets of initials possible using letters A through J -/
theorem count_initials_sets : (num_letters ^ initials_length : ℕ) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_count_initials_sets_l767_76788


namespace NUMINAMATH_CALUDE_symmetry_origin_symmetry_y_eq_x_vertex_2_neg_2_l767_76797

-- Define the curve E
def E (x y : ℝ) : Prop := x^2 + x*y + y^2 = 4

-- Symmetry with respect to the origin
theorem symmetry_origin : ∀ x y : ℝ, E x y ↔ E (-x) (-y) := by sorry

-- Symmetry with respect to the line y = x
theorem symmetry_y_eq_x : ∀ x y : ℝ, E x y ↔ E y x := by sorry

-- (2, -2) is a vertex of E
theorem vertex_2_neg_2 : E 2 (-2) ∧ (∃ ε > 0, ∀ x y : ℝ, 
  (x - 2)^2 + (y + 2)^2 < ε^2 → E x y → x^2 + y^2 ≥ 2^2 + (-2)^2) := by sorry

end NUMINAMATH_CALUDE_symmetry_origin_symmetry_y_eq_x_vertex_2_neg_2_l767_76797


namespace NUMINAMATH_CALUDE_region_volume_l767_76794

-- Define the region in three-dimensional space
def region (x y z : ℝ) : Prop :=
  abs x + abs y + abs z ≤ 1 ∧ abs x + abs y + abs (z - 2) ≤ 1

-- Define the volume of a region
noncomputable def volume (R : ℝ → ℝ → ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem region_volume : volume region = 1/12 := by sorry

end NUMINAMATH_CALUDE_region_volume_l767_76794


namespace NUMINAMATH_CALUDE_polynomial_identity_l767_76725

theorem polynomial_identity (P : ℝ → ℝ) : 
  (∀ a b c : ℝ, a * b + b * c + c * a = 0 → 
    P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)) → 
  ∃ α β : ℝ, ∀ x : ℝ, P x = α * x^4 + β * x^2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_l767_76725


namespace NUMINAMATH_CALUDE_cylindrical_tin_height_l767_76740

/-- The height of a cylindrical tin given its diameter and volume -/
theorem cylindrical_tin_height (diameter : ℝ) (volume : ℝ) (h_diameter : diameter = 8) (h_volume : volume = 80) :
  (volume / (π * (diameter / 2)^2)) = 80 / (π * 4^2) :=
by sorry

end NUMINAMATH_CALUDE_cylindrical_tin_height_l767_76740


namespace NUMINAMATH_CALUDE_matrix_square_result_l767_76706

theorem matrix_square_result (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : M.mulVec ![1, 0] = ![1, 0])
  (h2 : M.mulVec ![1, 1] = ![2, 2]) :
  (M ^ 2).mulVec ![1, -1] = ![-2, -4] := by
  sorry

end NUMINAMATH_CALUDE_matrix_square_result_l767_76706


namespace NUMINAMATH_CALUDE_sum_of_unknowns_l767_76777

theorem sum_of_unknowns (x₁ x₂ x₃ : ℝ) 
  (h : (1 + 2 + 3 + 4 + x₁ + x₂ + x₃) / 7 = 8) : 
  x₁ + x₂ + x₃ = 46 := by
sorry

end NUMINAMATH_CALUDE_sum_of_unknowns_l767_76777


namespace NUMINAMATH_CALUDE_cookie_jar_spending_ratio_l767_76769

theorem cookie_jar_spending_ratio (initial_amount : ℕ) (doris_spent : ℕ) (final_amount : ℕ) : 
  initial_amount = 21 →
  doris_spent = 6 →
  final_amount = 12 →
  (initial_amount - doris_spent - final_amount) / doris_spent = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_jar_spending_ratio_l767_76769


namespace NUMINAMATH_CALUDE_sum_product_difference_l767_76704

theorem sum_product_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x * y = 96) : 
  |x - y| = 4 := by sorry

end NUMINAMATH_CALUDE_sum_product_difference_l767_76704


namespace NUMINAMATH_CALUDE_l_shaped_area_specific_l767_76708

/-- Calculates the area of an 'L'-shaped figure formed by removing a smaller rectangle from a larger rectangle. -/
def l_shaped_area (large_width large_height small_width small_height : ℕ) : ℕ :=
  large_width * large_height - small_width * small_height

theorem l_shaped_area_specific : l_shaped_area 10 7 4 3 = 58 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_area_specific_l767_76708


namespace NUMINAMATH_CALUDE_tape_overlap_length_l767_76726

theorem tape_overlap_length 
  (num_pieces : ℕ) 
  (piece_length : ℝ) 
  (total_overlapped_length : ℝ) 
  (h1 : num_pieces = 4) 
  (h2 : piece_length = 250) 
  (h3 : total_overlapped_length = 925) :
  (num_pieces * piece_length - total_overlapped_length) / (num_pieces - 1) = 25 := by
sorry

end NUMINAMATH_CALUDE_tape_overlap_length_l767_76726


namespace NUMINAMATH_CALUDE_polynomial_composition_problem_l767_76757

-- Define the polynomial P
def P (x : ℝ) : ℝ := x^2 - 1

-- Define the theorem
theorem polynomial_composition_problem (a : ℝ) (m n : ℕ) 
  (h1 : a > 0)
  (h2 : P (P (P a)) = 99)
  (h3 : a^2 = m + Real.sqrt n)
  (h4 : n > 0)
  (h5 : ∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ n)) :
  m + n = 12 := by
sorry

end NUMINAMATH_CALUDE_polynomial_composition_problem_l767_76757


namespace NUMINAMATH_CALUDE_notebook_increase_l767_76795

theorem notebook_increase (initial_count mother_bought father_bought : ℕ) :
  initial_count = 33 →
  mother_bought = 7 →
  father_bought = 14 →
  (initial_count + mother_bought + father_bought) - initial_count = 21 := by
  sorry

end NUMINAMATH_CALUDE_notebook_increase_l767_76795
