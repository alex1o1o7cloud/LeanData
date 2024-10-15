import Mathlib

namespace NUMINAMATH_CALUDE_stones_required_l1487_148720

def hall_length : ℝ := 45
def hall_width : ℝ := 25
def stone_length : ℝ := 1.2  -- 12 dm = 1.2 m
def stone_width : ℝ := 0.7   -- 7 dm = 0.7 m

theorem stones_required :
  ⌈(hall_length * hall_width) / (stone_length * stone_width)⌉ = 1341 := by
  sorry

end NUMINAMATH_CALUDE_stones_required_l1487_148720


namespace NUMINAMATH_CALUDE_complement_of_M_l1487_148740

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M : (U \ M) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l1487_148740


namespace NUMINAMATH_CALUDE_ratio_percent_problem_l1487_148731

theorem ratio_percent_problem (ratio_percent : ℝ) (second_part : ℝ) (first_part : ℝ) :
  ratio_percent = 50 →
  second_part = 20 →
  first_part = ratio_percent / 100 * second_part →
  first_part = 10 :=
by sorry

end NUMINAMATH_CALUDE_ratio_percent_problem_l1487_148731


namespace NUMINAMATH_CALUDE_beyonce_song_count_l1487_148707

/-- The total number of songs released by Beyonce -/
def total_songs (singles : ℕ) (albums_15 : ℕ) (songs_per_album_15 : ℕ) (albums_20 : ℕ) (songs_per_album_20 : ℕ) : ℕ :=
  singles + albums_15 * songs_per_album_15 + albums_20 * songs_per_album_20

/-- Theorem stating that Beyonce has released 55 songs in total -/
theorem beyonce_song_count :
  total_songs 5 2 15 1 20 = 55 := by
  sorry


end NUMINAMATH_CALUDE_beyonce_song_count_l1487_148707


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1487_148795

theorem inequality_equivalence (x : ℝ) : (1/2 * x - 1 > 0) ↔ (x > 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1487_148795


namespace NUMINAMATH_CALUDE_bananas_per_box_l1487_148775

/-- Given 40 bananas and 8 boxes, prove that the number of bananas per box is 5. -/
theorem bananas_per_box (total_bananas : ℕ) (num_boxes : ℕ) 
  (h1 : total_bananas = 40) (h2 : num_boxes = 8) : 
  total_bananas / num_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_bananas_per_box_l1487_148775


namespace NUMINAMATH_CALUDE_balance_balls_l1487_148751

-- Define the weights of balls relative to blue balls
def red_weight : ℚ := 2
def orange_weight : ℚ := 7/3
def silver_weight : ℚ := 5/3

-- Theorem statement
theorem balance_balls :
  5 * red_weight + 3 * orange_weight + 4 * silver_weight = 71/3 := by
  sorry

end NUMINAMATH_CALUDE_balance_balls_l1487_148751


namespace NUMINAMATH_CALUDE_stamp_collection_problem_l1487_148721

/-- The number of red stamps Simon has -/
def simon_red_stamps : ℕ := 30

/-- The price of a red stamp in cents -/
def red_stamp_price : ℕ := 50

/-- The price of a white stamp in cents -/
def white_stamp_price : ℕ := 20

/-- The difference in earnings between Simon and Peter in dollars -/
def earnings_difference : ℚ := 1

/-- The number of white stamps Peter has -/
def peter_white_stamps : ℕ := 70

theorem stamp_collection_problem :
  (simon_red_stamps * red_stamp_price : ℚ) / 100 - 
  (peter_white_stamps * white_stamp_price : ℚ) / 100 = earnings_difference :=
sorry

end NUMINAMATH_CALUDE_stamp_collection_problem_l1487_148721


namespace NUMINAMATH_CALUDE_inequality_solution_l1487_148712

theorem inequality_solution : ∃! x : ℝ, 
  (Real.sqrt (x^3 - 10*x + 7) + 1) * abs (x^3 - 18*x + 28) ≤ 0 ∧
  x^3 - 10*x + 7 ≥ 0 :=
by
  -- The unique solution is x = -1 + √15
  use -1 + Real.sqrt 15
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1487_148712


namespace NUMINAMATH_CALUDE_calculate_premium_percentage_l1487_148739

/-- Given an investment scenario, calculate the premium percentage on shares. -/
theorem calculate_premium_percentage
  (total_investment : ℝ)
  (face_value : ℝ)
  (dividend_rate : ℝ)
  (total_dividend : ℝ)
  (h1 : total_investment = 14400)
  (h2 : face_value = 100)
  (h3 : dividend_rate = 0.05)
  (h4 : total_dividend = 600) :
  (total_investment / (total_dividend / (dividend_rate * face_value)) - face_value) / face_value * 100 = 20 := by
sorry


end NUMINAMATH_CALUDE_calculate_premium_percentage_l1487_148739


namespace NUMINAMATH_CALUDE_tangerine_problem_l1487_148716

/-- The number of tangerines initially in the basket -/
def initial_tangerines : ℕ := 24

/-- The number of tangerines Eunji ate -/
def eaten_tangerines : ℕ := 9

/-- The number of tangerines Eunji's mother added -/
def added_tangerines : ℕ := 5

/-- The final number of tangerines in the basket -/
def final_tangerines : ℕ := 20

theorem tangerine_problem :
  initial_tangerines - eaten_tangerines + added_tangerines = final_tangerines :=
by sorry

end NUMINAMATH_CALUDE_tangerine_problem_l1487_148716


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l1487_148748

theorem circle_symmetry_line (a b : ℝ) :
  let circle := fun (x y : ℝ) => x^2 + y^2 - 4*x + 2*y + 1 = 0
  let line := fun (x y : ℝ) => a*x - 2*b*y - 1 = 0
  let symmetric := ∀ (x y : ℝ), circle x y → (∃ (x' y' : ℝ), circle x' y' ∧ line ((x + x')/2) ((y + y')/2))
  symmetric → a*b ≤ 1/16 :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l1487_148748


namespace NUMINAMATH_CALUDE_investment_duration_l1487_148780

/-- Given an investment with simple interest, prove the duration is 2.5 years -/
theorem investment_duration (principal interest_rate interest : ℝ) 
  (h1 : principal = 7200)
  (h2 : interest_rate = 17.5)
  (h3 : interest = 3150) :
  interest = principal * interest_rate * 2.5 / 100 := by
  sorry

end NUMINAMATH_CALUDE_investment_duration_l1487_148780


namespace NUMINAMATH_CALUDE_henry_finishes_before_zoe_l1487_148717

/-- Represents the race parameters and results -/
structure RaceData where
  distance : ℕ  -- race distance in miles
  zoe_pace : ℕ  -- Zoe's pace in minutes per mile
  henry_pace : ℕ  -- Henry's pace in minutes per mile

/-- Calculates the time difference between Zoe and Henry finishing the race -/
def timeDifference (race : RaceData) : Int :=
  race.zoe_pace * race.distance - race.henry_pace * race.distance

/-- Theorem stating that Henry finishes 24 minutes before Zoe in the given race conditions -/
theorem henry_finishes_before_zoe (race : RaceData) 
  (h1 : race.distance = 12)
  (h2 : race.zoe_pace = 9)
  (h3 : race.henry_pace = 7) : 
  timeDifference race = 24 := by
  sorry

end NUMINAMATH_CALUDE_henry_finishes_before_zoe_l1487_148717


namespace NUMINAMATH_CALUDE_expected_rank_is_103_l1487_148779

/-- Represents a tennis tournament with the given conditions -/
structure TennisTournament where
  num_players : ℕ
  num_rounds : ℕ
  win_prob : ℚ

/-- Calculates the expected rank of the winner in a tennis tournament -/
def expected_rank (t : TennisTournament) : ℚ :=
  sorry

/-- The specific tournament described in the problem -/
def specific_tournament : TennisTournament :=
  { num_players := 256
  , num_rounds := 8
  , win_prob := 3/5 }

/-- Theorem stating that the expected rank of the winner in the specific tournament is 103 -/
theorem expected_rank_is_103 : expected_rank specific_tournament = 103 :=
  sorry

end NUMINAMATH_CALUDE_expected_rank_is_103_l1487_148779


namespace NUMINAMATH_CALUDE_solution_in_interval_one_two_l1487_148794

theorem solution_in_interval_one_two :
  ∃ x : ℝ, 2^x + x = 4 ∧ x ∈ Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_solution_in_interval_one_two_l1487_148794


namespace NUMINAMATH_CALUDE_women_on_bus_l1487_148718

theorem women_on_bus (total : ℕ) (men : ℕ) (children : ℕ) 
  (h1 : total = 54)
  (h2 : men = 18)
  (h3 : children = 10) :
  total - men - children = 26 := by
  sorry

end NUMINAMATH_CALUDE_women_on_bus_l1487_148718


namespace NUMINAMATH_CALUDE_basketball_tournament_l1487_148791

theorem basketball_tournament (n : ℕ) : n * (n - 1) / 2 = 10 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_basketball_tournament_l1487_148791


namespace NUMINAMATH_CALUDE_marble_ratio_proof_l1487_148713

theorem marble_ratio_proof (mabel_marbles katrina_marbles amanda_marbles : ℕ) 
  (h1 : mabel_marbles = 5 * katrina_marbles)
  (h2 : mabel_marbles = 85)
  (h3 : mabel_marbles = amanda_marbles + 63)
  : (amanda_marbles + 12) / katrina_marbles = 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_proof_l1487_148713


namespace NUMINAMATH_CALUDE_mikes_net_spending_l1487_148764

/-- The net amount Mike spent at the music store -/
def net_spent (trumpet_cost discount_percent stand_cost sheet_music_cost songbook_sale : ℚ) : ℚ :=
  (trumpet_cost * (1 - discount_percent / 100) + stand_cost + sheet_music_cost) - songbook_sale

/-- Theorem stating that Mike's net spending at the music store is $209.16 -/
theorem mikes_net_spending :
  net_spent 250 30 25 15 5.84 = 209.16 := by
  sorry

end NUMINAMATH_CALUDE_mikes_net_spending_l1487_148764


namespace NUMINAMATH_CALUDE_cubic_polynomials_with_rational_roots_l1487_148715

/-- A cubic polynomial with rational coefficients -/
structure CubicPolynomial where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The set of cubic polynomials with all rational roots -/
def CubicPolynomialsWithRationalRoots : Set CubicPolynomial :=
  {p : CubicPolynomial | ∃ (r₁ r₂ r₃ : ℚ),
    ∀ x, x^3 + p.a * x^2 + p.b * x + p.c = (x - r₁) * (x - r₂) * (x - r₃)}

/-- The two specific polynomials from the solution -/
def f₁ : CubicPolynomial := ⟨1, -2, 0⟩
def f₂ : CubicPolynomial := ⟨1, -1, -1⟩

/-- The main theorem stating that f₁ and f₂ are the only cubic polynomials with all rational roots -/
theorem cubic_polynomials_with_rational_roots :
  CubicPolynomialsWithRationalRoots = {f₁, f₂} :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomials_with_rational_roots_l1487_148715


namespace NUMINAMATH_CALUDE_square_areas_and_perimeters_l1487_148787

theorem square_areas_and_perimeters (x : ℝ) : 
  (∃ s₁ s₂ : ℝ, 
    s₁^2 = x^2 + 4*x + 4 ∧ 
    s₂^2 = 4*x^2 - 12*x + 9 ∧ 
    4*s₁ + 4*s₂ = 32) → 
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_square_areas_and_perimeters_l1487_148787


namespace NUMINAMATH_CALUDE_one_perm_scheduled_l1487_148759

/-- Represents the salon's pricing and scheduling for a day --/
structure SalonDay where
  haircut_price : ℕ
  perm_price : ℕ
  dye_job_price : ℕ
  dye_cost : ℕ
  num_haircuts : ℕ
  num_dye_jobs : ℕ
  tips : ℕ
  total_revenue : ℕ

/-- Calculates the number of perms scheduled given the salon day information --/
def calculate_perms (day : SalonDay) : ℕ :=
  let revenue_without_perms := day.haircut_price * day.num_haircuts +
                               (day.dye_job_price - day.dye_cost) * day.num_dye_jobs +
                               day.tips
  (day.total_revenue - revenue_without_perms) / day.perm_price

/-- Theorem stating that for the given salon day, exactly one perm is scheduled --/
theorem one_perm_scheduled (day : SalonDay) 
  (h1 : day.haircut_price = 30)
  (h2 : day.perm_price = 40)
  (h3 : day.dye_job_price = 60)
  (h4 : day.dye_cost = 10)
  (h5 : day.num_haircuts = 4)
  (h6 : day.num_dye_jobs = 2)
  (h7 : day.tips = 50)
  (h8 : day.total_revenue = 310) :
  calculate_perms day = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_perm_scheduled_l1487_148759


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l1487_148758

/-- If three lines x = 2, x - y - 1 = 0, and x + ky = 0 intersect at a single point, then k = -2 -/
theorem intersection_of_three_lines (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.1 = 2 ∧ p.1 - p.2 - 1 = 0 ∧ p.1 + k * p.2 = 0) → k = -2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l1487_148758


namespace NUMINAMATH_CALUDE_special_set_odd_sum_l1487_148736

def SpecialSet (S : Set (ℕ × ℕ)) : Prop :=
  (1, 0) ∈ S ∧
  ∀ (i j : ℕ), (i, j) ∈ S →
    (((i + 1, j) ∈ S ∧ (i, j + 1) ∉ S ∧ (i - 1, j - 1) ∉ S) ∨
     ((i + 1, j) ∉ S ∧ (i, j + 1) ∈ S ∧ (i - 1, j - 1) ∉ S) ∨
     ((i + 1, j) ∉ S ∧ (i, j + 1) ∉ S ∧ (i - 1, j - 1) ∈ S))

theorem special_set_odd_sum (S : Set (ℕ × ℕ)) (h : SpecialSet S) :
  ∀ (i j : ℕ), (i, j) ∈ S → Odd (i + j) := by
  sorry

end NUMINAMATH_CALUDE_special_set_odd_sum_l1487_148736


namespace NUMINAMATH_CALUDE_highlighter_box_cost_l1487_148709

theorem highlighter_box_cost (
  boxes : ℕ)
  (pens_per_box : ℕ)
  (rearranged_boxes : ℕ)
  (pens_per_package : ℕ)
  (package_price : ℚ)
  (pens_per_set : ℕ)
  (set_price : ℚ)
  (total_profit : ℚ)
  (h1 : boxes = 12)
  (h2 : pens_per_box = 30)
  (h3 : rearranged_boxes = 5)
  (h4 : pens_per_package = 6)
  (h5 : package_price = 3)
  (h6 : pens_per_set = 3)
  (h7 : set_price = 2)
  (h8 : total_profit = 115) :
  ∃ (cost_per_box : ℚ), abs (cost_per_box - 25/3) < 1/100 := by
  sorry

end NUMINAMATH_CALUDE_highlighter_box_cost_l1487_148709


namespace NUMINAMATH_CALUDE_frog_hop_probability_l1487_148728

-- Define the grid
def Grid := Fin 3 × Fin 3

-- Define corner squares
def is_corner (pos : Grid) : Prop :=
  (pos.1 = 0 ∧ pos.2 = 0) ∨ (pos.1 = 0 ∧ pos.2 = 2) ∨
  (pos.1 = 2 ∧ pos.2 = 0) ∨ (pos.1 = 2 ∧ pos.2 = 2)

-- Define center square
def center : Grid := (1, 1)

-- Define a single hop
def hop (pos : Grid) : Grid := sorry

-- Define the probability of reaching a corner in exactly n hops
def prob_corner_in (n : Nat) (start : Grid) : ℚ := sorry

-- Main theorem
theorem frog_hop_probability :
  prob_corner_in 2 center + prob_corner_in 3 center + prob_corner_in 4 center = 11/16 := by
  sorry

end NUMINAMATH_CALUDE_frog_hop_probability_l1487_148728


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_difference_l1487_148702

theorem consecutive_odd_numbers_difference (x : ℤ) : 
  (x + x + 2 + x + 4 + x + 6 + x + 8) / 5 = 55 → 
  (x + 8) - x = 8 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_difference_l1487_148702


namespace NUMINAMATH_CALUDE_major_axis_endpoints_of_ellipse_l1487_148792

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop := 6 * x^2 + y^2 = 6

/-- The endpoints of the major axis -/
def major_axis_endpoints : Set (ℝ × ℝ) := {(0, -Real.sqrt 6), (0, Real.sqrt 6)}

/-- Theorem: The endpoints of the major axis of the ellipse 6x^2 + y^2 = 6 
    are (0, -√6) and (0, √6) -/
theorem major_axis_endpoints_of_ellipse :
  ∀ (p : ℝ × ℝ), p ∈ major_axis_endpoints ↔ 
    (ellipse_equation p.1 p.2 ∧ 
     ∀ (q : ℝ × ℝ), ellipse_equation q.1 q.2 → p.1^2 + p.2^2 ≥ q.1^2 + q.2^2) :=
by sorry

end NUMINAMATH_CALUDE_major_axis_endpoints_of_ellipse_l1487_148792


namespace NUMINAMATH_CALUDE_least_positive_integer_exceeding_million_l1487_148797

theorem least_positive_integer_exceeding_million (n : ℕ) : 
  (∀ k < n, (8 : ℝ) ^ ((k * (k + 3)) / 22) ≤ 1000000) ∧
  (8 : ℝ) ^ ((n * (n + 3)) / 22) > 1000000 →
  n = 11 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_exceeding_million_l1487_148797


namespace NUMINAMATH_CALUDE_number_problem_l1487_148767

theorem number_problem (A B : ℝ) (h1 : A - B = 144) (h2 : A = 3 * B - 14) : A = 223 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1487_148767


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1487_148760

/-- A geometric sequence with specific properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 10 = -7 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1487_148760


namespace NUMINAMATH_CALUDE_quadratic_max_min_ratio_bound_l1487_148772

/-- Given a quadratic function f(x) = ax^2 + bx + c with positive coefficients and real roots,
    the maximum value of min{(b+c)/a, (c+a)/b, (a+b)/c} is 5/4 -/
theorem quadratic_max_min_ratio_bound 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hroots : b^2 ≥ 4*a*c) :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    min (min ((y+z)/x) ((z+x)/y)) ((x+y)/z) = 5/4 ∧
    ∀ (p q r : ℝ), p > 0 → q > 0 → r > 0 → 
      min (min ((q+r)/p) ((r+p)/q)) ((p+q)/r) ≤ 5/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_min_ratio_bound_l1487_148772


namespace NUMINAMATH_CALUDE_circumscribed_circle_properties_l1487_148799

/-- Triangle with vertices A(1,4), B(-2,3), and C(4,-5) -/
structure Triangle where
  A : ℝ × ℝ := (1, 4)
  B : ℝ × ℝ := (-2, 3)
  C : ℝ × ℝ := (4, -5)

/-- Circumscribed circle of a triangle -/
structure CircumscribedCircle (t : Triangle) where
  /-- Equation of the circle in the form ax^2 + ay^2 + bx + cy + d = 0 -/
  equation : ℝ → ℝ → ℝ
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem about the circumscribed circle of the specific triangle -/
theorem circumscribed_circle_properties (t : Triangle) :
  ∃ (c : CircumscribedCircle t),
    (∀ x y, c.equation x y = x^2 + y^2 - 2*x + 2*y - 23) ∧
    c.center = (1, -1) ∧
    c.radius = 5 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_circle_properties_l1487_148799


namespace NUMINAMATH_CALUDE_roots_sum_greater_than_2a_l1487_148700

noncomputable section

variables (a : ℝ) (x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := Real.log x + a / x

theorem roots_sum_greater_than_2a
  (h₁ : x₁ > 0)
  (h₂ : x₂ > 0)
  (h₃ : x₁ ≠ x₂)
  (h₄ : f a x₁ = a / 2)
  (h₅ : f a x₂ = a / 2) :
  x₁ + x₂ > 2 * a :=
sorry

end NUMINAMATH_CALUDE_roots_sum_greater_than_2a_l1487_148700


namespace NUMINAMATH_CALUDE_committee_safe_configuration_l1487_148710

/-- Represents a lock-key system for a committee safe --/
structure CommitteeSafe where
  numMembers : Nat
  numLocks : Nat
  keysPerMember : Nat

/-- Checks if a given number of members can open the safe --/
def canOpen (safe : CommitteeSafe) (presentMembers : Nat) : Prop :=
  presentMembers ≥ 3 ∧ presentMembers ≤ safe.numMembers

/-- Checks if the safe system is secure --/
def isSecure (safe : CommitteeSafe) : Prop :=
  ∀ (presentMembers : Nat), presentMembers ≤ safe.numMembers →
    (canOpen safe presentMembers ↔ 
      presentMembers * safe.keysPerMember ≥ safe.numLocks)

/-- The theorem stating the correct configuration for a 5-member committee --/
theorem committee_safe_configuration :
  ∃ (safe : CommitteeSafe),
    safe.numMembers = 5 ∧
    safe.numLocks = 10 ∧
    safe.keysPerMember = 6 ∧
    isSecure safe :=
sorry

end NUMINAMATH_CALUDE_committee_safe_configuration_l1487_148710


namespace NUMINAMATH_CALUDE_polynomial_inequality_l1487_148776

-- Define the polynomial P(x) = (x - x₁) ⋯ (x - xₙ)
def P (x : ℝ) (roots : List ℝ) : ℝ :=
  roots.foldl (fun acc r => acc * (x - r)) 1

-- State the theorem
theorem polynomial_inequality (roots : List ℝ) :
  ∀ x : ℝ, (deriv (P · roots) x)^2 ≥ (P x roots) * (deriv^[2] (P · roots) x) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_inequality_l1487_148776


namespace NUMINAMATH_CALUDE_two_digit_sum_with_reverse_is_square_l1487_148732

def reverse (n : Nat) : Nat :=
  10 * (n % 10) + (n / 10)

def is_two_digit (n : Nat) : Prop :=
  10 ≤ n ∧ n ≤ 99

def is_perfect_square (n : Nat) : Prop :=
  ∃ k : Nat, n = k * k

theorem two_digit_sum_with_reverse_is_square :
  {n : Nat | is_two_digit n ∧ is_perfect_square (n + reverse n)} =
  {29, 38, 47, 56, 65, 74, 83, 92} := by
sorry

end NUMINAMATH_CALUDE_two_digit_sum_with_reverse_is_square_l1487_148732


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1487_148711

/-- Two vectors in R² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Given vectors a = (-6, 3) and b = (2, x), if they are perpendicular, then x = -4 -/
theorem perpendicular_vectors_x_value :
  let a : ℝ × ℝ := (-6, 3)
  let b : ℝ → ℝ × ℝ := fun x ↦ (2, x)
  ∀ x : ℝ, perpendicular a (b x) → x = -4 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1487_148711


namespace NUMINAMATH_CALUDE_no_base6_digit_divisible_by_7_l1487_148750

theorem no_base6_digit_divisible_by_7 : 
  ¬ ∃ (d : ℕ), d < 6 ∧ (869 + 42 * d) % 7 = 0 := by
  sorry

#check no_base6_digit_divisible_by_7

end NUMINAMATH_CALUDE_no_base6_digit_divisible_by_7_l1487_148750


namespace NUMINAMATH_CALUDE_buffer_solution_composition_l1487_148781

/-- Represents the composition of a buffer solution -/
structure BufferSolution where
  chemicalA : Real
  water : Real
  chemicalB : Real
  totalVolume : Real

/-- Defines the specific buffer solution composition -/
def specificBuffer : BufferSolution where
  chemicalA := 0.05
  water := 0.025
  chemicalB := 0.02
  totalVolume := 0.075

/-- Theorem stating the required amounts of water and chemical B for 1.2 liters of buffer solution -/
theorem buffer_solution_composition 
  (desiredVolume : Real)
  (h1 : desiredVolume = 1.2) :
  let waterNeeded := desiredVolume * (specificBuffer.water / specificBuffer.totalVolume)
  let chemicalBNeeded := desiredVolume * (specificBuffer.chemicalB / specificBuffer.totalVolume)
  waterNeeded = 0.4 ∧ chemicalBNeeded = 0.032 := by
  sorry

end NUMINAMATH_CALUDE_buffer_solution_composition_l1487_148781


namespace NUMINAMATH_CALUDE_sector_central_angle_l1487_148733

/-- Given a sector with arc length 2m and radius 2cm, prove that its central angle is 100 radians. -/
theorem sector_central_angle (arc_length : ℝ) (radius : ℝ) (h1 : arc_length = 2) (h2 : radius = 0.02) :
  arc_length = radius * 100 := by
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1487_148733


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1487_148741

theorem complex_equation_solution (x y : ℝ) :
  (x : ℂ) - 3 * I = (8 * x - y : ℂ) * I → x = 0 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1487_148741


namespace NUMINAMATH_CALUDE_g_inverse_domain_l1487_148762

/-- The function g(x) = 2(x+1)^2 - 7 -/
def g (x : ℝ) : ℝ := 2 * (x + 1)^2 - 7

/-- d is the lower bound of the restricted domain [d,∞) -/
def d : ℝ := -1

/-- Theorem: -1 is the smallest value of d such that g has an inverse function on [d,∞) -/
theorem g_inverse_domain (x : ℝ) : 
  (∀ y z, x ≥ d → y ≥ d → z ≥ d → g y = g z → y = z) ∧ 
  (∀ e, e < d → ∃ y z, y > e ∧ z > e ∧ y ≠ z ∧ g y = g z) :=
sorry

end NUMINAMATH_CALUDE_g_inverse_domain_l1487_148762


namespace NUMINAMATH_CALUDE_sqrt_equation_l1487_148703

theorem sqrt_equation (m n : ℝ) : 
  1.55 * Real.sqrt (6 * m + 2 * Real.sqrt (9 * m^2 - n^2)) - 
  Real.sqrt (6 * m - 2 * Real.sqrt (9 * m^2 - n^2)) = 
  2 * Real.sqrt (3 * m - n) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_l1487_148703


namespace NUMINAMATH_CALUDE_loan_interest_rate_l1487_148755

theorem loan_interest_rate (principal time_period rate interest : ℝ) : 
  principal = 800 →
  time_period = rate →
  interest = 632 →
  interest = (principal * rate * time_period) / 100 →
  rate = Real.sqrt 79 :=
by sorry

end NUMINAMATH_CALUDE_loan_interest_rate_l1487_148755


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1487_148783

theorem fraction_to_decimal (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 175 / 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1487_148783


namespace NUMINAMATH_CALUDE_tank_fill_time_l1487_148754

-- Define the fill rates of pipes A and B
def fill_rate_A : ℚ := 1 / 60
def fill_rate_B : ℚ := 1 / 40

-- Define the total time to fill the tank
def total_time : ℚ := 30

-- Theorem stating that the tank will be filled in 30 minutes
theorem tank_fill_time :
  (total_time / 2) * fill_rate_B + (total_time / 2) * (fill_rate_A + fill_rate_B) = 1 :=
by sorry

end NUMINAMATH_CALUDE_tank_fill_time_l1487_148754


namespace NUMINAMATH_CALUDE_prize_distribution_l1487_148701

/-- The number of ways to distribute prizes in a class --/
theorem prize_distribution (n m : ℕ) (hn : n = 28) (hm : m = 4) :
  /- Identical prizes, at most one per student -/
  (n.choose m = 20475) ∧ 
  /- Identical prizes, more than one per student allowed -/
  ((n + m - 1).choose m = 31465) ∧ 
  /- Distinct prizes, at most one per student -/
  (n * (n - 1) * (n - 2) * (n - 3) = 491400) ∧ 
  /- Distinct prizes, more than one per student allowed -/
  (n ^ m = 614656) :=
by sorry

end NUMINAMATH_CALUDE_prize_distribution_l1487_148701


namespace NUMINAMATH_CALUDE_expression_evaluation_l1487_148765

theorem expression_evaluation :
  let x : ℝ := -1
  (x - 2)^2 - (2*x + 3)*(2*x - 3) - 4*x*(x - 1) = 6 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1487_148765


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1487_148737

theorem quadratic_solution_difference_squared : 
  ∀ α β : ℝ, 
  (α^2 = 2*α + 1) → 
  (β^2 = 2*β + 1) → 
  (α ≠ β) → 
  (α - β)^2 = 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l1487_148737


namespace NUMINAMATH_CALUDE_c_investment_determination_l1487_148774

/-- Represents the investment and profit distribution in a shop partnership --/
structure ShopPartnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit_share : ℕ

/-- Theorem stating that given the conditions of the problem, C's investment must be 30,000 --/
theorem c_investment_determination (shop : ShopPartnership)
  (h1 : shop.a_investment = 5000)
  (h2 : shop.b_investment = 15000)
  (h3 : shop.total_profit = 5000)
  (h4 : shop.c_profit_share = 3000)
  (h5 : shop.c_profit_share * (shop.a_investment + shop.b_investment + shop.c_investment) = 
        shop.total_profit * shop.c_investment) :
  shop.c_investment = 30000 := by
  sorry

#check c_investment_determination

end NUMINAMATH_CALUDE_c_investment_determination_l1487_148774


namespace NUMINAMATH_CALUDE_calculate_expression_no_solution_inequality_system_l1487_148782

-- Problem 1
theorem calculate_expression : (-2)^3 + |(-4)| - Real.sqrt 9 = -7 := by sorry

-- Problem 2
theorem no_solution_inequality_system :
  ¬∃ x : ℝ, (2*x > 3*x - 2) ∧ (x - 1 > (x + 2) / 3) := by sorry

end NUMINAMATH_CALUDE_calculate_expression_no_solution_inequality_system_l1487_148782


namespace NUMINAMATH_CALUDE_smallest_base_for_100_in_three_digits_l1487_148738

theorem smallest_base_for_100_in_three_digits :
  ∃ (b : ℕ), b = 5 ∧ b^2 ≤ 100 ∧ 100 < b^3 ∧ ∀ (x : ℕ), x < b → (x^2 ≤ 100 → 100 ≥ x^3) :=
sorry

end NUMINAMATH_CALUDE_smallest_base_for_100_in_three_digits_l1487_148738


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1487_148756

theorem sqrt_product_equality (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) := by sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1487_148756


namespace NUMINAMATH_CALUDE_circle_passes_through_origin_l1487_148798

/-- A circle in the 2D plane -/
structure Circle where
  a : ℝ  -- x-coordinate of the center
  b : ℝ  -- y-coordinate of the center
  r : ℝ  -- radius

/-- Predicate to check if a point (x, y) is on the circle -/
def onCircle (c : Circle) (x y : ℝ) : Prop :=
  (x - c.a)^2 + (y - c.b)^2 = c.r^2

/-- Theorem: A circle passes through the origin iff a^2 + b^2 = r^2 -/
theorem circle_passes_through_origin (c : Circle) :
  onCircle c 0 0 ↔ c.a^2 + c.b^2 = c.r^2 := by sorry

end NUMINAMATH_CALUDE_circle_passes_through_origin_l1487_148798


namespace NUMINAMATH_CALUDE_evening_customers_is_40_l1487_148735

/-- Represents the revenue and customer data for a movie theater on a Friday night. -/
structure TheaterData where
  matineePrice : ℕ
  eveningPrice : ℕ
  openingNightPrice : ℕ
  popcornPrice : ℕ
  matineeCustomers : ℕ
  openingNightCustomers : ℕ
  totalRevenue : ℕ

/-- Calculates the number of evening customers based on the theater data. -/
def eveningCustomers (data : TheaterData) : ℕ :=
  let totalCustomers := data.matineeCustomers + data.openingNightCustomers + (data.totalRevenue - 
    (data.matineePrice * data.matineeCustomers + 
     data.openingNightPrice * data.openingNightCustomers + 
     data.popcornPrice * (data.matineeCustomers + data.openingNightCustomers) / 2)) / data.eveningPrice
  (totalCustomers - data.matineeCustomers - data.openingNightCustomers)

/-- Theorem stating that the number of evening customers is 40 given the specific theater data. -/
theorem evening_customers_is_40 (data : TheaterData) 
  (h1 : data.matineePrice = 5)
  (h2 : data.eveningPrice = 7)
  (h3 : data.openingNightPrice = 10)
  (h4 : data.popcornPrice = 10)
  (h5 : data.matineeCustomers = 32)
  (h6 : data.openingNightCustomers = 58)
  (h7 : data.totalRevenue = 1670) :
  eveningCustomers data = 40 := by
  sorry

end NUMINAMATH_CALUDE_evening_customers_is_40_l1487_148735


namespace NUMINAMATH_CALUDE_square_of_sqrt_plus_two_l1487_148769

theorem square_of_sqrt_plus_two (n : ℕ) (h : ∃ k : ℤ, k^2 = 1 + 12*n^2) :
  ∃ m : ℕ, (2 + 2*(Int.sqrt (1 + 12*n^2)))^2 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sqrt_plus_two_l1487_148769


namespace NUMINAMATH_CALUDE_car_trip_average_speed_l1487_148763

/-- Given a car's trip with two segments:
    1. 40 miles on local roads at 20 mph
    2. 180 miles on highway at 60 mph
    The average speed of the entire trip is 44 mph -/
theorem car_trip_average_speed :
  let local_distance : ℝ := 40
  let local_speed : ℝ := 20
  let highway_distance : ℝ := 180
  let highway_speed : ℝ := 60
  let total_distance : ℝ := local_distance + highway_distance
  let total_time : ℝ := local_distance / local_speed + highway_distance / highway_speed
  total_distance / total_time = 44 := by sorry

end NUMINAMATH_CALUDE_car_trip_average_speed_l1487_148763


namespace NUMINAMATH_CALUDE_geometric_increasing_iff_second_greater_first_l1487_148734

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- An increasing sequence -/
def IncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- Theorem: For a geometric sequence with positive first term,
    the second term being greater than the first is equivalent to
    the sequence being increasing -/
theorem geometric_increasing_iff_second_greater_first
    (a : ℕ → ℝ) (h_geom : GeometricSequence a) (h_pos : a 1 > 0) :
    a 1 < a 2 ↔ IncreasingSequence a :=
  sorry

end NUMINAMATH_CALUDE_geometric_increasing_iff_second_greater_first_l1487_148734


namespace NUMINAMATH_CALUDE_laptop_sticker_price_l1487_148726

theorem laptop_sticker_price :
  ∀ (x : ℝ),
  (0.8 * x - 100 = 0.7 * x - 20) →
  x = 800 := by
sorry

end NUMINAMATH_CALUDE_laptop_sticker_price_l1487_148726


namespace NUMINAMATH_CALUDE_decimal_multiplication_l1487_148761

theorem decimal_multiplication (a b c : ℕ) (h : a * b = c) :
  (a : ℚ) / 100 * ((b : ℚ) / 100) = (c : ℚ) / 10000 :=
by
  sorry

-- Example usage
example : (268 : ℚ) / 100 * ((74 : ℚ) / 100) = (19832 : ℚ) / 10000 :=
decimal_multiplication 268 74 19832 rfl

end NUMINAMATH_CALUDE_decimal_multiplication_l1487_148761


namespace NUMINAMATH_CALUDE_sphere_radius_increase_l1487_148744

/-- Theorem: If the surface area of a sphere increases by 21.00000000000002%,
    then the radius of the sphere increases by approximately 10%. -/
theorem sphere_radius_increase (r : ℝ) (h : r > 0) :
  let new_surface_area := 4 * Real.pi * r^2 * 1.2100000000000002
  let new_radius := r * (1 + 10/100)
  abs (new_surface_area - 4 * Real.pi * new_radius^2) < 1e-6 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_increase_l1487_148744


namespace NUMINAMATH_CALUDE_arc_label_sum_bounds_l1487_148778

/-- Represents the color of a point on the circle -/
inductive Color
  | Red
  | Blue
  | Green

/-- Calculates the label for an arc based on endpoint colors -/
def arcLabel (c1 c2 : Color) : Nat :=
  match c1, c2 with
  | Color.Red, Color.Blue | Color.Blue, Color.Red => 1
  | Color.Red, Color.Green | Color.Green, Color.Red => 2
  | Color.Blue, Color.Green | Color.Green, Color.Blue => 3
  | _, _ => 0

/-- Represents the configuration of points on the circle -/
structure CircleConfig where
  points : List Color
  red_count : Nat
  blue_count : Nat
  green_count : Nat

/-- Calculates the sum of arc labels for a given configuration -/
def sumArcLabels (config : CircleConfig) : Nat :=
  let arcs := List.zip config.points (List.rotateLeft config.points 1)
  List.sum (List.map (fun (c1, c2) => arcLabel c1 c2) arcs)

/-- The main theorem statement -/
theorem arc_label_sum_bounds 
  (config : CircleConfig)
  (h_red : config.red_count = 40)
  (h_blue : config.blue_count = 30)
  (h_green : config.green_count = 20)
  (h_total : config.points.length = 90) :
  6 ≤ sumArcLabels config ∧ sumArcLabels config ≤ 140 := by
  sorry

end NUMINAMATH_CALUDE_arc_label_sum_bounds_l1487_148778


namespace NUMINAMATH_CALUDE_sod_square_size_l1487_148749

/-- Given a total area and number of squares, prove the side length of each square -/
theorem sod_square_size (total_area : ℝ) (num_squares : ℕ) 
  (h1 : total_area = 6000) 
  (h2 : num_squares = 1500) : 
  Real.sqrt (total_area / num_squares) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sod_square_size_l1487_148749


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1487_148729

theorem complex_modulus_problem (z : ℂ) (i : ℂ) (h : i^2 = -1) (hz : z = (3 - i) / (1 + i)) :
  Complex.abs (z + i) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1487_148729


namespace NUMINAMATH_CALUDE_projection_of_v_onto_Q_l1487_148730

/-- The plane Q is defined by the equation x + 2y - z = 0 -/
def Q : Set (Fin 3 → ℝ) :=
  {v | v 0 + 2 * v 1 - v 2 = 0}

/-- The vector v -/
def v : Fin 3 → ℝ := ![2, 3, 1]

/-- The projection of v onto Q -/
def projection (v : Fin 3 → ℝ) (Q : Set (Fin 3 → ℝ)) : Fin 3 → ℝ :=
  sorry

theorem projection_of_v_onto_Q :
  projection v Q = ![5/6, 4/6, 13/6] := by sorry

end NUMINAMATH_CALUDE_projection_of_v_onto_Q_l1487_148730


namespace NUMINAMATH_CALUDE_evaluate_P_at_negative_two_l1487_148790

def P (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x + 4

theorem evaluate_P_at_negative_two : P (-2) = -18 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_P_at_negative_two_l1487_148790


namespace NUMINAMATH_CALUDE_age_difference_l1487_148745

/-- Given four individuals with ages a, b, c, and d, prove that c is 10 years younger than a. -/
theorem age_difference (a b c d : ℝ) 
  (sum_ab_bc : a + b = b + c + 10)
  (sum_cd_ad : c + d = a + d - 15)
  (ratio_ad : a / d = 7 / 4) :
  a - c = 10 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1487_148745


namespace NUMINAMATH_CALUDE_stratified_sampling_specific_case_l1487_148773

/-- The number of ways to select students using stratified sampling -/
def stratified_sampling_ways (n_female : ℕ) (n_male : ℕ) (k_female : ℕ) (k_male : ℕ) : ℕ :=
  Nat.choose n_female k_female * Nat.choose n_male k_male

/-- Theorem stating the number of ways to select 5 students from 6 female and 4 male students -/
theorem stratified_sampling_specific_case :
  stratified_sampling_ways 6 4 3 2 = Nat.choose 6 3 * Nat.choose 4 2 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_specific_case_l1487_148773


namespace NUMINAMATH_CALUDE_train_passing_platform_l1487_148725

/-- A train passes a platform -/
theorem train_passing_platform
  (l : ℝ) -- length of the train
  (t : ℝ) -- time to pass a pole
  (v : ℝ) -- velocity of the train
  (h1 : t > 0) -- time is positive
  (h2 : l > 0) -- length is positive
  (h3 : v > 0) -- velocity is positive
  (h4 : l = v * t) -- relation between length, velocity, and time for passing a pole
  : v * (4 * t) = l + 3 * l := by sorry

end NUMINAMATH_CALUDE_train_passing_platform_l1487_148725


namespace NUMINAMATH_CALUDE_jan_extra_miles_l1487_148793

/-- Represents the driving data for a person -/
structure DrivingData where
  time : ℝ
  speed : ℝ
  distance : ℝ

/-- The problem statement -/
theorem jan_extra_miles (ian : DrivingData) (han : DrivingData) (jan : DrivingData) : 
  han.time = ian.time + 2 →
  han.speed = ian.speed + 5 →
  jan.time = ian.time + 3 →
  jan.speed = ian.speed + 15 →
  han.distance = ian.distance + 110 →
  jan.distance = ian.distance + 195 := by
  sorry


end NUMINAMATH_CALUDE_jan_extra_miles_l1487_148793


namespace NUMINAMATH_CALUDE_intersection_A_B_l1487_148784

def A : Set ℕ := {0, 1, 2, 3, 4}
def B : Set ℕ := {x | ∃ n ∈ A, x = 2 * n}

theorem intersection_A_B : A ∩ B = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1487_148784


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l1487_148757

theorem rectangle_area_diagonal (l w d : ℝ) (h1 : l / w = 5 / 2) (h2 : l ^ 2 + w ^ 2 = d ^ 2) :
  l * w = (10 / 29) * d ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l1487_148757


namespace NUMINAMATH_CALUDE_max_min_sum_of_f_l1487_148742

def f (g : ℝ → ℝ) (x : ℝ) : ℝ := g x + 2

theorem max_min_sum_of_f (g : ℝ → ℝ) (h : ∀ x, g (-x) = -g x) :
  let f := f g
  let M := ⨆ (x : ℝ) (hx : x ∈ Set.Icc (-3) 3), f x
  let N := ⨅ (x : ℝ) (hx : x ∈ Set.Icc (-3) 3), f x
  M + N = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_min_sum_of_f_l1487_148742


namespace NUMINAMATH_CALUDE_road_construction_equation_l1487_148704

theorem road_construction_equation (x : ℝ) (h : x > 0) :
  let road_length : ℝ := 1200
  let speed_increase : ℝ := 0.2
  let days_saved : ℝ := 2
  (road_length / x) - (road_length / ((1 + speed_increase) * x)) = days_saved :=
by sorry

end NUMINAMATH_CALUDE_road_construction_equation_l1487_148704


namespace NUMINAMATH_CALUDE_point_A_transformation_l1487_148796

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point by dx units horizontally and dy units vertically -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  ⟨p.x + dx, p.y + dy⟩

/-- Rotates a point 180 degrees around a center point -/
def rotate180 (p : Point) (center : Point) : Point :=
  ⟨2 * center.x - p.x, 2 * center.y - p.y⟩

/-- The main theorem stating that the given transformations on point A result in (1, -1) -/
theorem point_A_transformation :
  let A : Point := ⟨3, -2⟩
  let translatedA := translate A 4 3
  let rotationCenter : Point := ⟨4, 0⟩
  let finalA := rotate180 translatedA rotationCenter
  finalA = ⟨1, -1⟩ := by sorry

end NUMINAMATH_CALUDE_point_A_transformation_l1487_148796


namespace NUMINAMATH_CALUDE_platform_length_calculation_l1487_148753

-- Define the given parameters
def train_length : ℝ := 1500
def time_tree : ℝ := 120
def time_platform : ℝ := 160

-- Define the platform length as a variable
def platform_length : ℝ := sorry

-- Theorem statement
theorem platform_length_calculation :
  (train_length / time_tree) * time_platform = train_length + platform_length ∧
  platform_length = 500 := by sorry

end NUMINAMATH_CALUDE_platform_length_calculation_l1487_148753


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1487_148766

/-- Arithmetic sequence sum -/
def arithmetic_sum (a1 : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * a1 + (n * (n - 1) : ℚ) / 2 * d

/-- Arithmetic sequence term -/
def arithmetic_term (a1 : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a1 + (n - 1 : ℚ) * d

theorem arithmetic_sequence_common_difference :
  ∃ (a1 : ℚ), 
    arithmetic_sum a1 (1/5) 5 = 6 ∧ 
    arithmetic_term a1 (1/5) 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1487_148766


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1487_148708

theorem complex_equation_solution (x : ℝ) : 
  (↑x + 2 * Complex.I) * (↑x - Complex.I) = (6 : ℂ) + 2 * Complex.I → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1487_148708


namespace NUMINAMATH_CALUDE_inverse_sum_equals_one_l1487_148746

theorem inverse_sum_equals_one (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) :
  1 / x + 1 / y = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_one_l1487_148746


namespace NUMINAMATH_CALUDE_goldfish_cost_graph_piecewise_linear_l1487_148786

/-- The cost function for goldfish purchases -/
def cost (n : ℕ) : ℚ :=
  if n ≤ 10 then 20 * n else 20 * n - 5

/-- The graph of the cost function is piecewise linear -/
theorem goldfish_cost_graph_piecewise_linear :
  ∃ (f g : ℚ → ℚ),
    (∀ x, f x = 20 * x) ∧
    (∀ x, g x = 20 * x - 5) ∧
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 15 →
      (n ≤ 10 ∧ cost n = f n) ∨
      (10 < n ∧ cost n = g n)) :=
by sorry

end NUMINAMATH_CALUDE_goldfish_cost_graph_piecewise_linear_l1487_148786


namespace NUMINAMATH_CALUDE_max_m_value_l1487_148777

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 9

-- Define the points A and B
def point_A (m : ℝ) : ℝ × ℝ := (-m, 0)
def point_B (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define the condition for point P
def point_P_condition (P : ℝ × ℝ) (m : ℝ) : Prop :=
  circle_C P.1 P.2 ∧
  let AP := (P.1 + m, P.2)
  let BP := (P.1 - m, P.2)
  AP.1 * BP.1 + AP.2 * BP.2 = 0

theorem max_m_value :
  ∀ m : ℝ, m > 0 →
  (∃ P : ℝ × ℝ, point_P_condition P m) →
  m ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l1487_148777


namespace NUMINAMATH_CALUDE_family_structure_l1487_148770

/-- Represents a family with siblings -/
structure Family :=
  (sisters : ℕ)
  (brothers : ℕ)

/-- Calculates the number of sisters a girl has in the family, excluding herself -/
def sisters_of_girl (f : Family) : ℕ := f.sisters - 1

/-- Calculates the number of brothers a girl has in the family -/
def brothers_of_girl (f : Family) : ℕ := f.brothers

/-- Calculates the ratio of sisters to total siblings for a girl in the family -/
def sister_ratio (f : Family) : ℚ :=
  (sisters_of_girl f : ℚ) / (f.sisters + f.brothers - 1 : ℚ)

/-- Theorem about the family structure and sibling relationships -/
theorem family_structure (f : Family) 
  (h1 : f.sisters = 5)
  (h2 : f.brothers = 5) :
  sisters_of_girl f = 4 ∧
  brothers_of_girl f = 4 ∧
  sisters_of_girl f + brothers_of_girl f = 8 ∧
  sister_ratio f = 1/2 := by
  sorry

#check family_structure

end NUMINAMATH_CALUDE_family_structure_l1487_148770


namespace NUMINAMATH_CALUDE_cos_45_minus_cos_90_l1487_148771

theorem cos_45_minus_cos_90 : Real.cos (π/4) - Real.cos (π/2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_45_minus_cos_90_l1487_148771


namespace NUMINAMATH_CALUDE_compare_powers_l1487_148727

theorem compare_powers : 9^61 < 27^41 ∧ 27^41 < 81^31 := by
  sorry

end NUMINAMATH_CALUDE_compare_powers_l1487_148727


namespace NUMINAMATH_CALUDE_same_color_probability_l1487_148724

/-- Represents a 20-sided die with a specific color distribution -/
structure Die :=
  (maroon : Nat)
  (teal : Nat)
  (cyan : Nat)
  (sparkly : Nat)
  (total : Nat)
  (valid : maroon + teal + cyan + sparkly = total)

/-- The first die with its color distribution -/
def die1 : Die :=
  { maroon := 5
    teal := 6
    cyan := 7
    sparkly := 2
    total := 20
    valid := by simp }

/-- The second die with its color distribution -/
def die2 : Die :=
  { maroon := 4
    teal := 7
    cyan := 8
    sparkly := 1
    total := 20
    valid := by simp }

/-- Calculates the probability of a specific color on a die -/
def colorProbability (d : Die) (color : Nat) : Rat :=
  color / d.total

/-- Calculates the probability of both dice showing the same color -/
def sameProbability (d1 d2 : Die) : Rat :=
  (colorProbability d1 d1.maroon * colorProbability d2 d2.maroon) +
  (colorProbability d1 d1.teal * colorProbability d2 d2.teal) +
  (colorProbability d1 d1.cyan * colorProbability d2 d2.cyan) +
  (colorProbability d1 d1.sparkly * colorProbability d2 d2.sparkly)

theorem same_color_probability :
  sameProbability die1 die2 = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1487_148724


namespace NUMINAMATH_CALUDE_tape_length_l1487_148722

/-- Given 15 pieces of tape, each 20 cm long, overlapping by 5 cm, 
    the total length is 230 cm -/
theorem tape_length (n : ℕ) (piece_length overlap : ℝ) 
  (h1 : n = 15)
  (h2 : piece_length = 20)
  (h3 : overlap = 5) :
  piece_length + (n - 1) * (piece_length - overlap) = 230 :=
by
  sorry

end NUMINAMATH_CALUDE_tape_length_l1487_148722


namespace NUMINAMATH_CALUDE_chairperson_and_committee_count_l1487_148714

/-- The number of ways to choose a chairperson and a 3-person committee from a group of 10 people,
    where the chairperson is not a member of the committee. -/
def chairperson_and_committee (total_people : ℕ) (committee_size : ℕ) : ℕ :=
  total_people * (Nat.choose (total_people - 1) committee_size)

/-- Theorem stating that the number of ways to choose a chairperson and a 3-person committee
    from a group of 10 people, where the chairperson is not a member of the committee, is 840. -/
theorem chairperson_and_committee_count :
  chairperson_and_committee 10 3 = 840 := by
  sorry

end NUMINAMATH_CALUDE_chairperson_and_committee_count_l1487_148714


namespace NUMINAMATH_CALUDE_inequality_solution_l1487_148789

theorem inequality_solution (x : ℝ) :
  (x - 4) / (x^2 + 3*x + 10) ≥ 0 ↔ x ≥ 4 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1487_148789


namespace NUMINAMATH_CALUDE_becky_anna_size_ratio_l1487_148706

/-- Theorem: Given the sizes of Anna, Becky, and Ginger, prove the ratio of Becky's to Anna's size --/
theorem becky_anna_size_ratio :
  ∀ (anna_size becky_size ginger_size : ℕ),
  anna_size = 2 →
  ∃ k : ℕ, becky_size = k * anna_size →
  ginger_size = 2 * becky_size - 4 →
  ginger_size = 8 →
  becky_size / anna_size = 3 := by
sorry

end NUMINAMATH_CALUDE_becky_anna_size_ratio_l1487_148706


namespace NUMINAMATH_CALUDE_sandys_puppies_l1487_148768

/-- Given that Sandy initially had 8 puppies and gave away 4 puppies,
    prove that she now has 4 puppies. -/
theorem sandys_puppies (initial_puppies : ℕ) (given_away : ℕ) 
  (h1 : initial_puppies = 8) (h2 : given_away = 4) : 
  initial_puppies - given_away = 4 := by
  sorry

end NUMINAMATH_CALUDE_sandys_puppies_l1487_148768


namespace NUMINAMATH_CALUDE_product_of_a_values_l1487_148743

theorem product_of_a_values (a : ℂ) (α β γ : ℂ) : 
  (∀ x : ℂ, x^3 - x^2 + a*x - 1 = 0 ↔ (x = α ∨ x = β ∨ x = γ)) →
  (α^3 + 1) * (β^3 + 1) * (γ^3 + 1) = 2018 →
  ∃ (a₁ a₂ a₃ : ℂ), (∀ x : ℂ, x^3 - 6*x + 2009 = 0 ↔ (x = a₁ ∨ x = a₂ ∨ x = a₃)) ∧ a₁ * a₂ * a₃ = 2009 :=
by sorry

end NUMINAMATH_CALUDE_product_of_a_values_l1487_148743


namespace NUMINAMATH_CALUDE_not_possible_when_70_possible_when_80_l1487_148719

-- Define the original triangle
structure OriginalTriangle where
  alpha : Real
  is_valid : 0 < alpha ∧ alpha < 180

-- Define the resulting triangles after cutting
structure ResultingTriangle where
  angles : Fin 3 → Real
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  all_positive : ∀ i, 0 < angles i

-- Define the cutting process
def cut (t : OriginalTriangle) : Set ResultingTriangle := sorry

-- Theorem for the case when α = 70°
theorem not_possible_when_70 (t : OriginalTriangle) 
  (h : t.alpha = 70) : 
  ¬∃ (s : Set ResultingTriangle), s = cut t ∧ 
  (∀ rt ∈ s, ∀ i, rt.angles i < t.alpha) :=
sorry

-- Theorem for the case when α = 80°
theorem possible_when_80 (t : OriginalTriangle) 
  (h : t.alpha = 80) : 
  ∃ (s : Set ResultingTriangle), s = cut t ∧ 
  (∀ rt ∈ s, ∀ i, rt.angles i < t.alpha) :=
sorry

end NUMINAMATH_CALUDE_not_possible_when_70_possible_when_80_l1487_148719


namespace NUMINAMATH_CALUDE_lottery_problem_l1487_148788

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Yellow

/-- Represents the contents of the bag -/
structure Bag :=
  (red : ℕ)
  (yellow : ℕ)

/-- Calculates the probability of drawing a red ball -/
def prob_red (b : Bag) : ℚ :=
  b.red / (b.red + b.yellow)

/-- Calculates the probability of drawing two balls of the same color -/
def prob_same_color (b : Bag) : ℚ :=
  let total := b.red + b.yellow
  (b.red * (b.red - 1) + b.yellow * (b.yellow - 1)) / (total * (total - 1))

theorem lottery_problem :
  let initial_bag : Bag := ⟨1, 3⟩
  let red_added_bag : Bag := ⟨2, 3⟩
  let yellow_added_bag : Bag := ⟨1, 4⟩
  (prob_red initial_bag = 1/4) ∧
  (prob_same_color yellow_added_bag > prob_same_color red_added_bag) := by
  sorry


end NUMINAMATH_CALUDE_lottery_problem_l1487_148788


namespace NUMINAMATH_CALUDE_gp_ratio_proof_l1487_148752

/-- Given a geometric progression where the ratio of the sum of the first 6 terms
    to the sum of the first 3 terms is 28, prove that the common ratio is 3. -/
theorem gp_ratio_proof (a r : ℝ) (hr : r ≠ 1) :
  (a * (1 - r^6) / (1 - r)) / (a * (1 - r^3) / (1 - r)) = 28 →
  r = 3 :=
by sorry

end NUMINAMATH_CALUDE_gp_ratio_proof_l1487_148752


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1487_148785

/-- The equation of a hyperbola C with focal length 2√5, whose asymptotes are tangent to the parabola y = 1/16x² + 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a^2 + b^2 = 20) -- Focal length condition
  (h4 : ∃ x : ℝ, (1/16 * x^2 + 1 = (a/b) * x)) -- Tangency condition
  : a^2 = 4 ∧ b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1487_148785


namespace NUMINAMATH_CALUDE_comparison_sqrt_l1487_148723

theorem comparison_sqrt : 2 * Real.sqrt 3 < Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_comparison_sqrt_l1487_148723


namespace NUMINAMATH_CALUDE_intersection_lines_sum_l1487_148747

theorem intersection_lines_sum (c d : ℝ) : 
  (3 = (1/3) * 3 + c) → 
  (3 = (1/3) * 3 + d) → 
  c + d = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_lines_sum_l1487_148747


namespace NUMINAMATH_CALUDE_part_one_part_two_l1487_148705

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Part I
theorem part_one (m : ℝ) (h_m : m > 0) :
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 2 ↔ f (x + 1/2) ≤ 2*m + 1) → m = 3/2 := by
  sorry

-- Part II
theorem part_two :
  (∃ a : ℝ, ∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) ∧
  (∀ a : ℝ, (∀ x y : ℝ, f x ≤ 2^y + a/(2^y) + |2*x + 3|) → a ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1487_148705
