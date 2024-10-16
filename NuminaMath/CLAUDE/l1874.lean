import Mathlib

namespace NUMINAMATH_CALUDE_lottery_probability_l1874_187495

/-- Represents the lottery setup -/
structure LotterySetup where
  total_people : Nat
  total_tickets : Nat
  winning_tickets : Nat

/-- Calculates the probability of the lottery ending after a specific draw -/
def probability_end_after_draw (setup : LotterySetup) (draw : Nat) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem lottery_probability (setup : LotterySetup) :
  setup.total_people = 5 →
  setup.total_tickets = 5 →
  setup.winning_tickets = 3 →
  probability_end_after_draw setup 4 = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_l1874_187495


namespace NUMINAMATH_CALUDE_imaginary_unit_equation_l1874_187480

theorem imaginary_unit_equation : Complex.I ^ 3 - 2 / Complex.I = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_equation_l1874_187480


namespace NUMINAMATH_CALUDE_company_picnic_attendance_l1874_187402

/-- Represents the percentage of men who attended the company picnic -/
def percentage_men_attended : ℝ := 0.2

theorem company_picnic_attendance 
  (percent_women_attended : ℝ) 
  (percent_men_total : ℝ) 
  (percent_total_attended : ℝ) 
  (h1 : percent_women_attended = 0.4)
  (h2 : percent_men_total = 0.45)
  (h3 : percent_total_attended = 0.31000000000000007) :
  percentage_men_attended = 
    (percent_total_attended - (1 - percent_men_total) * percent_women_attended) / percent_men_total :=
by sorry

end NUMINAMATH_CALUDE_company_picnic_attendance_l1874_187402


namespace NUMINAMATH_CALUDE_jake_flower_charge_l1874_187441

/-- The amount Jake should charge for planting flowers -/
def flower_charge (mowing_rate : ℚ) (desired_rate : ℚ) (mowing_time : ℚ) (planting_time : ℚ) : ℚ :=
  planting_time * desired_rate + (desired_rate - mowing_rate) * mowing_time

/-- Theorem: Jake should charge $45 for planting flowers -/
theorem jake_flower_charge :
  flower_charge 15 20 1 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_jake_flower_charge_l1874_187441


namespace NUMINAMATH_CALUDE_mango_count_proof_l1874_187465

/-- Calculates the total number of mangoes in multiple boxes -/
def total_mangoes (mangoes_per_dozen : ℕ) (dozens_per_box : ℕ) (num_boxes : ℕ) : ℕ :=
  mangoes_per_dozen * dozens_per_box * num_boxes

/-- Proves that 36 boxes of 10 dozen mangoes each contain 4,320 mangoes in total -/
theorem mango_count_proof : total_mangoes 12 10 36 = 4320 := by
  sorry

#eval total_mangoes 12 10 36

end NUMINAMATH_CALUDE_mango_count_proof_l1874_187465


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l1874_187431

theorem imaginary_part_of_one_minus_i_squared : Complex.im ((1 - Complex.I) ^ 2) = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l1874_187431


namespace NUMINAMATH_CALUDE_sin_fourth_power_decomposition_l1874_187473

theorem sin_fourth_power_decomposition :
  ∃ (b₁ b₂ b₃ b₄ : ℝ),
    (∀ θ : ℝ, Real.sin θ ^ 4 = b₁ * Real.sin θ + b₂ * Real.sin (2 * θ) + b₃ * Real.sin (3 * θ) + b₄ * Real.sin (4 * θ)) →
    b₁^2 + b₂^2 + b₃^2 + b₄^2 = 17 / 64 :=
by sorry

end NUMINAMATH_CALUDE_sin_fourth_power_decomposition_l1874_187473


namespace NUMINAMATH_CALUDE_students_not_enrolled_l1874_187416

theorem students_not_enrolled (total : ℕ) (math : ℕ) (chem : ℕ) (both : ℕ) :
  total = 60 ∧ math = 40 ∧ chem = 30 ∧ both = 25 →
  total - (math + chem - both) = 15 :=
by sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l1874_187416


namespace NUMINAMATH_CALUDE_transformed_quadratic_has_root_l1874_187448

/-- Given a quadratic polynomial with two roots, adding one root to the linear coefficient
    and subtracting its square from the constant term results in a polynomial with at least one root -/
theorem transformed_quadratic_has_root (a b r : ℝ) : 
  (∃ x y : ℝ, x^2 + a*x + b = 0 ∧ y^2 + a*y + b = 0 ∧ x ≠ y) →
  (∃ z : ℝ, z^2 + (a + r)*z + (b - r^2) = 0) ∧ 
  (r^2 + a*r + b = 0) :=
sorry

end NUMINAMATH_CALUDE_transformed_quadratic_has_root_l1874_187448


namespace NUMINAMATH_CALUDE_trigonometric_problem_l1874_187442

theorem trigonometric_problem (α : Real) 
  (h1 : α > π / 2 ∧ α < π) 
  (h2 : Real.sin (α / 2) + Real.cos (α / 2) = 3 * Real.sqrt 5 / 5) : 
  Real.sin α = 4 / 5 ∧ 
  Real.cos (2 * α + π / 3) = (24 * Real.sqrt 3 - 7) / 50 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l1874_187442


namespace NUMINAMATH_CALUDE_parabola_vertex_l1874_187478

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 5

/-- The vertex coordinates of the parabola -/
def vertex : ℝ × ℝ := (3, -4)

/-- Theorem: The vertex coordinates of the parabola y = x^2 - 6x + 5 are (3, -4) -/
theorem parabola_vertex : 
  ∀ x : ℝ, parabola x ≥ parabola (vertex.1) ∧ parabola (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1874_187478


namespace NUMINAMATH_CALUDE_fourth_term_value_l1874_187410

def S (n : ℕ) : ℤ := n^2 - 3*n

def a (n : ℕ) : ℤ := S n - S (n-1)

theorem fourth_term_value : a 4 = 4 := by sorry

end NUMINAMATH_CALUDE_fourth_term_value_l1874_187410


namespace NUMINAMATH_CALUDE_marks_spending_l1874_187433

-- Define constants for item quantities
def notebooks : ℕ := 4
def pens : ℕ := 3
def books : ℕ := 1
def magazines : ℕ := 2

-- Define prices
def notebook_price : ℚ := 2
def pen_price : ℚ := 1.5
def book_price : ℚ := 12
def magazine_original_price : ℚ := 3

-- Define discount and coupon
def magazine_discount : ℚ := 0.25
def coupon_value : ℚ := 3
def coupon_threshold : ℚ := 20

-- Calculate discounted magazine price
def discounted_magazine_price : ℚ := magazine_original_price * (1 - magazine_discount)

-- Calculate total cost before coupon
def total_before_coupon : ℚ :=
  notebooks * notebook_price +
  pens * pen_price +
  books * book_price +
  magazines * discounted_magazine_price

-- Apply coupon if total is over the threshold
def final_total : ℚ :=
  if total_before_coupon ≥ coupon_threshold
  then total_before_coupon - coupon_value
  else total_before_coupon

-- Theorem to prove
theorem marks_spending :
  final_total = 26 := by sorry

end NUMINAMATH_CALUDE_marks_spending_l1874_187433


namespace NUMINAMATH_CALUDE_floor_sqrt_27_squared_l1874_187482

theorem floor_sqrt_27_squared : ⌊Real.sqrt 27⌋^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_27_squared_l1874_187482


namespace NUMINAMATH_CALUDE_xiao_ma_calculation_l1874_187447

theorem xiao_ma_calculation (x : ℤ) : 41 - x = 12 → 41 + x = 70 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ma_calculation_l1874_187447


namespace NUMINAMATH_CALUDE_max_pairs_sum_l1874_187488

theorem max_pairs_sum (n : ℕ) (h : n = 3009) :
  let S := Finset.range n
  ∃ (k : ℕ) (f : Finset (ℕ × ℕ)),
    (∀ (p : ℕ × ℕ), p ∈ f → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ f → q ∈ f → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ f → q ∈ f → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ f → p.1 + p.2 ≤ n + 1) ∧
    f.card = k ∧
    k = 1203 ∧
    (∀ (m : ℕ) (g : Finset (ℕ × ℕ)),
      (∀ (p : ℕ × ℕ), p ∈ g → p.1 ∈ S ∧ p.2 ∈ S ∧ p.1 < p.2) →
      (∀ (p q : ℕ × ℕ), p ∈ g → q ∈ g → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
      (∀ (p q : ℕ × ℕ), p ∈ g → q ∈ g → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
      (∀ (p : ℕ × ℕ), p ∈ g → p.1 + p.2 ≤ n + 1) →
      g.card = m →
      m ≤ k) :=
by
  sorry

end NUMINAMATH_CALUDE_max_pairs_sum_l1874_187488


namespace NUMINAMATH_CALUDE_standard_deviation_proof_l1874_187498

/-- The standard deviation of a test score distribution. -/
def standard_deviation : ℝ := 20

/-- The mean score of the test. -/
def mean_score : ℝ := 60

/-- The lowest possible score within 2 standard deviations of the mean. -/
def lowest_score : ℝ := 20

/-- Theorem stating that the standard deviation is correct given the conditions. -/
theorem standard_deviation_proof :
  lowest_score = mean_score - 2 * standard_deviation :=
by sorry

end NUMINAMATH_CALUDE_standard_deviation_proof_l1874_187498


namespace NUMINAMATH_CALUDE_system_solution_unique_l1874_187453

theorem system_solution_unique (x y : ℝ) : 
  (x + 3 * y = 2 ∧ 4 * x - y = 8) ↔ (x = 2 ∧ y = 0) := by
sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1874_187453


namespace NUMINAMATH_CALUDE_ratio_problem_l1874_187438

theorem ratio_problem (a b : ℝ) : 
  (a / b = 5 / 1) → (a = 45) → (b = 9) := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1874_187438


namespace NUMINAMATH_CALUDE_factor_expression_l1874_187424

theorem factor_expression (x y : ℝ) : -x^2*y + 6*y^2*x - 9*y^3 = -y*(x-3*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1874_187424


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1874_187461

theorem sum_of_fractions : (3 : ℚ) / 462 + 17 / 42 + 1 / 11 = 116 / 231 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1874_187461


namespace NUMINAMATH_CALUDE_revenue_calculation_l1874_187445

/-- Calculates the total revenue given the salary expense and ratio of salary to stock purchase --/
def total_revenue (salary_expense : ℚ) (salary_ratio : ℚ) (stock_ratio : ℚ) : ℚ :=
  salary_expense * (salary_ratio + stock_ratio) / salary_ratio

/-- Proves that the total revenue is 3000 given the conditions --/
theorem revenue_calculation :
  let salary_expense : ℚ := 800
  let salary_ratio : ℚ := 4
  let stock_ratio : ℚ := 11
  total_revenue salary_expense salary_ratio stock_ratio = 3000 := by
sorry

#eval total_revenue 800 4 11

end NUMINAMATH_CALUDE_revenue_calculation_l1874_187445


namespace NUMINAMATH_CALUDE_line_in_plane_theorem_l1874_187425

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relations we need
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_line_line : Line → Line → Prop)
variable (passes_through : Line → Point → Prop)
variable (point_in_plane : Point → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_in_plane_theorem 
  (a b : Line) (α : Plane) (M : Point)
  (h1 : parallel_line_plane a α)
  (h2 : parallel_line_line b a)
  (h3 : passes_through b M)
  (h4 : point_in_plane M α) :
  line_in_plane b α :=
sorry

end NUMINAMATH_CALUDE_line_in_plane_theorem_l1874_187425


namespace NUMINAMATH_CALUDE_members_playing_neither_in_given_club_l1874_187400

/-- Represents a music club with members playing different instruments -/
structure MusicClub where
  total : ℕ
  guitar : ℕ
  piano : ℕ
  both : ℕ

/-- Calculates the number of members who don't play either instrument -/
def membersPlayingNeither (club : MusicClub) : ℕ :=
  club.total - (club.guitar + club.piano - club.both)

/-- Theorem stating the number of members not playing either instrument in the given club -/
theorem members_playing_neither_in_given_club :
  let club : MusicClub := {
    total := 80,
    guitar := 50,
    piano := 40,
    both := 25
  }
  membersPlayingNeither club = 15 := by
  sorry

end NUMINAMATH_CALUDE_members_playing_neither_in_given_club_l1874_187400


namespace NUMINAMATH_CALUDE_candles_from_leftovers_l1874_187401

/-- Represents the number of candles of a certain size --/
structure CandleSet where
  count : ℕ
  size : ℚ

/-- Calculates the total wax from a set of candles --/
def waxFrom (cs : CandleSet) (leftoverRatio : ℚ) : ℚ :=
  cs.count * cs.size * leftoverRatio

/-- The main theorem --/
theorem candles_from_leftovers 
  (leftoverRatio : ℚ)
  (bigCandles smallCandles tinyCandles : CandleSet)
  (newCandleSize : ℚ)
  (h_leftover : leftoverRatio = 1/10)
  (h_big : bigCandles = ⟨5, 20⟩)
  (h_small : smallCandles = ⟨5, 5⟩)
  (h_tiny : tinyCandles = ⟨25, 1⟩)
  (h_new : newCandleSize = 5) :
  (waxFrom bigCandles leftoverRatio + 
   waxFrom smallCandles leftoverRatio + 
   waxFrom tinyCandles leftoverRatio) / newCandleSize = 3 := by
  sorry

end NUMINAMATH_CALUDE_candles_from_leftovers_l1874_187401


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l1874_187427

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l1874_187427


namespace NUMINAMATH_CALUDE_sequence_ratio_l1874_187492

-- Define the arithmetic sequence
def arithmetic_sequence (a b : ℝ) : Prop :=
  ∃ r : ℝ, b - a = r ∧ -4 - b = r ∧ a - (-1) = r

-- Define the geometric sequence
def geometric_sequence (c d e : ℝ) : Prop :=
  ∃ q : ℝ, c = -1 * q ∧ d = c * q ∧ e = d * q ∧ -4 = e * q

-- State the theorem
theorem sequence_ratio (a b c d e : ℝ) 
  (h1 : arithmetic_sequence a b)
  (h2 : geometric_sequence c d e) :
  (b - a) / d = 1/2 := by sorry

end NUMINAMATH_CALUDE_sequence_ratio_l1874_187492


namespace NUMINAMATH_CALUDE_exponential_linear_critical_point_l1874_187437

/-- A function with a positive critical point -/
def has_positive_critical_point (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ (deriv f) x = 0

/-- The main theorem -/
theorem exponential_linear_critical_point (a : ℝ) :
  has_positive_critical_point (fun x => Real.exp x + a * x) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_exponential_linear_critical_point_l1874_187437


namespace NUMINAMATH_CALUDE_bluegrass_percentage_in_x_l1874_187417

/-- Represents a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The final mixture of X and Y -/
def finalMixture (x y : SeedMixture) (xWeight : ℝ) : SeedMixture :=
  { ryegrass := xWeight * x.ryegrass + (1 - xWeight) * y.ryegrass,
    bluegrass := xWeight * x.bluegrass + (1 - xWeight) * y.bluegrass,
    fescue := xWeight * x.fescue + (1 - xWeight) * y.fescue }

theorem bluegrass_percentage_in_x 
  (x : SeedMixture)
  (y : SeedMixture)
  (h1 : x.ryegrass = 0.4)
  (h2 : y.ryegrass = 0.25)
  (h3 : y.fescue = 0.75)
  (h4 : (finalMixture x y 0.6667).ryegrass = 0.35)
  : x.bluegrass = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_bluegrass_percentage_in_x_l1874_187417


namespace NUMINAMATH_CALUDE_quadrant_I_solution_range_l1874_187418

theorem quadrant_I_solution_range (c : ℝ) :
  (∃ x y : ℝ, x - y = 5 ∧ 2 * c * x + y = 8 ∧ x > 0 ∧ y > 0) ↔ -1/2 < c ∧ c < 4/5 := by
  sorry

end NUMINAMATH_CALUDE_quadrant_I_solution_range_l1874_187418


namespace NUMINAMATH_CALUDE_pushups_total_l1874_187435

/-- The number of push-ups Zachary and David did altogether -/
def total_pushups (zachary_pushups : ℕ) (david_extra_pushups : ℕ) : ℕ :=
  zachary_pushups + (zachary_pushups + david_extra_pushups)

/-- Theorem stating that given the conditions, the total number of push-ups is 146 -/
theorem pushups_total : total_pushups 44 58 = 146 := by
  sorry

end NUMINAMATH_CALUDE_pushups_total_l1874_187435


namespace NUMINAMATH_CALUDE_solve_fish_problem_l1874_187421

def fish_problem (initial_fish : ℕ) (yearly_increase : ℕ) (years : ℕ) (final_fish : ℕ) : ℕ → Prop :=
  λ yearly_deaths : ℕ =>
    initial_fish + years * yearly_increase - years * yearly_deaths = final_fish

theorem solve_fish_problem :
  ∃ yearly_deaths : ℕ, fish_problem 2 2 5 7 yearly_deaths :=
by
  sorry

end NUMINAMATH_CALUDE_solve_fish_problem_l1874_187421


namespace NUMINAMATH_CALUDE_selling_price_calculation_l1874_187476

theorem selling_price_calculation (cost_price : ℝ) (gain_percent : ℝ) (selling_price : ℝ) : 
  cost_price = 110 →
  gain_percent = 13.636363636363626 →
  selling_price = cost_price * (1 + gain_percent / 100) →
  selling_price = 125 := by
sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l1874_187476


namespace NUMINAMATH_CALUDE_symmetric_point_about_x_axis_l1874_187449

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of symmetry about the x-axis --/
def symmetricAboutXAxis (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

/-- The theorem to prove --/
theorem symmetric_point_about_x_axis :
  let A : Point := ⟨2, 1⟩
  let B : Point := ⟨2, -1⟩
  symmetricAboutXAxis A B := by sorry

end NUMINAMATH_CALUDE_symmetric_point_about_x_axis_l1874_187449


namespace NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l1874_187444

theorem mans_age_to_sons_age_ratio : 
  ∀ (sons_current_age mans_current_age : ℕ),
    sons_current_age = 26 →
    mans_current_age = sons_current_age + 28 →
    (mans_current_age + 2) / (sons_current_age + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l1874_187444


namespace NUMINAMATH_CALUDE_incorrect_inequality_l1874_187415

theorem incorrect_inequality (a b : ℝ) (h : a < b) : ¬(-4*a < -4*b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l1874_187415


namespace NUMINAMATH_CALUDE_problem_solution_l1874_187428

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) (sum : x + y = 5) :
  x = (7 + Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1874_187428


namespace NUMINAMATH_CALUDE_common_divisors_84_90_l1874_187414

theorem common_divisors_84_90 : 
  (Finset.filter (λ x => x ∣ 84 ∧ x ∣ 90) (Finset.range (min 84 90 + 1))).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_common_divisors_84_90_l1874_187414


namespace NUMINAMATH_CALUDE_last_four_digits_l1874_187411

theorem last_four_digits : (301 * 402 * 503 * 604 * 646 * 547 * 448 * 349) ^ 4 % 10000 = 5856 := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_l1874_187411


namespace NUMINAMATH_CALUDE_tank_width_is_six_l1874_187469

/-- Represents the properties of a rectangular tank being filled with water. -/
structure Tank where
  fill_rate : ℝ  -- Cubic feet per hour
  fill_time : ℝ  -- Hours
  length : ℝ     -- Feet
  depth : ℝ      -- Feet

/-- Calculates the volume of a rectangular tank. -/
def tank_volume (t : Tank) (width : ℝ) : ℝ :=
  t.length * width * t.depth

/-- Calculates the volume of water filled in the tank. -/
def filled_volume (t : Tank) : ℝ :=
  t.fill_rate * t.fill_time

/-- Theorem stating that the width of the tank is 6 feet. -/
theorem tank_width_is_six (t : Tank) 
  (h1 : t.fill_rate = 5)
  (h2 : t.fill_time = 60)
  (h3 : t.length = 10)
  (h4 : t.depth = 5) :
  ∃ (w : ℝ), w = 6 ∧ tank_volume t w = filled_volume t :=
sorry

end NUMINAMATH_CALUDE_tank_width_is_six_l1874_187469


namespace NUMINAMATH_CALUDE_simplify_fraction_division_l1874_187471

theorem simplify_fraction_division (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 2) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 3*x + 2) / (x^2 - 4*x + 4)) = (x - 2) / (x - 3) :=
by sorry

end NUMINAMATH_CALUDE_simplify_fraction_division_l1874_187471


namespace NUMINAMATH_CALUDE_integral_inequality_l1874_187496

noncomputable def a : ℝ := ∫ x in (1:ℝ)..2, 1/x
noncomputable def b : ℝ := ∫ x in (1:ℝ)..3, 1/x
noncomputable def c : ℝ := ∫ x in (1:ℝ)..5, 1/x

theorem integral_inequality : c/5 < a/2 ∧ a/2 < b/3 := by
  sorry

end NUMINAMATH_CALUDE_integral_inequality_l1874_187496


namespace NUMINAMATH_CALUDE_angle_GFH_l1874_187404

-- Define the sphere and points
def Sphere : Type := Unit
def Point : Type := Unit
def Q : Point := Unit.unit
def F : Point := Unit.unit
def G : Point := Unit.unit
def H : Point := Unit.unit

-- Define the properties
def radius (s : Sphere) : ℝ := 4
def touches_parallel_lines (s : Sphere) (F G H : Point) : Prop := sorry
def area_triangle (A B C : Point) : ℝ := sorry
def angle (A B C : Point) : ℝ := sorry

-- State the theorem
theorem angle_GFH (s : Sphere) :
  radius s = 4 →
  touches_parallel_lines s F G H →
  area_triangle Q G H = 4 * Real.sqrt 2 →
  area_triangle F G H > 16 →
  angle G F H = 67.5 := by sorry

end NUMINAMATH_CALUDE_angle_GFH_l1874_187404


namespace NUMINAMATH_CALUDE_largest_consecutive_odd_sum_55_l1874_187460

theorem largest_consecutive_odd_sum_55 :
  (∃ (n : ℕ) (x : ℕ),
    n > 0 ∧
    x > 0 ∧
    x % 2 = 1 ∧
    n * (x + n - 1) = 55 ∧
    ∀ (m : ℕ), m > n →
      ¬∃ (y : ℕ), y > 0 ∧ y % 2 = 1 ∧ m * (y + m - 1) = 55) →
  (∃ (x : ℕ),
    x > 0 ∧
    x % 2 = 1 ∧
    11 * (x + 11 - 1) = 55 ∧
    ∀ (m : ℕ), m > 11 →
      ¬∃ (y : ℕ), y > 0 ∧ y % 2 = 1 ∧ m * (y + m - 1) = 55) :=
by sorry

end NUMINAMATH_CALUDE_largest_consecutive_odd_sum_55_l1874_187460


namespace NUMINAMATH_CALUDE_unique_solution_for_m_squared_minus_eight_equals_three_to_n_l1874_187472

theorem unique_solution_for_m_squared_minus_eight_equals_three_to_n :
  ∀ m n : ℕ, m^2 - 8 = 3^n ↔ m = 3 ∧ n = 0 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_for_m_squared_minus_eight_equals_three_to_n_l1874_187472


namespace NUMINAMATH_CALUDE_chord_count_l1874_187489

/-- The number of chords formed by connecting any two of n points on a circle's circumference -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- There are 9 points on the circumference of a circle -/
def num_points : ℕ := 9

theorem chord_count : num_chords num_points = 36 := by
  sorry

end NUMINAMATH_CALUDE_chord_count_l1874_187489


namespace NUMINAMATH_CALUDE_nested_radical_value_l1874_187450

theorem nested_radical_value : 
  ∃ x : ℝ, x = Real.sqrt (2 + x) ∧ x ≥ 0 ∧ 2 + x ≥ 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_value_l1874_187450


namespace NUMINAMATH_CALUDE_angle_measure_l1874_187420

theorem angle_measure (x : ℝ) : 
  (90 - x = 2/3 * (180 - x) - 40) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l1874_187420


namespace NUMINAMATH_CALUDE_fedya_can_keep_below_1000_l1874_187463

/-- Represents the state of the number on the screen -/
structure ScreenNumber where
  value : ℕ
  minutes : ℕ

/-- Increases the number by 102 -/
def increment (n : ScreenNumber) : ScreenNumber :=
  { value := n.value + 102, minutes := n.minutes + 1 }

/-- Rearranges the digits of a number -/
def rearrange (n : ℕ) : ℕ := sorry

/-- Fedya's strategy to keep the number below 1000 -/
def fedya_strategy (n : ScreenNumber) : ScreenNumber :=
  if n.value < 1000 then n else { n with value := rearrange n.value }

/-- Theorem stating that Fedya can always keep the number below 1000 -/
theorem fedya_can_keep_below_1000 :
  ∀ (n : ℕ), n < 1000 →
  ∃ (strategy : ℕ → ScreenNumber),
    (∀ (k : ℕ), (strategy k).value < 1000) ∧
    strategy 0 = { value := 123, minutes := 0 } ∧
    (∀ (k : ℕ), strategy (k + 1) = fedya_strategy (increment (strategy k))) :=
sorry

end NUMINAMATH_CALUDE_fedya_can_keep_below_1000_l1874_187463


namespace NUMINAMATH_CALUDE_percentage_students_taking_music_l1874_187490

/-- The percentage of students taking music, given the total number of students
    and the number of students taking dance and art. -/
theorem percentage_students_taking_music
  (total_students : ℕ)
  (dance_students : ℕ)
  (art_students : ℕ)
  (h1 : total_students = 400)
  (h2 : dance_students = 120)
  (h3 : art_students = 200) :
  (((total_students - dance_students - art_students) : ℚ) / total_students) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_students_taking_music_l1874_187490


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1874_187470

theorem negation_of_universal_proposition :
  (¬ ∀ (x : ℕ+), (1/2 : ℝ)^(x : ℝ) ≤ 1/2) ↔ (∃ (x : ℕ+), (1/2 : ℝ)^(x : ℝ) > 1/2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1874_187470


namespace NUMINAMATH_CALUDE_subcommittee_count_l1874_187479

/-- The number of members in the planning committee -/
def totalMembers : ℕ := 12

/-- The number of professors in the planning committee -/
def professorCount : ℕ := 5

/-- The size of the subcommittee -/
def subcommitteeSize : ℕ := 4

/-- The minimum number of professors required in the subcommittee -/
def minProfessors : ℕ := 2

/-- Calculates the number of valid subcommittees -/
def validSubcommittees : ℕ := sorry

theorem subcommittee_count :
  validSubcommittees = 285 := by sorry

end NUMINAMATH_CALUDE_subcommittee_count_l1874_187479


namespace NUMINAMATH_CALUDE_last_two_digits_of_2006_factorial_l1874_187499

theorem last_two_digits_of_2006_factorial (n : ℕ) (h : n = 2006) : n! % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_2006_factorial_l1874_187499


namespace NUMINAMATH_CALUDE_ava_mia_difference_l1874_187455

/-- The number of shells each person has -/
structure ShellCounts where
  david : ℕ
  mia : ℕ
  ava : ℕ
  alice : ℕ

/-- The conditions of the problem -/
def problem_conditions (counts : ShellCounts) : Prop :=
  counts.david = 15 ∧
  counts.mia = 4 * counts.david ∧
  counts.ava > counts.mia ∧
  counts.alice = counts.ava / 2 ∧
  counts.david + counts.mia + counts.ava + counts.alice = 195

/-- The theorem to prove -/
theorem ava_mia_difference (counts : ShellCounts) :
  problem_conditions counts → counts.ava - counts.mia = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_ava_mia_difference_l1874_187455


namespace NUMINAMATH_CALUDE_largest_number_l1874_187419

-- Define the base conversion function
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

-- Define the numbers in their respective bases
def A : List Nat := [1, 0, 1, 1, 1, 1]
def B : List Nat := [1, 2, 1, 0]
def C : List Nat := [1, 1, 2]
def D : List Nat := [6, 9]

-- Theorem statement
theorem largest_number :
  to_decimal D 12 > to_decimal A 2 ∧
  to_decimal D 12 > to_decimal B 3 ∧
  to_decimal D 12 > to_decimal C 8 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l1874_187419


namespace NUMINAMATH_CALUDE_polynomial_degree_bound_l1874_187467

theorem polynomial_degree_bound (m n k : ℕ) (P : Polynomial ℤ) :
  m > 0 →
  n > 0 →
  k ≥ 2 →
  (∀ i, Odd (P.coeff i)) →
  P.degree = n →
  (X - 1 : Polynomial ℤ) ^ m ∣ P →
  m ≥ 2^k →
  n ≥ 2^(k+1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_bound_l1874_187467


namespace NUMINAMATH_CALUDE_sara_hotdog_cost_l1874_187406

/-- The cost of Sara's lunch items -/
structure LunchCost where
  total : ℝ
  salad : ℝ
  hotdog : ℝ

/-- Sara's lunch satisfies the given conditions -/
def sara_lunch : LunchCost where
  total := 10.46
  salad := 5.10
  hotdog := 10.46 - 5.10

/-- Theorem: Sara spent $5.36 on the hotdog -/
theorem sara_hotdog_cost : sara_lunch.hotdog = 5.36 := by
  sorry

end NUMINAMATH_CALUDE_sara_hotdog_cost_l1874_187406


namespace NUMINAMATH_CALUDE_NoahAndLucasAreZoesSiblings_l1874_187458

-- Define the characteristics
inductive EyeColor
| Green
| Gray

inductive HairColor
| Red
| Brown

inductive Height
| Tall
| Short

-- Define a child's characteristics
structure ChildCharacteristics where
  eyeColor : EyeColor
  hairColor : HairColor
  height : Height

-- Define the children
def Emma : ChildCharacteristics := ⟨EyeColor.Green, HairColor.Red, Height.Tall⟩
def Zoe : ChildCharacteristics := ⟨EyeColor.Gray, HairColor.Brown, Height.Short⟩
def Liam : ChildCharacteristics := ⟨EyeColor.Green, HairColor.Brown, Height.Short⟩
def Noah : ChildCharacteristics := ⟨EyeColor.Gray, HairColor.Red, Height.Tall⟩
def Mia : ChildCharacteristics := ⟨EyeColor.Green, HairColor.Red, Height.Short⟩
def Lucas : ChildCharacteristics := ⟨EyeColor.Gray, HairColor.Brown, Height.Tall⟩

-- Define a function to check if two children share a characteristic
def shareCharacteristic (c1 c2 : ChildCharacteristics) : Prop :=
  c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor ∨ c1.height = c2.height

-- Theorem to prove
theorem NoahAndLucasAreZoesSiblings :
  (shareCharacteristic Zoe Noah ∧ shareCharacteristic Zoe Lucas ∧ shareCharacteristic Noah Lucas) ∧
  (¬(shareCharacteristic Zoe Emma ∧ shareCharacteristic Zoe Mia ∧ shareCharacteristic Emma Mia)) ∧
  (¬(shareCharacteristic Zoe Liam ∧ shareCharacteristic Zoe Mia ∧ shareCharacteristic Liam Mia)) :=
by sorry

end NUMINAMATH_CALUDE_NoahAndLucasAreZoesSiblings_l1874_187458


namespace NUMINAMATH_CALUDE_share_of_y_l1874_187466

theorem share_of_y (total : ℝ) (x y z : ℝ) : 
  total = 273 →
  y = (45/100) * x →
  z = (50/100) * x →
  total = x + y + z →
  y = 63 := by
sorry

end NUMINAMATH_CALUDE_share_of_y_l1874_187466


namespace NUMINAMATH_CALUDE_geometric_sum_problem_l1874_187456

/-- Sum of a finite geometric series -/
def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sum_problem : 
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometric_sum a r n = 4095/12288 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_problem_l1874_187456


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1874_187408

theorem simplify_and_evaluate (x y : ℚ) (hx : x = 1/2) (hy : y = -3) :
  (x - 2*y)^2 - (x + y)*(x - y) - 5*y^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1874_187408


namespace NUMINAMATH_CALUDE_total_fish_caught_l1874_187422

/-- Represents the number of fish caught per fishing line -/
def fish_per_line : ℕ := 3

/-- Represents the initial number of fishing lines -/
def initial_lines : ℕ := 226

/-- Represents the number of broken and discarded fishing lines -/
def broken_lines : ℕ := 3

/-- Theorem stating the total number of fish caught by the fisherman -/
theorem total_fish_caught :
  (initial_lines - broken_lines) * fish_per_line = 669 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_caught_l1874_187422


namespace NUMINAMATH_CALUDE_line_points_property_l1874_187452

theorem line_points_property (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h1 : y₁ = -2 * x₁ + 3)
  (h2 : y₂ = -2 * x₂ + 3)
  (h3 : y₃ = -2 * x₃ + 3)
  (h4 : x₁ < x₂)
  (h5 : x₂ < x₃)
  (h6 : x₂ * x₃ < 0) :
  y₁ * y₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_points_property_l1874_187452


namespace NUMINAMATH_CALUDE_negative_cube_squared_l1874_187497

theorem negative_cube_squared (x : ℝ) : (-x^3)^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l1874_187497


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_intersection_complements_A_B_l1874_187483

-- Define the universal set U as ℝ
def U := Set ℝ

-- Define set A
def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}

-- Define set B
def B : Set ℝ := {x | 2 < x ∧ x < 8}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 5} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 8} := by sorry

-- Theorem for the intersection of complements of A and B
theorem intersection_complements_A_B : (Aᶜ : Set ℝ) ∩ (Bᶜ : Set ℝ) = {x : ℝ | x < 1 ∨ x ≥ 8} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_intersection_complements_A_B_l1874_187483


namespace NUMINAMATH_CALUDE_waiter_problem_l1874_187429

/-- Calculates the number of men at tables given the number of tables, women, and average customers per table. -/
def number_of_men (tables : Float) (women : Float) (avg_customers : Float) : Float :=
  tables * avg_customers - women

/-- Theorem stating that given 9.0 tables, 7.0 women, and an average of 1.111111111 customers per table, the number of men at the tables is 3.0. -/
theorem waiter_problem :
  number_of_men 9.0 7.0 1.111111111 = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_waiter_problem_l1874_187429


namespace NUMINAMATH_CALUDE_dalton_needs_four_more_l1874_187486

def jump_rope_cost : ℕ := 7
def board_game_cost : ℕ := 12
def playground_ball_cost : ℕ := 4
def saved_allowance : ℕ := 6
def money_from_uncle : ℕ := 13

theorem dalton_needs_four_more :
  jump_rope_cost + board_game_cost + playground_ball_cost - (saved_allowance + money_from_uncle) = 4 := by
  sorry

end NUMINAMATH_CALUDE_dalton_needs_four_more_l1874_187486


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l1874_187440

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a * b * c * d * e = 15120 →
  e = 9 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l1874_187440


namespace NUMINAMATH_CALUDE_quadratic_equations_properties_l1874_187451

/-- The quadratic equation x^2 + mx + 1 = 0 has two distinct negative real roots -/
def has_two_distinct_negative_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

/-- The quadratic equation 4x^2 + (4m-2)x + 1 = 0 does not have any real roots -/
def has_no_real_roots (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + (4*m-2)*x + 1 ≠ 0

theorem quadratic_equations_properties (m : ℝ) :
  (has_no_real_roots m ↔ 1/2 < m ∧ m < 3/2) ∧
  (has_two_distinct_negative_roots m ∧ ¬has_no_real_roots m ↔ m > 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_properties_l1874_187451


namespace NUMINAMATH_CALUDE_value_of_x_minus_4y_l1874_187462

theorem value_of_x_minus_4y (x y : ℝ) (h1 : x + y = 5) (h2 : 2 * x - 3 * y = 10) :
  x - 4 * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_minus_4y_l1874_187462


namespace NUMINAMATH_CALUDE_union_of_P_and_complement_of_Q_l1874_187413

open Set

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}
def Q : Set ℝ := {x | x^2 - 4 < 0}

-- State the theorem
theorem union_of_P_and_complement_of_Q :
  P ∪ (univ \ Q) = Iic (-2) ∪ Ici 1 := by sorry

end NUMINAMATH_CALUDE_union_of_P_and_complement_of_Q_l1874_187413


namespace NUMINAMATH_CALUDE_cos_n_eq_sin_312_l1874_187493

theorem cos_n_eq_sin_312 :
  ∃ (n : ℤ), -90 ≤ n ∧ n ≤ 90 ∧ (Real.cos (n * π / 180) = Real.sin (312 * π / 180)) ∧ n = 42 := by
  sorry

end NUMINAMATH_CALUDE_cos_n_eq_sin_312_l1874_187493


namespace NUMINAMATH_CALUDE_m_eq_n_necessary_not_sufficient_l1874_187487

/-- Defines a circle in R^2 --/
def is_circle (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b r : ℝ), r > 0 ∧ ∀ (x y : ℝ), f x y = 0 ↔ (x - a)^2 + (y - b)^2 = r^2

/-- The equation mx^2 + ny^2 = 3 --/
def equation (m n : ℝ) (x y : ℝ) : ℝ := m * x^2 + n * y^2 - 3

theorem m_eq_n_necessary_not_sufficient :
  (∀ m n : ℝ, is_circle (equation m n) → m = n) ∧
  (∃ m n : ℝ, m = n ∧ ¬is_circle (equation m n)) :=
sorry

end NUMINAMATH_CALUDE_m_eq_n_necessary_not_sufficient_l1874_187487


namespace NUMINAMATH_CALUDE_x_is_bounded_l1874_187412

/-- Product of all decimal digits of a natural number -/
def P (x : ℕ) : ℕ := sorry

/-- Sequence defined recursively by xₙ₊₁ = xₙ + P(xₙ) -/
def x : ℕ → ℕ
  | 0 => sorry  -- x₁ is some positive integer
  | n + 1 => x n + P (x n)

/-- The sequence (xₙ) is bounded -/
theorem x_is_bounded : ∃ (M : ℕ), ∀ (n : ℕ), x n ≤ M := by sorry

end NUMINAMATH_CALUDE_x_is_bounded_l1874_187412


namespace NUMINAMATH_CALUDE_prob_A_misses_at_least_once_prob_A_hits_twice_B_hits_thrice_l1874_187485

-- Define the probabilities of hitting the target for A and B
def prob_A_hit : ℚ := 2/3
def prob_B_hit : ℚ := 3/4

-- Define the number of shots
def num_shots : ℕ := 4

-- Theorem for the first question
theorem prob_A_misses_at_least_once :
  1 - prob_A_hit ^ num_shots = 65/81 :=
sorry

-- Theorem for the second question
theorem prob_A_hits_twice_B_hits_thrice :
  (Nat.choose num_shots 2 : ℚ) * prob_A_hit^2 * (1 - prob_A_hit)^(num_shots - 2) *
  (Nat.choose num_shots 3 : ℚ) * prob_B_hit^3 * (1 - prob_B_hit)^(num_shots - 3) = 1/8 :=
sorry

end NUMINAMATH_CALUDE_prob_A_misses_at_least_once_prob_A_hits_twice_B_hits_thrice_l1874_187485


namespace NUMINAMATH_CALUDE_power_relation_l1874_187464

theorem power_relation (x : ℝ) (n : ℕ) (h : x^(2*n) = 3) : x^(4*n) = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_relation_l1874_187464


namespace NUMINAMATH_CALUDE_no_relationship_between_mites_and_wilt_resistance_l1874_187405

def total_plants : ℕ := 88
def infected_plants : ℕ := 33
def resistant_infected : ℕ := 19
def susceptible_infected : ℕ := 14
def not_infected_plants : ℕ := 55
def resistant_not_infected : ℕ := 28
def susceptible_not_infected : ℕ := 27

def chi_square (n a b c d : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

def critical_value : ℚ := 3841 / 1000

theorem no_relationship_between_mites_and_wilt_resistance :
  chi_square total_plants resistant_infected resistant_not_infected 
             susceptible_infected susceptible_not_infected < critical_value := by
  sorry

end NUMINAMATH_CALUDE_no_relationship_between_mites_and_wilt_resistance_l1874_187405


namespace NUMINAMATH_CALUDE_common_remainder_proof_l1874_187407

theorem common_remainder_proof : 
  let n := 1398 - 7
  (n % 7 = 5) ∧ (n % 9 = 5) ∧ (n % 11 = 5) :=
by sorry

end NUMINAMATH_CALUDE_common_remainder_proof_l1874_187407


namespace NUMINAMATH_CALUDE_jim_marathon_training_l1874_187484

theorem jim_marathon_training (days_phase1 days_phase2 days_phase3 : ℕ)
  (miles_per_day_phase1 miles_per_day_phase2 miles_per_day_phase3 : ℕ)
  (h1 : days_phase1 = 30)
  (h2 : days_phase2 = 30)
  (h3 : days_phase3 = 30)
  (h4 : miles_per_day_phase1 = 5)
  (h5 : miles_per_day_phase2 = 10)
  (h6 : miles_per_day_phase3 = 20) :
  days_phase1 * miles_per_day_phase1 +
  days_phase2 * miles_per_day_phase2 +
  days_phase3 * miles_per_day_phase3 = 1050 := by
  sorry

end NUMINAMATH_CALUDE_jim_marathon_training_l1874_187484


namespace NUMINAMATH_CALUDE_nathan_ate_four_boxes_l1874_187481

def gumballs_per_box : ℕ := 5
def gumballs_eaten : ℕ := 20

theorem nathan_ate_four_boxes : 
  gumballs_eaten / gumballs_per_box = 4 := by
  sorry

end NUMINAMATH_CALUDE_nathan_ate_four_boxes_l1874_187481


namespace NUMINAMATH_CALUDE_powerless_common_divisor_l1874_187491

def is_powerless_digit (d : ℕ) : Prop :=
  d ≤ 9 ∧ d ≠ 0 ∧ d ≠ 1 ∧ d ≠ 4 ∧ d ≠ 8 ∧ d ≠ 9

def is_powerless_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n ≤ 99 ∧ is_powerless_digit (n / 10) ∧ is_powerless_digit (n % 10)

def smallest_powerless : ℕ := 22
def largest_powerless : ℕ := 77

theorem powerless_common_divisor :
  is_powerless_number smallest_powerless ∧
  is_powerless_number largest_powerless ∧
  smallest_powerless % 11 = 0 ∧
  largest_powerless % 11 = 0 := by sorry

end NUMINAMATH_CALUDE_powerless_common_divisor_l1874_187491


namespace NUMINAMATH_CALUDE_cat_and_mouse_positions_after_347_moves_l1874_187403

/-- Represents the positions around a pentagon -/
inductive PentagonPosition
| Top
| RightUpper
| RightLower
| LeftLower
| LeftUpper

/-- Represents the positions for the mouse, including edges -/
inductive MousePosition
| TopLeftEdge
| LeftUpperVertex
| LeftMiddleEdge
| LeftLowerVertex
| BottomEdge
| RightLowerVertex
| RightMiddleEdge
| RightUpperVertex
| TopRightEdge
| TopVertex

/-- Function to determine the cat's position after a given number of moves -/
def catPosition (moves : ℕ) : PentagonPosition :=
  match moves % 5 with
  | 0 => PentagonPosition.LeftUpper
  | 1 => PentagonPosition.Top
  | 2 => PentagonPosition.RightUpper
  | 3 => PentagonPosition.RightLower
  | _ => PentagonPosition.LeftLower

/-- Function to determine the mouse's position after a given number of moves -/
def mousePosition (moves : ℕ) : MousePosition :=
  match moves % 10 with
  | 0 => MousePosition.TopVertex
  | 1 => MousePosition.TopLeftEdge
  | 2 => MousePosition.LeftUpperVertex
  | 3 => MousePosition.LeftMiddleEdge
  | 4 => MousePosition.LeftLowerVertex
  | 5 => MousePosition.BottomEdge
  | 6 => MousePosition.RightLowerVertex
  | 7 => MousePosition.RightMiddleEdge
  | 8 => MousePosition.RightUpperVertex
  | _ => MousePosition.TopRightEdge

theorem cat_and_mouse_positions_after_347_moves :
  (catPosition 347 = PentagonPosition.RightUpper) ∧
  (mousePosition 347 = MousePosition.RightMiddleEdge) := by
  sorry


end NUMINAMATH_CALUDE_cat_and_mouse_positions_after_347_moves_l1874_187403


namespace NUMINAMATH_CALUDE_special_circle_properties_special_circle_unique_l1874_187432

/-- The circle passing through points A(1,-1) and B(-1,1) with its center on the line x+y-2=0 -/
def special_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 4

theorem special_circle_properties :
  ∀ x y : ℝ,
    special_circle x y →
    ((x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1)) ∧
    ∃ c_x c_y : ℝ, c_x + c_y - 2 = 0 ∧
                   (x - c_x)^2 + (y - c_y)^2 = (c_x - 1)^2 + (c_y + 1)^2 :=
by
  sorry

theorem special_circle_unique :
  ∀ f : ℝ → ℝ → Prop,
    (∀ x y : ℝ, f x y → ((x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1))) →
    (∀ x y : ℝ, f x y → ∃ c_x c_y : ℝ, c_x + c_y - 2 = 0 ∧
                                      (x - c_x)^2 + (y - c_y)^2 = (c_x - 1)^2 + (c_y + 1)^2) →
    ∀ x y : ℝ, f x y ↔ special_circle x y :=
by
  sorry

end NUMINAMATH_CALUDE_special_circle_properties_special_circle_unique_l1874_187432


namespace NUMINAMATH_CALUDE_tan_alpha_eq_two_l1874_187459

theorem tan_alpha_eq_two (α : ℝ) (h : 2 * Real.sin α + Real.cos α = -Real.sqrt 5) :
  Real.tan α = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_eq_two_l1874_187459


namespace NUMINAMATH_CALUDE_equilateral_hyperbola_equation_l1874_187443

-- Define the hyperbola
def Hyperbola (x y : ℝ) := x^2 - y^2 = 8

-- Define the line containing one focus
def FocusLine (x y : ℝ) := 3*x - 4*y + 12 = 0

-- Theorem statement
theorem equilateral_hyperbola_equation :
  ∃ (a b : ℝ), 
    -- One focus is on the line
    FocusLine a b ∧
    -- The focus is on the x-axis (real axis)
    b = 0 ∧
    -- The hyperbola passes through this focus
    Hyperbola a b ∧
    -- The hyperbola is equilateral (a² = b²)
    a^2 = (8:ℝ) := by
  sorry

end NUMINAMATH_CALUDE_equilateral_hyperbola_equation_l1874_187443


namespace NUMINAMATH_CALUDE_sum_of_xyz_l1874_187474

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 30) (hxz : x * z = 60) (hyz : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l1874_187474


namespace NUMINAMATH_CALUDE_marble_replacement_l1874_187426

theorem marble_replacement (total : ℕ) (red blue yellow white black : ℕ) : 
  total = red + blue + yellow + white + black →
  red = (40 * total) / 100 →
  blue = (25 * total) / 100 →
  yellow = (10 * total) / 100 →
  white = (15 * total) / 100 →
  black = 20 →
  (blue + red / 3 : ℕ) = 77 := by
  sorry

end NUMINAMATH_CALUDE_marble_replacement_l1874_187426


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_l1874_187457

theorem arithmetic_geometric_progression (b c : ℝ) 
  (not_both_one : ¬(b = 1 ∧ c = 1))
  (arithmetic_prog : ∃ n : ℝ, b = 1 + n ∧ c = 1 + 2*n)
  (geometric_prog : c * b = c^2) : 
  100 * (b - c) = 75 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_l1874_187457


namespace NUMINAMATH_CALUDE_isle_of_misfortune_l1874_187494

/-- Represents a person who is either a knight (truth-teller) or a liar -/
inductive Person
| Knight
| Liar

/-- The total number of people in the group -/
def total_people : Nat := 101

/-- A function that returns true if removing a person results in a majority of liars -/
def majority_liars_if_removed (knights : Nat) (liars : Nat) (person : Person) : Prop :=
  match person with
  | Person.Knight => liars ≥ knights - 1
  | Person.Liar => liars - 1 ≥ knights

theorem isle_of_misfortune :
  ∀ (knights liars : Nat),
    knights + liars = total_people →
    (∀ (p : Person), majority_liars_if_removed knights liars p) →
    knights = 50 ∧ liars = 51 := by
  sorry

end NUMINAMATH_CALUDE_isle_of_misfortune_l1874_187494


namespace NUMINAMATH_CALUDE_equal_roots_condition_l1874_187434

/-- 
If the quadratic equation 2x^2 - ax + 2 = 0 has two equal real roots, 
then a = 4 or a = -4
-/
theorem equal_roots_condition (a : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - a * x + 2 = 0 ∧ 
   (∀ y : ℝ, 2 * y^2 - a * y + 2 = 0 → y = x)) → 
  (a = 4 ∨ a = -4) := by
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l1874_187434


namespace NUMINAMATH_CALUDE_exponent_calculations_l1874_187436

theorem exponent_calculations :
  (16 ^ (1/2 : ℝ) + (1/81 : ℝ) ^ (-1/4 : ℝ) - (-1/2 : ℝ) ^ (0 : ℝ) = 10/3) ∧
  (∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → 
    ((2 * a ^ (1/4 : ℝ) * b ^ (-1/3 : ℝ)) * (-3 * a ^ (-1/2 : ℝ) * b ^ (2/3 : ℝ))) / 
    (-1/4 * a ^ (-1/4 : ℝ) * b ^ (-2/3 : ℝ)) = 24 * b) := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculations_l1874_187436


namespace NUMINAMATH_CALUDE_area_PQR_l1874_187468

/-- Triangle ABC with given side lengths and points M, N, P, Q, R as described -/
structure TriangleABC where
  -- Side lengths
  AB : ℝ
  BC : ℝ
  CA : ℝ
  -- Point M on AB
  AM : ℝ
  MB : ℝ
  -- Point N on BC
  CN : ℝ
  NB : ℝ
  -- Conditions
  side_lengths : AB = 20 ∧ BC = 21 ∧ CA = 29
  M_ratio : AM / MB = 3 / 2
  N_ratio : CN / NB = 2
  -- Existence of points P, Q, R (not explicitly defined)
  P_exists : ∃ P : ℝ × ℝ, True  -- P on AC
  Q_exists : ∃ Q : ℝ × ℝ, True  -- Q on AC
  R_exists : ∃ R : ℝ × ℝ, True  -- R as intersection of MP and NQ
  MP_parallel_BC : True  -- MP is parallel to BC
  NQ_parallel_AB : True  -- NQ is parallel to AB

/-- The area of triangle PQR is 224/15 -/
theorem area_PQR (t : TriangleABC) : ∃ area_PQR : ℝ, area_PQR = 224/15 := by
  sorry

end NUMINAMATH_CALUDE_area_PQR_l1874_187468


namespace NUMINAMATH_CALUDE_sequence_properties_l1874_187454

def sequence_term (n : ℕ) : ℕ :=
  3 * (n^2 - n + 1)

theorem sequence_properties : 
  (∃ k, sequence_term k = 48 ∧ sequence_term (k + 1) = 63) ∧ 
  sequence_term 8 = 168 ∧
  sequence_term 2013 = 9120399 := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l1874_187454


namespace NUMINAMATH_CALUDE_stratified_sampling_proof_l1874_187439

/-- Represents a sampling method -/
inductive SamplingMethod
| Stratified
| Simple
| Cluster
| Systematic

/-- Represents the student population -/
structure Population where
  total : Nat
  male : Nat
  female : Nat

/-- Represents the sample -/
structure Sample where
  total : Nat
  male : Nat
  female : Nat

def is_stratified (pop : Population) (sam : Sample) : Prop :=
  (pop.male : Real) / pop.total = (sam.male : Real) / sam.total ∧
  (pop.female : Real) / pop.total = (sam.female : Real) / sam.total

theorem stratified_sampling_proof 
  (pop : Population) 
  (sam : Sample) 
  (h1 : pop.total = 1000) 
  (h2 : pop.male = 400) 
  (h3 : pop.female = 600) 
  (h4 : sam.total = 100) 
  (h5 : sam.male = 40) 
  (h6 : sam.female = 60) 
  (h7 : is_stratified pop sam) : 
  SamplingMethod.Stratified = SamplingMethod.Stratified :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_proof_l1874_187439


namespace NUMINAMATH_CALUDE_triangle_area_l1874_187430

/-- Given a triangle with sides in ratio 5:12:13 and perimeter 300, prove its area is 3000 -/
theorem triangle_area (a b c : ℝ) (h_ratio : (a, b, c) = (5 * (300 / 30), 12 * (300 / 30), 13 * (300 / 30))) 
  (h_perimeter : a + b + c = 300) : 
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 3000 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l1874_187430


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1874_187423

theorem solution_set_of_inequality (x : ℝ) :
  (3 * x^2 + 7 * x ≤ 2) ↔ (-2 ≤ x ∧ x ≤ 1/3) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1874_187423


namespace NUMINAMATH_CALUDE_ordering_abc_l1874_187446

theorem ordering_abc :
  let a : ℝ := Real.log 2
  let b : ℝ := 2023 / 2022
  let c : ℝ := Real.log 2023 / Real.log 2022
  a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_ordering_abc_l1874_187446


namespace NUMINAMATH_CALUDE_five_T_three_l1874_187477

-- Define the operation T
def T (a b : ℤ) : ℤ := 4*a + 7*b + 2*a*b

-- Theorem statement
theorem five_T_three : T 5 3 = 71 := by
  sorry

end NUMINAMATH_CALUDE_five_T_three_l1874_187477


namespace NUMINAMATH_CALUDE_sourball_candies_count_l1874_187475

/-- The number of sourball candies in the bucket initially -/
def initial_candies : ℕ := 30

/-- The number of candies Nellie can eat before crying -/
def nellie_candies : ℕ := 12

/-- The number of candies Jacob can eat before crying -/
def jacob_candies : ℕ := nellie_candies / 2

/-- The number of candies Lana can eat before crying -/
def lana_candies : ℕ := jacob_candies - 3

/-- The number of candies each person gets after division -/
def remaining_per_person : ℕ := 3

/-- The number of people -/
def num_people : ℕ := 3

theorem sourball_candies_count :
  initial_candies = nellie_candies + jacob_candies + lana_candies + remaining_per_person * num_people :=
by sorry

end NUMINAMATH_CALUDE_sourball_candies_count_l1874_187475


namespace NUMINAMATH_CALUDE_choose_3_from_13_l1874_187409

theorem choose_3_from_13 : Nat.choose 13 3 = 286 := by sorry

end NUMINAMATH_CALUDE_choose_3_from_13_l1874_187409
