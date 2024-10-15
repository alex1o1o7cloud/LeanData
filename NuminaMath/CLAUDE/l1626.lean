import Mathlib

namespace NUMINAMATH_CALUDE_clock_strikes_l1626_162683

/-- If a clock strikes three times in 12 seconds, it will strike six times in 30 seconds. -/
theorem clock_strikes (strike_interval : ℝ) : 
  (3 * strike_interval = 12) → (6 * strike_interval = 30) := by
  sorry

end NUMINAMATH_CALUDE_clock_strikes_l1626_162683


namespace NUMINAMATH_CALUDE_fibonacci_parity_l1626_162691

def E : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | n + 3 => E (n + 2) + E (n + 1)

def isEven (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem fibonacci_parity : 
  isEven (E 2021) ∧ ¬isEven (E 2022) ∧ ¬isEven (E 2023) := by sorry

end NUMINAMATH_CALUDE_fibonacci_parity_l1626_162691


namespace NUMINAMATH_CALUDE_more_sad_left_l1626_162685

/-- Represents the state of a player in the game -/
inductive PlayerState
| Sad
| Cheerful

/-- Represents the game with its rules and initial state -/
structure Game where
  initial_players : ℕ
  remaining_player : ℕ
  sad_left : ℕ
  cheerful_left : ℕ

/-- The game rules ensure that when only one player remains, more sad players have left than cheerful players -/
theorem more_sad_left (g : Game) 
  (h1 : g.initial_players = 36)
  (h2 : g.remaining_player = 1)
  (h3 : g.sad_left + g.cheerful_left = g.initial_players - g.remaining_player) :
  g.sad_left > g.cheerful_left := by
  sorry

#check more_sad_left

end NUMINAMATH_CALUDE_more_sad_left_l1626_162685


namespace NUMINAMATH_CALUDE_percent_decrease_l1626_162688

theorem percent_decrease (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 55) : 
  (original_price - sale_price) / original_price * 100 = 45 := by
  sorry

end NUMINAMATH_CALUDE_percent_decrease_l1626_162688


namespace NUMINAMATH_CALUDE_no_roots_of_composite_l1626_162673

/-- A quadratic function f(x) = x^2 + bx + c -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- The condition that f(x) = x has no real roots -/
def no_real_roots (b c : ℝ) : Prop := ∀ x : ℝ, f b c x ≠ x

/-- The theorem stating that if f(x) = x has no real roots, then f(f(x)) = x has no real roots -/
theorem no_roots_of_composite (b c : ℝ) (h : no_real_roots b c) :
  ∀ x : ℝ, f b c (f b c x) ≠ x := by
  sorry

end NUMINAMATH_CALUDE_no_roots_of_composite_l1626_162673


namespace NUMINAMATH_CALUDE_discount_difference_l1626_162615

/-- The cover price of the book in cents -/
def cover_price : ℕ := 3000

/-- The percentage discount as a fraction -/
def percent_discount : ℚ := 1/4

/-- The fixed discount in cents -/
def fixed_discount : ℕ := 500

/-- Applies the percentage discount followed by the fixed discount -/
def percent_then_fixed (price : ℕ) : ℚ :=
  (price : ℚ) * (1 - percent_discount) - fixed_discount

/-- Applies the fixed discount followed by the percentage discount -/
def fixed_then_percent (price : ℕ) : ℚ :=
  ((price : ℚ) - fixed_discount) * (1 - percent_discount)

theorem discount_difference :
  fixed_then_percent cover_price - percent_then_fixed cover_price = 125 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l1626_162615


namespace NUMINAMATH_CALUDE_machine_y_efficiency_l1626_162611

/-- The number of widgets produced by both machines -/
def total_widgets : ℕ := 1080

/-- The number of widgets Machine X produces per hour -/
def machine_x_rate : ℕ := 3

/-- The difference in hours between Machine X and Machine Y to produce the total widgets -/
def time_difference : ℕ := 60

/-- Calculate the percentage difference between two numbers -/
def percentage_difference (a b : ℚ) : ℚ := (b - a) / a * 100

/-- Theorem stating that Machine Y produces 20% more widgets per hour than Machine X -/
theorem machine_y_efficiency : 
  let machine_x_time := total_widgets / machine_x_rate
  let machine_y_time := machine_x_time - time_difference
  let machine_y_rate := total_widgets / machine_y_time
  percentage_difference machine_x_rate machine_y_rate = 20 := by sorry

end NUMINAMATH_CALUDE_machine_y_efficiency_l1626_162611


namespace NUMINAMATH_CALUDE_medical_team_composition_l1626_162600

theorem medical_team_composition (total : ℕ) 
  (female_nurses male_nurses female_doctors male_doctors : ℕ) :
  total = 13 →
  female_nurses + male_nurses + female_doctors + male_doctors = total →
  female_nurses + male_nurses ≥ female_doctors + male_doctors →
  male_doctors > female_nurses →
  female_nurses > male_nurses →
  female_doctors ≥ 1 →
  female_nurses = 4 ∧ male_nurses = 3 ∧ female_doctors = 1 ∧ male_doctors = 5 :=
by sorry

end NUMINAMATH_CALUDE_medical_team_composition_l1626_162600


namespace NUMINAMATH_CALUDE_range_of_a_l1626_162665

/-- Custom operation ⊗ -/
def custom_op (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the range of a given the condition -/
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, custom_op (x - a) (x + a) < 1) → -1/2 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1626_162665


namespace NUMINAMATH_CALUDE_number_solution_l1626_162693

theorem number_solution : ∃ x : ℝ, 3 * x - 5 = 40 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l1626_162693


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l1626_162639

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l1626_162639


namespace NUMINAMATH_CALUDE_matrix_sum_of_squares_l1626_162627

theorem matrix_sum_of_squares (x y z w : ℝ) :
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]
  (B.transpose = 2 • B⁻¹) → x^2 + y^2 + z^2 + w^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_matrix_sum_of_squares_l1626_162627


namespace NUMINAMATH_CALUDE_max_value_in_D_l1626_162687

-- Define the region D
def D : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 - p.2^2 = 0 ∧ p.1 ≤ 1 ∧ p.1 ≥ 0}

-- Define the objective function
def z (p : ℝ × ℝ) : ℝ := p.1 - 2*p.2 + 5

-- Theorem statement
theorem max_value_in_D :
  ∃ (m : ℝ), m = 8 ∧ ∀ p ∈ D, z p ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_in_D_l1626_162687


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l1626_162617

theorem sphere_volume_ratio (R : ℝ) (h : R > 0) :
  (4 / 3 * Real.pi * (2 * R)^3) / (4 / 3 * Real.pi * R^3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l1626_162617


namespace NUMINAMATH_CALUDE_binomial_sum_equals_power_of_two_l1626_162640

theorem binomial_sum_equals_power_of_two : 
  3^2006 - Nat.choose 2006 1 * 3^2005 + Nat.choose 2006 2 * 3^2004 - Nat.choose 2006 3 * 3^2003 +
  Nat.choose 2006 4 * 3^2002 - Nat.choose 2006 5 * 3^2001 + 
  -- ... (omitting middle terms for brevity)
  Nat.choose 2006 2004 * 3^2 - Nat.choose 2006 2005 * 3 + 1 = 2^2006 :=
by sorry

end NUMINAMATH_CALUDE_binomial_sum_equals_power_of_two_l1626_162640


namespace NUMINAMATH_CALUDE_pet_shop_dogs_l1626_162601

/-- Given a pet shop with dogs, bunnies, and birds in the ratio 3:9:11,
    and a total of 816 animals, prove that there are 105 dogs. -/
theorem pet_shop_dogs (total : ℕ) (h_total : total = 816) :
  let ratio_sum := 3 + 9 + 11
  let part_size := total / ratio_sum
  let dogs := 3 * part_size
  dogs = 105 := by
  sorry

#check pet_shop_dogs

end NUMINAMATH_CALUDE_pet_shop_dogs_l1626_162601


namespace NUMINAMATH_CALUDE_leigh_has_16_seashells_l1626_162670

/-- The number of seashells Leigh has, given the conditions of the problem -/
def leighs_seashells : ℕ :=
  let mimis_shells := 2 * 12  -- 2 dozen
  let kyles_shells := 2 * mimis_shells  -- twice as many as Mimi
  kyles_shells / 3  -- one-third of Kyle's shells

/-- Theorem stating that Leigh has 16 seashells -/
theorem leigh_has_16_seashells : leighs_seashells = 16 := by
  sorry

end NUMINAMATH_CALUDE_leigh_has_16_seashells_l1626_162670


namespace NUMINAMATH_CALUDE_prime_sum_equality_l1626_162671

theorem prime_sum_equality (p q : ℕ) (hp : Prime p) (hq : Prime q) 
  (h_sum : (Finset.range q).sum (λ i => p ^ (i + 1)) = (Finset.range p).sum (λ i => q ^ (i + 1))) : 
  p = q := by
sorry

end NUMINAMATH_CALUDE_prime_sum_equality_l1626_162671


namespace NUMINAMATH_CALUDE_ohara_triple_49_16_l1626_162628

/-- Definition of O'Hara triple -/
def is_ohara_triple (a b x : ℕ) : Prop :=
  Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) = x

/-- Theorem: If (49, 16, x) is an O'Hara triple, then x = 11 -/
theorem ohara_triple_49_16 (x : ℕ) :
  is_ohara_triple 49 16 x → x = 11 := by
  sorry

end NUMINAMATH_CALUDE_ohara_triple_49_16_l1626_162628


namespace NUMINAMATH_CALUDE_min_bailing_rate_l1626_162632

/-- Minimum bailing rate problem -/
theorem min_bailing_rate (distance : ℝ) (leak_rate : ℝ) (capacity : ℝ) (speed : ℝ) 
  (h1 : distance = 2)
  (h2 : leak_rate = 15)
  (h3 : capacity = 50)
  (h4 : speed = 3) : 
  ∃ (bailing_rate : ℝ), bailing_rate ≥ 14 ∧ 
  (distance / speed * 60 * (leak_rate - bailing_rate) ≤ capacity) := by
  sorry

end NUMINAMATH_CALUDE_min_bailing_rate_l1626_162632


namespace NUMINAMATH_CALUDE_socks_cost_proof_l1626_162620

/-- The cost of a uniform item without discount -/
structure UniformItem where
  shirt : ℝ
  pants : ℝ
  socks : ℝ

/-- The cost of a uniform item with discount -/
structure DiscountedUniformItem where
  shirt : ℝ
  pants : ℝ
  socks : ℝ

def team_size : ℕ := 12
def team_savings : ℝ := 36

def regular_uniform : UniformItem :=
  { shirt := 7.5,
    pants := 15,
    socks := 4.5 }  -- We use the answer here as we're proving this value

def discounted_uniform : DiscountedUniformItem :=
  { shirt := 6.75,
    pants := 13.5,
    socks := 3.75 }

theorem socks_cost_proof :
  let regular_total := team_size * (regular_uniform.shirt + regular_uniform.pants + regular_uniform.socks)
  let discounted_total := team_size * (discounted_uniform.shirt + discounted_uniform.pants + discounted_uniform.socks)
  regular_total - discounted_total = team_savings :=
by sorry

end NUMINAMATH_CALUDE_socks_cost_proof_l1626_162620


namespace NUMINAMATH_CALUDE_aquarium_water_after_45_days_l1626_162657

/-- Calculates the remaining water in an aquarium after a given time period. -/
def remainingWater (initialVolume : ℝ) (lossRate : ℝ) (days : ℝ) : ℝ :=
  initialVolume - lossRate * days

/-- Theorem stating the remaining water volume in the aquarium after 45 days. -/
theorem aquarium_water_after_45_days :
  remainingWater 500 1.2 45 = 446 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_water_after_45_days_l1626_162657


namespace NUMINAMATH_CALUDE_dish_price_l1626_162612

/-- The original price of a dish given specific discount and tip conditions --/
def original_price : ℝ → Prop :=
  λ price =>
    let john_payment := price * 0.9 + price * 0.15
    let jane_payment := price * 0.9 + price * 0.9 * 0.15
    john_payment - jane_payment = 0.60

theorem dish_price : ∃ (price : ℝ), original_price price ∧ price = 40 := by
  sorry

end NUMINAMATH_CALUDE_dish_price_l1626_162612


namespace NUMINAMATH_CALUDE_units_digit_of_base_l1626_162661

/-- Given a natural number, return its unit's digit -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The product of the given terms -/
def product (x : ℕ) : ℕ := (x ^ 41) * (41 ^ 14) * (14 ^ 87) * (87 ^ 76)

/-- The theorem stating that if the unit's digit of the product is 4, 
    then the unit's digit of x must be 1 -/
theorem units_digit_of_base (x : ℕ) : 
  unitsDigit (product x) = 4 → unitsDigit x = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_base_l1626_162661


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l1626_162681

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 16 = 48 → Nat.gcd n 16 = 8 → n = 24 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l1626_162681


namespace NUMINAMATH_CALUDE_square_difference_252_248_l1626_162637

theorem square_difference_252_248 : 252^2 - 248^2 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_252_248_l1626_162637


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1626_162625

theorem rectangle_perimeter (area : ℝ) (width : ℝ) (length : ℝ) :
  area = 500 →
  length = 2 * width →
  area = length * width →
  2 * (length + width) = 30 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1626_162625


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1626_162689

theorem necessary_but_not_sufficient :
  let p := fun x : ℝ => x^2 - 2*x ≥ 3
  let q := fun x : ℝ => -1 < x ∧ x < 2
  (∀ x, q x → ¬(p x)) ∧ 
  (∃ x, ¬(p x) ∧ ¬(q x)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1626_162689


namespace NUMINAMATH_CALUDE_tangent_line_equation_point_B_coordinates_fixed_point_on_AB_l1626_162645

-- Define the parabola Γ
def Γ (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define point D
def D (p x₀ y₀ : ℝ) : Prop := y₀^2 > 2*p*x₀

-- Define tangent line through D intersecting Γ at A and B
def tangent_line (p x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  D p x₀ y₀ ∧ Γ p x₁ y₁ ∧ Γ p x₂ y₂

-- Theorem 1: Line yy₁ = p(x + x₁) is tangent to Γ
theorem tangent_line_equation (p x₀ y₀ x₁ y₁ : ℝ) :
  tangent_line p x₀ y₀ x₁ y₁ x₁ y₁ → ∀ x y, y * y₁ = p * (x + x₁) := by sorry

-- Theorem 2: Coordinates of B when A(4, 4) and D on directrix
theorem point_B_coordinates (p : ℝ) :
  Γ p 4 4 → D p (-p/2) (3/2) → ∃ x₂ y₂, Γ p x₂ y₂ ∧ x₂ = 1/4 ∧ y₂ = -1 := by sorry

-- Theorem 3: AB passes through fixed point when D moves on x + p = 0
theorem fixed_point_on_AB (p x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) :
  tangent_line p x₀ y₀ x₁ y₁ x₂ y₂ → x₀ = -p → 
  ∃ k b, y₁ - y₂ = k * (x₁ - x₂) ∧ y₁ = k * x₁ + b ∧ 0 = k * p + b := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_point_B_coordinates_fixed_point_on_AB_l1626_162645


namespace NUMINAMATH_CALUDE_monotone_increasing_interval_l1626_162653

noncomputable def f (x : ℝ) : ℝ := (Real.cos x + Real.sin x) * Real.cos (x - Real.pi / 2)

theorem monotone_increasing_interval (k : ℤ) :
  StrictMonoOn f { x | k * Real.pi - Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + 3 * Real.pi / 8 } :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_interval_l1626_162653


namespace NUMINAMATH_CALUDE_jessica_seashells_l1626_162638

theorem jessica_seashells (initial_seashells : ℕ) (given_seashells : ℕ) :
  initial_seashells = 8 →
  given_seashells = 6 →
  initial_seashells - given_seashells = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_jessica_seashells_l1626_162638


namespace NUMINAMATH_CALUDE_xenia_earnings_l1626_162694

/-- Xenia's work and earnings over two weeks -/
theorem xenia_earnings 
  (hours_week1 : ℕ) 
  (hours_week2 : ℕ) 
  (wage : ℚ) 
  (extra_earnings : ℚ) 
  (h1 : hours_week1 = 12)
  (h2 : hours_week2 = 20)
  (h3 : extra_earnings = 36)
  (h4 : wage * (hours_week2 - hours_week1) = extra_earnings) :
  wage * (hours_week1 + hours_week2) = 144 :=
sorry

end NUMINAMATH_CALUDE_xenia_earnings_l1626_162694


namespace NUMINAMATH_CALUDE_min_cuts_for_20_gons_l1626_162658

/-- Represents a polygon with a given number of sides -/
structure Polygon where
  sides : ℕ

/-- Represents a cut operation on a piece of paper -/
inductive Cut
  | straight : Cut

/-- Represents the state of the paper cutting process -/
structure PaperState where
  pieces : ℕ
  polygons : List Polygon

/-- Defines the initial state with a single rectangular piece of paper -/
def initial_state : PaperState :=
  { pieces := 1, polygons := [⟨4⟩] }

/-- Applies a cut to a paper state -/
def apply_cut (state : PaperState) (cut : Cut) : PaperState :=
  { pieces := state.pieces + 1, polygons := state.polygons }

/-- Checks if the goal of at least 100 20-sided polygons is achieved -/
def goal_achieved (state : PaperState) : Prop :=
  (state.polygons.filter (λ p => p.sides = 20)).length ≥ 100

/-- The main theorem stating the minimum number of cuts required -/
theorem min_cuts_for_20_gons : 
  ∃ (n : ℕ), n = 1699 ∧ 
  (∀ (m : ℕ), m < n → 
    ¬∃ (cuts : List Cut), 
      goal_achieved (cuts.foldl apply_cut initial_state)) ∧
  (∃ (cuts : List Cut), 
    cuts.length = n ∧ 
    goal_achieved (cuts.foldl apply_cut initial_state)) :=
sorry

end NUMINAMATH_CALUDE_min_cuts_for_20_gons_l1626_162658


namespace NUMINAMATH_CALUDE_max_runs_in_match_l1626_162666

/-- Represents the number of overs in a cricket match -/
def overs : ℕ := 20

/-- Represents the maximum number of runs a batsman can score -/
def max_runs : ℕ := 663

/-- Represents the number of balls in an over -/
def balls_per_over : ℕ := 6

/-- Represents the maximum runs that can be scored off a single ball -/
def max_runs_per_ball : ℕ := 6

/-- Represents the total number of balls in the match -/
def total_balls : ℕ := overs * balls_per_over

/-- Theorem stating that under certain conditions, the maximum runs a batsman can score in the match is 663 -/
theorem max_runs_in_match : 
  ∃ (balls_faced : ℕ) (runs_per_ball : ℕ), 
    balls_faced ≤ total_balls ∧ 
    runs_per_ball ≤ max_runs_per_ball ∧ 
    balls_faced * runs_per_ball = max_runs :=
sorry

end NUMINAMATH_CALUDE_max_runs_in_match_l1626_162666


namespace NUMINAMATH_CALUDE_orthogonal_vectors_l1626_162674

theorem orthogonal_vectors (y : ℝ) : y = 28 / 3 →
  (3 : ℝ) * y + 7 * (-4 : ℝ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_l1626_162674


namespace NUMINAMATH_CALUDE_reflection_about_y_eq_neg_x_l1626_162675

def reflect_point (x y : ℝ) : ℝ × ℝ :=
  (-y, -x)

theorem reflection_about_y_eq_neg_x (x y : ℝ) :
  reflect_point 4 (-3) = (3, -4) := by
  sorry

end NUMINAMATH_CALUDE_reflection_about_y_eq_neg_x_l1626_162675


namespace NUMINAMATH_CALUDE_soup_donation_theorem_l1626_162616

theorem soup_donation_theorem (shelters cans_per_person total_cans : ℕ) 
  (h1 : shelters = 6)
  (h2 : cans_per_person = 10)
  (h3 : total_cans = 1800) :
  total_cans / (shelters * cans_per_person) = 30 := by
  sorry

end NUMINAMATH_CALUDE_soup_donation_theorem_l1626_162616


namespace NUMINAMATH_CALUDE_skew_lines_theorem_l1626_162651

-- Define the concept of a line in 3D space
structure Line3D where
  -- This is a placeholder definition. In a real scenario, we would need to define
  -- what constitutes a line in 3D space, likely using vectors or points.
  mk :: (dummy : Unit)

-- Define what it means for lines to be skew
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Two lines are skew if they are not coplanar and do not intersect
  sorry

-- Define what it means for lines to be parallel
def are_parallel (l1 l2 : Line3D) : Prop :=
  -- Two lines are parallel if they are coplanar and do not intersect
  sorry

-- Define what it means for lines to intersect
def do_intersect (l1 l2 : Line3D) : Prop :=
  -- Two lines intersect if they share a point
  sorry

-- The main theorem
theorem skew_lines_theorem (a b c : Line3D) 
  (h1 : are_skew a b)
  (h2 : are_parallel a c)
  (h3 : ¬do_intersect b c) :
  are_skew b c := by
  sorry

end NUMINAMATH_CALUDE_skew_lines_theorem_l1626_162651


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_6_l1626_162602

/-- A number is a four-digit number if it's greater than or equal to 1000 and less than 10000 -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

/-- The smallest four-digit number divisible by 6 -/
def smallest_four_digit_div_by_6 : ℕ := 1002

theorem smallest_four_digit_divisible_by_6 :
  (is_four_digit smallest_four_digit_div_by_6) ∧
  (smallest_four_digit_div_by_6 % 6 = 0) ∧
  (∀ n : ℕ, is_four_digit n ∧ n % 6 = 0 → smallest_four_digit_div_by_6 ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_6_l1626_162602


namespace NUMINAMATH_CALUDE_sheridan_cats_l1626_162664

/-- The total number of cats Mrs. Sheridan has after buying more -/
def total_cats (initial : Float) (bought : Float) : Float :=
  initial + bought

/-- Theorem stating that Mrs. Sheridan's total number of cats is 54.0 -/
theorem sheridan_cats : total_cats 11.0 43.0 = 54.0 := by
  sorry

end NUMINAMATH_CALUDE_sheridan_cats_l1626_162664


namespace NUMINAMATH_CALUDE_smallest_tree_height_l1626_162677

/-- Given three trees with specific height relationships, prove the height of the smallest tree -/
theorem smallest_tree_height (tallest middle smallest : ℝ) : 
  tallest = 108 →
  middle = tallest / 2 - 6 →
  smallest = middle / 4 →
  smallest = 12 := by sorry

end NUMINAMATH_CALUDE_smallest_tree_height_l1626_162677


namespace NUMINAMATH_CALUDE_owls_on_fence_l1626_162679

theorem owls_on_fence (initial_owls joining_owls : ℕ) :
  initial_owls = 12 → joining_owls = 7 → initial_owls + joining_owls = 19 := by
  sorry

end NUMINAMATH_CALUDE_owls_on_fence_l1626_162679


namespace NUMINAMATH_CALUDE_pepperoni_count_l1626_162669

/-- Represents a pizza with pepperoni slices -/
structure Pizza :=
  (total_slices : ℕ)

/-- Represents a quarter of a pizza -/
def QuarterPizza := Pizza

theorem pepperoni_count (p : Pizza) (q : QuarterPizza) :
  (p.total_slices = 4 * q.total_slices) →
  (q.total_slices = 10) →
  (p.total_slices = 40) := by
  sorry

end NUMINAMATH_CALUDE_pepperoni_count_l1626_162669


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l1626_162629

/-- An isosceles trapezoid with given base lengths and angle -/
structure IsoscelesTrapezoid where
  larger_base : ℝ
  smaller_base : ℝ
  angle_at_larger_base : ℝ

/-- The area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The area of the specific isosceles trapezoid is 15 -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    larger_base := 8,
    smaller_base := 2,
    angle_at_larger_base := Real.pi / 4  -- 45° in radians
  }
  area t = 15 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l1626_162629


namespace NUMINAMATH_CALUDE_two_cubed_and_three_squared_are_like_terms_l1626_162697

-- Define what it means for two expressions to be like terms
def are_like_terms (a b : ℕ) : Prop :=
  (∃ (x y : ℕ), a = x ∧ b = y) ∨ (∀ (x y : ℕ), a ≠ x ∧ b ≠ y)

-- Theorem statement
theorem two_cubed_and_three_squared_are_like_terms :
  are_like_terms (2^3) (3^2) :=
sorry

end NUMINAMATH_CALUDE_two_cubed_and_three_squared_are_like_terms_l1626_162697


namespace NUMINAMATH_CALUDE_britney_tea_service_l1626_162682

/-- Given a total number of cups and cups per person, calculate the number of people served -/
def people_served (total_cups : ℕ) (cups_per_person : ℕ) : ℕ :=
  total_cups / cups_per_person

/-- Theorem: Britney served 5 people given the conditions -/
theorem britney_tea_service :
  people_served 10 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_britney_tea_service_l1626_162682


namespace NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l1626_162607

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The first term of our specific sequence -/
def a₁ : ℝ := 3

/-- The common difference of our specific sequence -/
def d : ℝ := 5

/-- The 150th term of our specific sequence -/
def a₁₅₀ : ℝ := arithmetic_sequence a₁ d 150

theorem arithmetic_sequence_150th_term : a₁₅₀ = 748 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_150th_term_l1626_162607


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1626_162634

/-- Given a sphere with volume 4√3π, its surface area is 12π -/
theorem sphere_surface_area (V : ℝ) (R : ℝ) (S : ℝ) : 
  V = 4 * Real.sqrt 3 * Real.pi → 
  V = (4 / 3) * Real.pi * R^3 →
  S = 4 * Real.pi * R^2 →
  S = 12 * Real.pi := by
sorry


end NUMINAMATH_CALUDE_sphere_surface_area_l1626_162634


namespace NUMINAMATH_CALUDE_audiobook_disc_content_l1626_162631

theorem audiobook_disc_content (total_time min_per_disc : ℕ) 
  (h1 : total_time = 520) 
  (h2 : min_per_disc = 65) : 
  ∃ (num_discs : ℕ), 
    num_discs > 0 ∧ 
    num_discs * min_per_disc = total_time ∧ 
    ∀ (n : ℕ), n > 0 → n * min_per_disc < total_time → n < num_discs :=
by sorry

end NUMINAMATH_CALUDE_audiobook_disc_content_l1626_162631


namespace NUMINAMATH_CALUDE_ellipse_dot_product_range_l1626_162605

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Definition of the left focus -/
def F₁ : ℝ × ℝ := (-1, 0)

/-- Definition of the right focus -/
def F₂ : ℝ × ℝ := (1, 0)

/-- Definition of a point being on a line through F₂ -/
def is_on_line_through_F₂ (x y : ℝ) : Prop :=
  ∃ k : ℝ, y = k * (x - F₂.1)

/-- The dot product of vectors F₁M and F₁N -/
def F₁M_dot_F₁N (M N : ℝ × ℝ) : ℝ :=
  (M.1 - F₁.1) * (N.1 - F₁.1) + (M.2 - F₁.2) * (N.2 - F₁.2)

/-- The main theorem -/
theorem ellipse_dot_product_range :
  ∀ M N : ℝ × ℝ,
  is_on_ellipse M.1 M.2 →
  is_on_ellipse N.1 N.2 →
  is_on_line_through_F₂ M.1 M.2 →
  is_on_line_through_F₂ N.1 N.2 →
  M ≠ N →
  -1 ≤ F₁M_dot_F₁N M N ∧ F₁M_dot_F₁N M N ≤ 7/2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_dot_product_range_l1626_162605


namespace NUMINAMATH_CALUDE_greatest_common_factor_three_digit_palindromes_l1626_162692

-- Define a three-digit palindrome
def three_digit_palindrome (a b : Nat) : Nat :=
  100 * a + 10 * b + a

-- Define the set of all three-digit palindromes
def all_three_digit_palindromes : Set Nat :=
  {n | ∃ (a b : Nat), a ≤ 9 ∧ b ≤ 9 ∧ n = three_digit_palindrome a b}

-- Theorem statement
theorem greatest_common_factor_three_digit_palindromes :
  ∃ (gcf : Nat), gcf = 11 ∧ 
  (∀ (n : Nat), n ∈ all_three_digit_palindromes → gcf ∣ n) ∧
  (∀ (d : Nat), (∀ (n : Nat), n ∈ all_three_digit_palindromes → d ∣ n) → d ≤ gcf) :=
sorry

end NUMINAMATH_CALUDE_greatest_common_factor_three_digit_palindromes_l1626_162692


namespace NUMINAMATH_CALUDE_sasha_kolya_distance_l1626_162647

/-- Represents the race scenario with three runners -/
structure RaceScenario where
  race_length : ℝ
  sasha_speed : ℝ
  lesha_speed : ℝ
  kolya_speed : ℝ
  sasha_lesha_gap : ℝ
  lesha_kolya_gap : ℝ
  (sasha_speed_pos : sasha_speed > 0)
  (lesha_speed_pos : lesha_speed > 0)
  (kolya_speed_pos : kolya_speed > 0)
  (race_length_pos : race_length > 0)
  (sasha_lesha_gap_pos : sasha_lesha_gap > 0)
  (lesha_kolya_gap_pos : lesha_kolya_gap > 0)
  (sasha_fastest : sasha_speed > lesha_speed ∧ sasha_speed > kolya_speed)
  (lesha_second : lesha_speed > kolya_speed)
  (sasha_lesha_relation : lesha_speed * race_length = sasha_speed * (race_length - sasha_lesha_gap))
  (lesha_kolya_relation : kolya_speed * race_length = lesha_speed * (race_length - lesha_kolya_gap))

/-- Theorem stating the distance between Sasha and Kolya when Sasha finishes -/
theorem sasha_kolya_distance (scenario : RaceScenario) :
  let sasha_finish_time := scenario.race_length / scenario.sasha_speed
  let kolya_distance := scenario.kolya_speed * sasha_finish_time
  scenario.race_length - kolya_distance = 19 := by sorry

end NUMINAMATH_CALUDE_sasha_kolya_distance_l1626_162647


namespace NUMINAMATH_CALUDE_rectangle_area_l1626_162686

theorem rectangle_area (d : ℝ) (w : ℝ) (h : w > 0) : 
  (3 * w)^2 + w^2 = d^2 → 3 * w^2 = 3 * d^2 / 10 :=
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1626_162686


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1626_162668

/-- Two 2D vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Given vectors a and b, prove that if they are perpendicular, then x = 6 -/
theorem perpendicular_vectors_x_value :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  perpendicular a b → x = 6 :=
by
  sorry

#check perpendicular_vectors_x_value

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1626_162668


namespace NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l1626_162698

/-- 
For a regular polygon where each exterior angle measures 30 degrees, 
the sum of the measures of the interior angles is 1800 degrees.
-/
theorem sum_interior_angles_regular_polygon : 
  ∀ (n : ℕ) (exterior_angle : ℝ),
  n > 2 → 
  exterior_angle = 30 →
  n * exterior_angle = 360 →
  (n - 2) * 180 = 1800 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l1626_162698


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1626_162654

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (3 : ℝ) / (2 * Real.sqrt 7 + 3 * Real.sqrt 13) = (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = -6 ∧ B = 7 ∧ C = -9 ∧ D = 13 ∧ E = 89 ∧
    Int.gcd (Int.gcd A C) E = 1 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1626_162654


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l1626_162660

/-- Given a bus that stops for 15 minutes per hour and has a speed of 48 km/hr including stoppages,
    its speed excluding stoppages is 64 km/hr. -/
theorem bus_speed_excluding_stoppages 
  (stop_time : ℝ) 
  (speed_with_stoppages : ℝ) 
  (h1 : stop_time = 15) 
  (h2 : speed_with_stoppages = 48) : 
  speed_with_stoppages * (60 / (60 - stop_time)) = 64 :=
sorry

end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l1626_162660


namespace NUMINAMATH_CALUDE_downstream_speed_l1626_162696

/-- 
Theorem: Given a man's upstream rowing speed and still water speed, 
we can determine his downstream rowing speed.
-/
theorem downstream_speed 
  (upstream_speed : ℝ) 
  (still_water_speed : ℝ) 
  (h1 : upstream_speed = 22) 
  (h2 : still_water_speed = 32) : 
  ∃ downstream_speed : ℝ, 
    downstream_speed = 2 * still_water_speed - upstream_speed ∧ 
    downstream_speed = 42 := by
  sorry

end NUMINAMATH_CALUDE_downstream_speed_l1626_162696


namespace NUMINAMATH_CALUDE_function_value_2007_l1626_162606

def is_multiplicative (f : ℕ+ → ℕ+) : Prop :=
  ∀ x y : ℕ+, f (x + y) = f x * f y

theorem function_value_2007 (f : ℕ+ → ℕ+) 
  (h_mult : is_multiplicative f) (h_base : f 1 = 2) : 
  f 2007 = 2^2007 := by
  sorry

end NUMINAMATH_CALUDE_function_value_2007_l1626_162606


namespace NUMINAMATH_CALUDE_grid_paths_7x3_l1626_162676

theorem grid_paths_7x3 : 
  let m : ℕ := 7  -- width of the grid
  let n : ℕ := 3  -- height of the grid
  (Nat.choose (m + n) n) = 120 := by
sorry

end NUMINAMATH_CALUDE_grid_paths_7x3_l1626_162676


namespace NUMINAMATH_CALUDE_subtract_negative_l1626_162684

theorem subtract_negative : 2 - (-3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_l1626_162684


namespace NUMINAMATH_CALUDE_max_a_value_l1626_162648

/-- A lattice point in an xy-coordinate system -/
def LatticePoint (x y : ℤ) : Prop := True

/-- The line equation y = mx + 3 -/
def LineEquation (m : ℚ) (x y : ℤ) : Prop := y = m * x + 3

/-- Predicate for a line not passing through any lattice point in the given range -/
def NoLatticePointIntersection (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x → x ≤ 150 → LatticePoint x y → ¬LineEquation m x y

/-- The theorem statement -/
theorem max_a_value :
  (∀ m : ℚ, 1/3 < m → m < 50/149 → NoLatticePointIntersection m) ∧
  ¬(∀ m : ℚ, 1/3 < m → m < 50/149 + ε → NoLatticePointIntersection m) :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l1626_162648


namespace NUMINAMATH_CALUDE_variance_of_surviving_trees_l1626_162659

/-- The number of trees transplanted -/
def n : ℕ := 4

/-- The survival probability of each tree -/
def p : ℚ := 4/5

/-- The variance of a binomial distribution -/
def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

/-- 
Theorem: The variance of the number of surviving trees 
in a binomial distribution with n = 4 trials and 
probability of success p = 4/5 is equal to 16/25.
-/
theorem variance_of_surviving_trees : 
  binomial_variance n p = 16/25 := by sorry

end NUMINAMATH_CALUDE_variance_of_surviving_trees_l1626_162659


namespace NUMINAMATH_CALUDE_p_minus_m_equals_2010_l1626_162656

-- Define the set of positive integers
def PositiveInt : Set ℕ := {n : ℕ | n > 0}

-- Define set M
def M : Set ℕ := {x ∈ PositiveInt | 1 ≤ x ∧ x ≤ 2009}

-- Define set P
def P : Set ℕ := {y ∈ PositiveInt | 2 ≤ y ∧ y ≤ 2010}

-- Define the set difference operation
def SetDifference (A B : Set ℕ) : Set ℕ := {x ∈ A | x ∉ B}

-- Theorem statement
theorem p_minus_m_equals_2010 : SetDifference P M = {2010} := by
  sorry

end NUMINAMATH_CALUDE_p_minus_m_equals_2010_l1626_162656


namespace NUMINAMATH_CALUDE_car_speed_problem_l1626_162646

theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ) :
  speed_second_hour = 75 →
  average_speed = 82.5 →
  (speed_second_hour + (average_speed * 2 - speed_second_hour)) / 2 = average_speed →
  average_speed * 2 - speed_second_hour = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1626_162646


namespace NUMINAMATH_CALUDE_collinear_points_solution_l1626_162695

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- The theorem states that if points A(a,2), B(5,1), and C(-4,2a) are collinear, 
    then a = 5 ± √21 -/
theorem collinear_points_solution (a : ℝ) :
  collinear a 2 5 1 (-4) (2*a) → a = 5 + Real.sqrt 21 ∨ a = 5 - Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_collinear_points_solution_l1626_162695


namespace NUMINAMATH_CALUDE_saltwater_solution_l1626_162610

/-- Represents the saltwater tank problem --/
def saltwater_problem (x : ℝ) : Prop :=
  let original_salt := 0.2 * x
  let volume_after_evaporation := 0.75 * x
  let salt_after_addition := original_salt + 14
  let final_volume := salt_after_addition / (1/3)
  let water_added := final_volume - volume_after_evaporation
  (x = 104.99999999999997) ∧ (water_added = 26.25)

/-- Theorem stating the solution to the saltwater problem --/
theorem saltwater_solution :
  ∃ (x : ℝ), saltwater_problem x :=
sorry

end NUMINAMATH_CALUDE_saltwater_solution_l1626_162610


namespace NUMINAMATH_CALUDE_original_price_correct_l1626_162623

/-- The original price of water bottles that satisfies the given conditions --/
def original_price : ℝ :=
  let number_of_bottles : ℕ := 60
  let reduced_price : ℝ := 1.85
  let shortfall : ℝ := 9
  2

theorem original_price_correct :
  let number_of_bottles : ℕ := 60
  let reduced_price : ℝ := 1.85
  let shortfall : ℝ := 9
  (number_of_bottles : ℝ) * original_price = 
    (number_of_bottles : ℝ) * reduced_price + shortfall :=
by
  sorry

#eval original_price

end NUMINAMATH_CALUDE_original_price_correct_l1626_162623


namespace NUMINAMATH_CALUDE_min_value_parallel_vectors_l1626_162624

/-- Given two vectors a and b, where a is parallel to b, 
    prove that the minimum value of 3/x + 2/y is 8 -/
theorem min_value_parallel_vectors (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (x, y - 1)
  (∃ (k : ℝ), a.1 * b.2 = k * a.2 * b.1) →  -- parallelism condition
  (3 / x + 2 / y) ≥ 8 ∧ 
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 / x₀ + 2 / y₀ = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_value_parallel_vectors_l1626_162624


namespace NUMINAMATH_CALUDE_max_xyz_value_l1626_162678

theorem max_xyz_value (x y z : ℝ) 
  (eq1 : x + x*y + x*y*z = 1)
  (eq2 : y + y*z + x*y*z = 2)
  (eq3 : z + x*z + x*y*z = 4) :
  x*y*z ≤ (5 + Real.sqrt 17) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_xyz_value_l1626_162678


namespace NUMINAMATH_CALUDE_seating_arrangements_l1626_162643

def number_of_seats : ℕ := 9
def number_of_families : ℕ := 3
def members_per_family : ℕ := 3

theorem seating_arrangements :
  (number_of_seats = number_of_families * members_per_family) →
  (number_of_different_seating_arrangements : ℕ) = (Nat.factorial number_of_families)^(number_of_families + 1) :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l1626_162643


namespace NUMINAMATH_CALUDE_m_plus_n_equals_plus_minus_one_l1626_162635

theorem m_plus_n_equals_plus_minus_one (m n : ℤ) 
  (hm : |m| = 3) 
  (hn : |n| = 2) 
  (hmn : m * n < 0) : 
  m + n = 1 ∨ m + n = -1 := by
sorry

end NUMINAMATH_CALUDE_m_plus_n_equals_plus_minus_one_l1626_162635


namespace NUMINAMATH_CALUDE_fermat_numbers_not_cubes_l1626_162642

theorem fermat_numbers_not_cubes : ∀ (n : ℕ), ¬ ∃ (k : ℤ), 2^(2^n) + 1 = k^3 := by
  sorry

end NUMINAMATH_CALUDE_fermat_numbers_not_cubes_l1626_162642


namespace NUMINAMATH_CALUDE_cousins_arrangement_l1626_162690

/-- The number of ways to arrange cousins in rooms -/
def arrange_cousins (n : ℕ) (m : ℕ) : ℕ :=
  -- n is the number of cousins
  -- m is the number of rooms
  sorry

/-- Theorem: Arranging 5 cousins in 4 rooms with at least one empty room -/
theorem cousins_arrangement :
  arrange_cousins 5 4 = 56 :=
by sorry

end NUMINAMATH_CALUDE_cousins_arrangement_l1626_162690


namespace NUMINAMATH_CALUDE_circle_configuration_implies_zero_area_l1626_162680

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line structure
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def CircleTangentToLine (c : Circle) (l : Line) : Prop :=
  sorry

def CirclesExternallyTangent (c1 c2 : Circle) : Prop :=
  sorry

def PointBetween (p1 p2 p3 : ℝ × ℝ) : Prop :=
  sorry

def TriangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  sorry

theorem circle_configuration_implies_zero_area 
  (P Q R : Circle)
  (l : Line)
  (P' Q' R' : ℝ × ℝ)
  (h1 : P.radius = 2)
  (h2 : Q.radius = 3)
  (h3 : R.radius = 4)
  (h4 : CircleTangentToLine P l)
  (h5 : CircleTangentToLine Q l)
  (h6 : CircleTangentToLine R l)
  (h7 : CirclesExternallyTangent Q P)
  (h8 : CirclesExternallyTangent Q R)
  (h9 : PointBetween P' Q' R')
  (h10 : P' = (P.center.1, l.a * P.center.1 + l.b))
  (h11 : Q' = (Q.center.1, l.a * Q.center.1 + l.b))
  (h12 : R' = (R.center.1, l.a * R.center.1 + l.b)) :
  TriangleArea P.center Q.center R.center = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_configuration_implies_zero_area_l1626_162680


namespace NUMINAMATH_CALUDE_tree_planting_problem_l1626_162603

theorem tree_planting_problem (x : ℝ) : 
  (∀ y : ℝ, y = x + 5 → 60 / y = 45 / x) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_problem_l1626_162603


namespace NUMINAMATH_CALUDE_last_non_zero_digit_30_factorial_l1626_162667

/-- The last non-zero digit of a natural number -/
def lastNonZeroDigit (n : ℕ) : ℕ :=
  n % 10 -- Definition, not from solution steps

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem last_non_zero_digit_30_factorial :
  lastNonZeroDigit (factorial 30) = 8 := by
  sorry

end NUMINAMATH_CALUDE_last_non_zero_digit_30_factorial_l1626_162667


namespace NUMINAMATH_CALUDE_max_a_condition_1_range_a_condition_2_l1626_162613

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

-- Theorem for the first part of the problem
theorem max_a_condition_1 :
  (∀ a : ℝ, (∀ x : ℝ, g x ≤ 5 → f a x ≤ 6) → a ≤ 1) ∧
  (∃ a : ℝ, a = 1 ∧ ∀ x : ℝ, g x ≤ 5 → f a x ≤ 6) :=
sorry

-- Theorem for the second part of the problem
theorem range_a_condition_2 :
  ∀ a : ℝ, (∀ x : ℝ, f a x + g x ≥ 3) → a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_max_a_condition_1_range_a_condition_2_l1626_162613


namespace NUMINAMATH_CALUDE_inverse_variation_solution_l1626_162604

/-- Inverse variation constant -/
def k : ℝ := 9

/-- The relation between x and y -/
def inverse_variation (x y : ℝ) : Prop := x = k / (y ^ 2)

theorem inverse_variation_solution :
  ∀ x y : ℝ,
  inverse_variation x y →
  inverse_variation 1 3 →
  inverse_variation 0.1111111111111111 y →
  y = 9 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_solution_l1626_162604


namespace NUMINAMATH_CALUDE_remainder_3_800_mod_17_l1626_162630

theorem remainder_3_800_mod_17 : 3^800 % 17 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_800_mod_17_l1626_162630


namespace NUMINAMATH_CALUDE_win_sector_area_l1626_162662

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 6) (h2 : p = 1/3) :
  p * (π * r^2) = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l1626_162662


namespace NUMINAMATH_CALUDE_sugar_for_recipe_l1626_162626

/-- The amount of sugar required for a cake recipe -/
theorem sugar_for_recipe (sugar_frosting sugar_cake : ℚ) 
  (h1 : sugar_frosting = 6/10)
  (h2 : sugar_cake = 2/10) :
  sugar_frosting + sugar_cake = 8/10 := by
  sorry

end NUMINAMATH_CALUDE_sugar_for_recipe_l1626_162626


namespace NUMINAMATH_CALUDE_sin_cube_identity_l1626_162621

theorem sin_cube_identity (θ : Real) : 
  Real.sin θ ^ 3 = (-1/4) * Real.sin (3 * θ) + (3/4) * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_sin_cube_identity_l1626_162621


namespace NUMINAMATH_CALUDE_log_base_5_inequality_l1626_162636

theorem log_base_5_inequality (x : ℝ) (h1 : 0 < x) (h2 : Real.log x / Real.log 5 < 1) : 1 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_log_base_5_inequality_l1626_162636


namespace NUMINAMATH_CALUDE_total_students_accommodated_l1626_162619

/-- Represents a bus with its seating configuration and broken seats -/
structure Bus where
  columns : Nat
  rows : Nat
  broken_seats : Nat

/-- Calculates the number of usable seats in a bus -/
def usable_seats (bus : Bus) : Nat :=
  bus.columns * bus.rows - bus.broken_seats

/-- The list of buses with their configurations -/
def buses : List Bus := [
  ⟨4, 10, 2⟩,
  ⟨5, 8, 4⟩,
  ⟨3, 12, 3⟩,
  ⟨4, 12, 1⟩,
  ⟨6, 8, 5⟩,
  ⟨5, 10, 2⟩
]

/-- Theorem: The total number of students that can be accommodated is 245 -/
theorem total_students_accommodated : (buses.map usable_seats).sum = 245 := by
  sorry


end NUMINAMATH_CALUDE_total_students_accommodated_l1626_162619


namespace NUMINAMATH_CALUDE_complex_arithmetic_simplification_l1626_162641

theorem complex_arithmetic_simplification :
  57.6 * (8 / 5) + 28.8 * (184 / 5) - 14.4 * 80 + 12.5 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_simplification_l1626_162641


namespace NUMINAMATH_CALUDE_even_shifted_implies_equality_l1626_162655

def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 1) = f (1 - x)

theorem even_shifted_implies_equality (f : ℝ → ℝ) 
  (h : is_even_shifted f) : f 0 = f 2 := by
  sorry

end NUMINAMATH_CALUDE_even_shifted_implies_equality_l1626_162655


namespace NUMINAMATH_CALUDE_inequality_proof_l1626_162672

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  x * (x - z)^2 + y * (y - z)^2 ≥ (x - z) * (y - z) * (x + y - z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1626_162672


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1626_162608

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (f x ^ 2 + f y) = x * f x + y

/-- The main theorem stating that any function satisfying the equation
    must be either the identity function or the negation function -/
theorem functional_equation_solution (f : ℝ → ℝ) 
    (h : SatisfiesEquation f) : 
    (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1626_162608


namespace NUMINAMATH_CALUDE_polynomial_ratio_l1626_162663

-- Define the polynomial coefficients
variable (a₀ a₁ a₂ a₃ a₄ a₅ : ℚ)

-- Define the main equation
def main_equation (x : ℚ) : Prop :=
  (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5

-- State the theorem
theorem polynomial_ratio :
  (∀ x, main_equation a₀ a₁ a₂ a₃ a₄ a₅ x) →
  (a₀ + a₂ + a₄) / (a₁ + a₃) = -61 / 60 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_ratio_l1626_162663


namespace NUMINAMATH_CALUDE_correct_statements_l1626_162633

/-- Represents a mathematical statement about proofs and principles -/
inductive MathStatement
  | InductionInfinite
  | ProofStructure
  | TheoremProof
  | AxiomPostulate
  | NoUnprovenConjectures

/-- Determines if a given mathematical statement is correct -/
def is_correct (statement : MathStatement) : Prop :=
  match statement with
  | MathStatement.InductionInfinite => False
  | MathStatement.ProofStructure => True
  | MathStatement.TheoremProof => True
  | MathStatement.AxiomPostulate => True
  | MathStatement.NoUnprovenConjectures => True

/-- Theorem stating that statement A is incorrect while B, C, D, and E are correct -/
theorem correct_statements :
  ¬(is_correct MathStatement.InductionInfinite) ∧
  (is_correct MathStatement.ProofStructure) ∧
  (is_correct MathStatement.TheoremProof) ∧
  (is_correct MathStatement.AxiomPostulate) ∧
  (is_correct MathStatement.NoUnprovenConjectures) :=
sorry

end NUMINAMATH_CALUDE_correct_statements_l1626_162633


namespace NUMINAMATH_CALUDE_fraction_addition_l1626_162699

theorem fraction_addition : (3 / 4) / (5 / 8) + 1 / 2 = 17 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1626_162699


namespace NUMINAMATH_CALUDE_reorganize_32_city_graph_l1626_162649

/-- A graph with n vertices, where each pair of vertices is connected by a directed edge. -/
structure DirectedGraph (n : ℕ) where
  edges : Fin n → Fin n → Bool

/-- The number of steps required to reorganize a directed graph with n vertices
    such that the resulting graph has no cycles. -/
def reorganization_steps (n : ℕ) : ℕ :=
  if n ≤ 2 then 0 else 2^(n-2) * (2^n - n - 1)

/-- Theorem stating that for a graph with 32 vertices, it's possible to reorganize
    the edge directions in at most 208 steps to eliminate all cycles. -/
theorem reorganize_32_city_graph :
  reorganization_steps 32 ≤ 208 :=
sorry

end NUMINAMATH_CALUDE_reorganize_32_city_graph_l1626_162649


namespace NUMINAMATH_CALUDE_inscribed_triangle_angle_l1626_162652

theorem inscribed_triangle_angle (x : ℝ) : 
  let arc_DE := x + 90
  let arc_EF := 2*x + 15
  let arc_FD := 3*x - 30
  -- Sum of arcs is 360°
  arc_DE + arc_EF + arc_FD = 360 →
  -- Triangle inscribed in circle
  -- Interior angles are half the corresponding arc measures
  ∃ (angle : ℝ), (angle = arc_EF / 2 ∨ angle = arc_FD / 2 ∨ angle = arc_DE / 2) ∧ 
  (angle ≥ 68.5 ∧ angle ≤ 69.5) :=
by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_angle_l1626_162652


namespace NUMINAMATH_CALUDE_smallest_x_value_l1626_162618

theorem smallest_x_value : 
  ∃ (x : ℝ), x > 1 ∧ 
  ((5*x - 20) / (4*x - 5))^2 + (5*x - 20) / (4*x - 5) = 20 ∧
  (∀ (y : ℝ), y > 1 ∧ 
   ((5*y - 20) / (4*y - 5))^2 + (5*y - 20) / (4*y - 5) = 20 → 
   x ≤ y) ∧
  x = 9/5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l1626_162618


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1626_162644

theorem quadratic_inequality_equivalence (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x - 4*a ≥ 0) ↔ (-16 ≤ a ∧ a ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l1626_162644


namespace NUMINAMATH_CALUDE_sqrt_trig_identity_l1626_162622

theorem sqrt_trig_identity : 
  Real.sqrt (2 - Real.sin 2 ^ 2 + Real.cos 4) = -Real.sqrt 3 * Real.cos 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_trig_identity_l1626_162622


namespace NUMINAMATH_CALUDE_locomotive_whistle_distance_l1626_162650

/-- The speed of the locomotive in meters per second -/
def locomotive_speed : ℝ := 20

/-- The speed of sound in meters per second -/
def sound_speed : ℝ := 340

/-- The time difference between hearing the whistle and the train's arrival in seconds -/
def time_difference : ℝ := 4

/-- The distance of the locomotive when it started whistling in meters -/
def whistle_distance : ℝ := 85

theorem locomotive_whistle_distance :
  (whistle_distance / locomotive_speed) - time_difference = whistle_distance / sound_speed :=
by sorry

end NUMINAMATH_CALUDE_locomotive_whistle_distance_l1626_162650


namespace NUMINAMATH_CALUDE_persistent_iff_two_l1626_162609

/-- A number T is persistent if for any a, b, c, d ∈ ℝ \ {0, 1} satisfying
    a + b + c + d = T and 1/a + 1/b + 1/c + 1/d = T,
    we also have 1/(1-a) + 1/(1-b) + 1/(1-c) + 1/(1-d) = T -/
def isPersistent (T : ℝ) : Prop :=
  ∀ a b c d : ℝ, a ≠ 0 ∧ a ≠ 1 ∧ b ≠ 0 ∧ b ≠ 1 ∧ c ≠ 0 ∧ c ≠ 1 ∧ d ≠ 0 ∧ d ≠ 1 →
    a + b + c + d = T →
    1/a + 1/b + 1/c + 1/d = T →
    1/(1-a) + 1/(1-b) + 1/(1-c) + 1/(1-d) = T

/-- The only persistent number is 2 -/
theorem persistent_iff_two : ∀ T : ℝ, isPersistent T ↔ T = 2 := by
  sorry

end NUMINAMATH_CALUDE_persistent_iff_two_l1626_162609


namespace NUMINAMATH_CALUDE_yeongsoo_initial_amount_l1626_162614

/-- Given the initial amounts of money for Yeongsoo, Hyogeun, and Woong,
    this function returns their final amounts after the transactions. -/
def final_amounts (y h w : ℕ) : ℕ × ℕ × ℕ :=
  (y - 200 + 1000, h + 200 - 500, w + 500 - 1000)

/-- Theorem stating that Yeongsoo's initial amount was 1200 won -/
theorem yeongsoo_initial_amount :
  ∃ (h w : ℕ), final_amounts 1200 h w = (2000, 2000, 2000) :=
sorry

end NUMINAMATH_CALUDE_yeongsoo_initial_amount_l1626_162614
