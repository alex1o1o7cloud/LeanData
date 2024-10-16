import Mathlib

namespace NUMINAMATH_CALUDE_land_division_theorem_l3975_397564

/-- Represents a rectangular plot of land -/
structure Plot where
  length : ℝ
  width : ℝ

/-- Represents a division of land into three plots -/
structure Division where
  plot1 : Plot
  plot2 : Plot
  plot3 : Plot

def total_area (p : Plot) : ℝ := p.length * p.width

def is_valid_division (d : Division) (total_length : ℝ) (total_width : ℝ) : Prop :=
  d.plot1.length + d.plot2.length + d.plot3.length = total_length ∧
  d.plot1.width = total_width ∧ d.plot2.width = total_width ∧ d.plot3.width = total_width ∧
  total_area d.plot1 = 300 ∧ total_area d.plot2 = 300 ∧ total_area d.plot3 = 300

def fence_length (d : Division) : ℝ :=
  d.plot1.length + d.plot2.length

def count_valid_divisions (total_length : ℝ) (total_width : ℝ) : ℕ :=
  sorry

theorem land_division_theorem (total_length : ℝ) (total_width : ℝ) 
  (h1 : total_length = 25) (h2 : total_width = 36) :
  (count_valid_divisions total_length total_width = 4) ∧
  (∃ d : Division, is_valid_division d total_length total_width ∧ 
    (∀ d' : Division, is_valid_division d' total_length total_width → 
      fence_length d ≤ fence_length d') ∧
    fence_length d = 49) := by
  sorry

end NUMINAMATH_CALUDE_land_division_theorem_l3975_397564


namespace NUMINAMATH_CALUDE_circle_tangency_l3975_397503

theorem circle_tangency (m : ℝ) : 
  (∃ x y : ℝ, (x - m)^2 + (y + 2)^2 = 9 ∧ (x + 1)^2 + (y - m)^2 = 4) →
  (∃ x y : ℝ, (x - m)^2 + (y + 2)^2 = 9) →
  (∃ x y : ℝ, (x + 1)^2 + (y - m)^2 = 4) →
  (m = -2 ∨ m = -1) := by
sorry

end NUMINAMATH_CALUDE_circle_tangency_l3975_397503


namespace NUMINAMATH_CALUDE_angle_set_inclusion_l3975_397527

def M : Set ℝ := { x | 0 < x ∧ x ≤ 90 }
def N : Set ℝ := { x | 0 < x ∧ x < 90 }
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 90 }

theorem angle_set_inclusion : N ⊆ M ∧ M ⊆ P := by sorry

end NUMINAMATH_CALUDE_angle_set_inclusion_l3975_397527


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_diff_l3975_397543

theorem crazy_silly_school_series_diff (total_books total_movies : ℕ) 
  (h1 : total_books = 20) 
  (h2 : total_movies = 12) : 
  total_books - total_movies = 8 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_diff_l3975_397543


namespace NUMINAMATH_CALUDE_max_omega_for_monotonic_sin_l3975_397568

theorem max_omega_for_monotonic_sin (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = Real.sin (ω * x)) →
  ω > 0 →
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π / 3 → f x < f y) →
  ω ≤ 3 / 2 ∧ ∀ ω' > 3 / 2, ∃ x y, 0 ≤ x ∧ x < y ∧ y ≤ π / 3 ∧ f x ≥ f y :=
by sorry

end NUMINAMATH_CALUDE_max_omega_for_monotonic_sin_l3975_397568


namespace NUMINAMATH_CALUDE_perfect_square_count_l3975_397565

theorem perfect_square_count : 
  ∃! (count : ℕ), ∃ (S : Finset ℕ), 
    (Finset.card S = count) ∧ 
    (∀ n, n ∈ S ↔ ∃ x : ℤ, (4:ℤ)^n - 15 = x^2) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_count_l3975_397565


namespace NUMINAMATH_CALUDE_rational_function_zero_l3975_397549

-- Define the numerator and denominator of the rational function
def numerator (x : ℝ) : ℝ := x^2 - x - 6
def denominator (x : ℝ) : ℝ := 5*x - 15

-- Define the domain of the function (all real numbers except 3)
def domain (x : ℝ) : Prop := x ≠ 3

-- State the theorem
theorem rational_function_zero (x : ℝ) (h : domain x) : 
  (numerator x) / (denominator x) = 0 ↔ x = -2 :=
sorry

end NUMINAMATH_CALUDE_rational_function_zero_l3975_397549


namespace NUMINAMATH_CALUDE_unique_number_property_l3975_397553

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by sorry

end NUMINAMATH_CALUDE_unique_number_property_l3975_397553


namespace NUMINAMATH_CALUDE_identify_scientists_l3975_397581

/-- Represents the type of scientist: chemist or alchemist -/
inductive ScientistType
| Chemist
| Alchemist

/-- Represents a scientist at the conference -/
structure Scientist where
  id : Nat
  type : ScientistType

/-- Represents the conference of scientists -/
structure Conference where
  scientists : List Scientist
  num_chemists : Nat
  num_alchemists : Nat
  chemists_outnumber_alchemists : num_chemists > num_alchemists

/-- Represents a question asked by the mathematician -/
def Question := Scientist → Scientist → ScientistType

/-- The main theorem to be proved -/
theorem identify_scientists (conf : Conference) :
  ∃ (questions : List Question), questions.length ≤ 2 * conf.scientists.length - 2 ∧
  (∀ s : Scientist, s ∈ conf.scientists → 
    ∃ (determined_type : ScientistType), determined_type = s.type) :=
sorry

end NUMINAMATH_CALUDE_identify_scientists_l3975_397581


namespace NUMINAMATH_CALUDE_function_identity_l3975_397593

theorem function_identity (f : ℝ → ℝ) 
  (h1 : ∀ x, |f x + Real.cos x ^ 2| ≤ 3/4)
  (h2 : ∀ x, |f x - Real.sin x ^ 2| ≤ 1/4) :
  ∀ x, f x = 1/2 - Real.sin x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l3975_397593


namespace NUMINAMATH_CALUDE_sqrt_50_plus_sqrt_32_l3975_397588

theorem sqrt_50_plus_sqrt_32 : Real.sqrt 50 + Real.sqrt 32 = 9 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_plus_sqrt_32_l3975_397588


namespace NUMINAMATH_CALUDE_square_root_equation_implies_product_l3975_397524

theorem square_root_equation_implies_product (x : ℝ) :
  Real.sqrt (8 + x) + Real.sqrt (25 - x^2) = 9 →
  (8 + x) * (25 - x^2) = 576 := by
sorry

end NUMINAMATH_CALUDE_square_root_equation_implies_product_l3975_397524


namespace NUMINAMATH_CALUDE_negative_integer_square_plus_self_l3975_397592

theorem negative_integer_square_plus_self (N : ℤ) : 
  N < 0 → N^2 + N = 12 → N = -4 := by sorry

end NUMINAMATH_CALUDE_negative_integer_square_plus_self_l3975_397592


namespace NUMINAMATH_CALUDE_amy_biking_week_l3975_397519

def total_miles_biked (monday_miles : ℕ) : ℕ :=
  let tuesday_miles := 2 * monday_miles - 3
  let wednesday_miles := tuesday_miles + 2
  let thursday_miles := wednesday_miles + 2
  let friday_miles := thursday_miles + 2
  let saturday_miles := friday_miles + 2
  let sunday_miles := saturday_miles + 2
  monday_miles + tuesday_miles + wednesday_miles + thursday_miles + friday_miles + saturday_miles + sunday_miles

theorem amy_biking_week (monday_miles : ℕ) (h : monday_miles = 12) : 
  total_miles_biked monday_miles = 168 := by
  sorry

end NUMINAMATH_CALUDE_amy_biking_week_l3975_397519


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l3975_397584

theorem circumscribed_sphere_surface_area (a : ℝ) (h : a = 2 * Real.sqrt 3 / 3) :
  let R := Real.sqrt 3 * a / 2
  4 * Real.pi * R^2 = 4 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l3975_397584


namespace NUMINAMATH_CALUDE_barbie_earrings_problem_l3975_397512

theorem barbie_earrings_problem (barbie_earrings : ℕ) 
  (h1 : barbie_earrings % 2 = 0)  -- Ensures barbie_earrings is even
  (h2 : ∃ (alissa_given : ℕ), alissa_given = barbie_earrings / 2)
  (h3 : ∃ (alissa_total : ℕ), alissa_total = 3 * (barbie_earrings / 2))
  (h4 : 3 * (barbie_earrings / 2) = 36) :
  barbie_earrings / 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_barbie_earrings_problem_l3975_397512


namespace NUMINAMATH_CALUDE_cube_root_sqrt_64_l3975_397500

theorem cube_root_sqrt_64 : 
  {x : ℝ | x^3 = Real.sqrt 64} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_cube_root_sqrt_64_l3975_397500


namespace NUMINAMATH_CALUDE_computer_table_cost_price_l3975_397547

-- Define the markup percentage
def markup : ℚ := 24 / 100

-- Define the selling price
def selling_price : ℚ := 8215

-- Define the cost price calculation function
def cost_price (sp : ℚ) (m : ℚ) : ℚ := sp / (1 + m)

-- Theorem statement
theorem computer_table_cost_price : 
  cost_price selling_price markup = 6625 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_cost_price_l3975_397547


namespace NUMINAMATH_CALUDE_circle_tangent_chord_relation_l3975_397530

/-- Given a circle O with radius r, prove the relationship between x and y -/
theorem circle_tangent_chord_relation (r : ℝ) (x y : ℝ) : y^2 = x^3 / (2*r - x) :=
  sorry

end NUMINAMATH_CALUDE_circle_tangent_chord_relation_l3975_397530


namespace NUMINAMATH_CALUDE_first_level_spots_l3975_397548

/-- Represents the number of open parking spots on each level of a 4-story parking area -/
structure ParkingArea where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The parking area satisfies the given conditions -/
def validParkingArea (p : ParkingArea) : Prop :=
  p.second = p.first + 7 ∧
  p.third = p.second + 6 ∧
  p.fourth = 14 ∧
  p.first + p.second + p.third + p.fourth = 46

theorem first_level_spots (p : ParkingArea) (h : validParkingArea p) : p.first = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_level_spots_l3975_397548


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3975_397552

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

/-- The 9th term is the arithmetic mean of 1 and 3 -/
def ninth_term_is_mean (b : ℕ → ℝ) : Prop :=
  b 9 = (1 + 3) / 2

theorem geometric_sequence_product (b : ℕ → ℝ) 
  (h1 : geometric_sequence b) 
  (h2 : ninth_term_is_mean b) : 
  b 2 * b 16 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3975_397552


namespace NUMINAMATH_CALUDE_function_value_at_three_l3975_397580

/-- Given a continuous and differentiable function f satisfying
    f(2x + 1) = 2f(x) + 1 for all real x, and f(0) = 2,
    prove that f(3) = 11. -/
theorem function_value_at_three
  (f : ℝ → ℝ)
  (hcont : Continuous f)
  (hdiff : Differentiable ℝ f)
  (hfunc : ∀ x : ℝ, f (2 * x + 1) = 2 * f x + 1)
  (hf0 : f 0 = 2) :
  f 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_three_l3975_397580


namespace NUMINAMATH_CALUDE_minimum_matches_theorem_l3975_397528

/-- Represents the number of points for each match result -/
structure PointSystem where
  win : Nat
  draw : Nat
  loss : Nat

/-- Represents the state of a team in the competition -/
structure TeamState where
  gamesPlayed : Nat
  points : Nat

/-- Represents the requirements for the team -/
structure TeamRequirement where
  targetPoints : Nat
  minWinsNeeded : Nat

def minimumTotalMatches (initialState : TeamState) (pointSystem : PointSystem) (requirement : TeamRequirement) : Nat :=
  initialState.gamesPlayed + requirement.minWinsNeeded +
    ((requirement.targetPoints - initialState.points - requirement.minWinsNeeded * pointSystem.win + pointSystem.draw - 1) / pointSystem.draw)

theorem minimum_matches_theorem (initialState : TeamState) (pointSystem : PointSystem) (requirement : TeamRequirement) :
  initialState.gamesPlayed = 5 ∧
  initialState.points = 14 ∧
  pointSystem.win = 3 ∧
  pointSystem.draw = 1 ∧
  pointSystem.loss = 0 ∧
  requirement.targetPoints = 40 ∧
  requirement.minWinsNeeded = 6 →
  minimumTotalMatches initialState pointSystem requirement = 13 := by
  sorry

end NUMINAMATH_CALUDE_minimum_matches_theorem_l3975_397528


namespace NUMINAMATH_CALUDE_pages_read_l3975_397535

/-- Given that Tom read a certain number of chapters in a book with a fixed number of pages per chapter,
    prove that the total number of pages read is equal to the product of chapters and pages per chapter. -/
theorem pages_read (chapters : ℕ) (pages_per_chapter : ℕ) (h1 : chapters = 20) (h2 : pages_per_chapter = 15) :
  chapters * pages_per_chapter = 300 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_l3975_397535


namespace NUMINAMATH_CALUDE_pinecone_problem_l3975_397533

theorem pinecone_problem :
  ∃! n : ℕ, n < 350 ∧ 
  2 ∣ n ∧ 3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n ∧ 9 ∣ n ∧ 
  ¬(7 ∣ n) ∧ ¬(8 ∣ n) ∧
  n = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_pinecone_problem_l3975_397533


namespace NUMINAMATH_CALUDE_parabola_line_intersection_dot_product_l3975_397577

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := y = (2/3)*(x + 2)

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the intersection points
def M : ℝ × ℝ := (1, 2)
def N : ℝ × ℝ := (4, 4)

-- Define vectors FM and FN
def FM : ℝ × ℝ := (M.1 - focus.1, M.2 - focus.2)
def FN : ℝ × ℝ := (N.1 - focus.1, N.2 - focus.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem parabola_line_intersection_dot_product :
  parabola M.1 M.2 ∧ parabola N.1 N.2 ∧
  line M.1 M.2 ∧ line N.1 N.2 →
  dot_product FM FN = 8 :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_dot_product_l3975_397577


namespace NUMINAMATH_CALUDE_problem_solution_l3975_397587

theorem problem_solution : (-15) / (1/3 - 3 - 3/2) * 6 = 108/5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3975_397587


namespace NUMINAMATH_CALUDE_equation_solution_exists_l3975_397582

theorem equation_solution_exists : ∃ x : ℝ, (0.75 : ℝ) ^ x + 2 = 8 := by sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l3975_397582


namespace NUMINAMATH_CALUDE_initial_student_count_l3975_397511

theorem initial_student_count (initial_avg : ℝ) (new_avg : ℝ) (new_student_weight : ℝ) :
  initial_avg = 15 →
  new_avg = 14.9 →
  new_student_weight = 13 →
  ∃ n : ℕ, n * initial_avg + new_student_weight = (n + 1) * new_avg ∧ n = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_student_count_l3975_397511


namespace NUMINAMATH_CALUDE_sphere_cross_section_distance_l3975_397522

theorem sphere_cross_section_distance (V : ℝ) (A : ℝ) (r : ℝ) (r_cross : ℝ) (d : ℝ) :
  V = 4 * Real.sqrt 3 * Real.pi →
  (4 / 3) * Real.pi * r^3 = V →
  A = Real.pi →
  Real.pi * r_cross^2 = A →
  d^2 + r_cross^2 = r^2 →
  d = Real.sqrt 2 := by
  sorry

#check sphere_cross_section_distance

end NUMINAMATH_CALUDE_sphere_cross_section_distance_l3975_397522


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3975_397574

theorem complex_fraction_simplification :
  (I : ℂ) / (Real.sqrt 7 + 3 * I) = (3 : ℂ) / 16 + (Real.sqrt 7 / 16) * I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3975_397574


namespace NUMINAMATH_CALUDE_cell_population_after_9_days_l3975_397502

/-- Calculates the number of cells after a given number of days, 
    given an initial population and a tripling rate every 3 days -/
def cell_population (initial_cells : ℕ) (days : ℕ) : ℕ :=
  initial_cells * (3 ^ (days / 3))

/-- Theorem stating that the cell population after 9 days is 36, 
    given an initial population of 4 cells -/
theorem cell_population_after_9_days :
  cell_population 4 9 = 36 := by
  sorry

#eval cell_population 4 9

end NUMINAMATH_CALUDE_cell_population_after_9_days_l3975_397502


namespace NUMINAMATH_CALUDE_largest_d_for_negative_three_in_range_l3975_397518

/-- The function f(x) = x^2 + 4x + d -/
def f (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

/-- Proposition: The largest value of d such that -3 is in the range of f(x) = x^2 + 4x + d is 1 -/
theorem largest_d_for_negative_three_in_range :
  (∃ (d : ℝ), ∀ (e : ℝ), (∃ (x : ℝ), f d x = -3) → e ≤ d) ∧
  (∃ (x : ℝ), f 1 x = -3) :=
sorry

end NUMINAMATH_CALUDE_largest_d_for_negative_three_in_range_l3975_397518


namespace NUMINAMATH_CALUDE_geometric_sequence_3_pow_l3975_397538

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 1 / a 0

theorem geometric_sequence_3_pow (a : ℕ → ℝ) :
  (∀ n, a n = 3^n) →
  geometric_sequence a ∧
  (∀ n, a (n + 1) > a n) ∧
  a 5^2 = a 10 ∧
  ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_3_pow_l3975_397538


namespace NUMINAMATH_CALUDE_coin_toss_and_die_roll_probability_l3975_397523

/-- The probability of getting exactly three heads and one tail when tossing four coins -/
def prob_three_heads_one_tail : ℚ := 1 / 4

/-- The probability of rolling a number greater than 4 on a six-sided die -/
def prob_die_greater_than_four : ℚ := 1 / 3

/-- The number of coins tossed -/
def num_coins : ℕ := 4

/-- The number of sides on the die -/
def num_die_sides : ℕ := 6

theorem coin_toss_and_die_roll_probability :
  prob_three_heads_one_tail * prob_die_greater_than_four = 1 / 12 :=
sorry

end NUMINAMATH_CALUDE_coin_toss_and_die_roll_probability_l3975_397523


namespace NUMINAMATH_CALUDE_custom_mul_result_l3975_397507

/-- Custom multiplication operation for rational numbers -/
noncomputable def custom_mul (a b : ℚ) (x y : ℚ) : ℚ := a * x + b * y

theorem custom_mul_result 
  (a b : ℚ) 
  (h1 : custom_mul a b 1 2 = 1) 
  (h2 : custom_mul a b (-3) 3 = 6) :
  custom_mul a b 2 (-5) = -7 := by
  sorry

end NUMINAMATH_CALUDE_custom_mul_result_l3975_397507


namespace NUMINAMATH_CALUDE_g_range_l3975_397544

noncomputable def g (x : ℝ) : ℝ := 3 * (x - 2)

theorem g_range :
  Set.range g = {y : ℝ | y ≠ -21} := by sorry

end NUMINAMATH_CALUDE_g_range_l3975_397544


namespace NUMINAMATH_CALUDE_solution_correctness_l3975_397595

theorem solution_correctness : 
  (∃ x y : ℚ, 2 * x - y = 3 ∧ 3 * x + 2 * y = 8 ∧ x = 2 ∧ y = 1) ∧
  (∃ x y : ℚ, 2 * x + y = 2 ∧ 8 * x + 3 * y = 9 ∧ x = 3/2 ∧ y = -1) := by
  sorry

#check solution_correctness

end NUMINAMATH_CALUDE_solution_correctness_l3975_397595


namespace NUMINAMATH_CALUDE_stella_annual_income_l3975_397537

def monthly_income : ℕ := 4919
def unpaid_leave_months : ℕ := 2
def months_in_year : ℕ := 12

theorem stella_annual_income :
  (monthly_income * (months_in_year - unpaid_leave_months)) = 49190 := by
  sorry

end NUMINAMATH_CALUDE_stella_annual_income_l3975_397537


namespace NUMINAMATH_CALUDE_school_referendum_l3975_397551

theorem school_referendum (U : Finset ℕ) (A B : Finset ℕ) : 
  Finset.card U = 198 →
  Finset.card A = 149 →
  Finset.card B = 119 →
  Finset.card (U \ (A ∪ B)) = 29 →
  Finset.card (A ∩ B) = 99 := by
  sorry

end NUMINAMATH_CALUDE_school_referendum_l3975_397551


namespace NUMINAMATH_CALUDE_candy_distribution_l3975_397525

theorem candy_distribution (total red blue : ℕ) (h1 : total = 25689) (h2 : red = 1342) (h3 : blue = 8965) :
  let remaining := total - (red + blue)
  ∃ (green : ℕ), green * 3 = remaining ∧ green = 5127 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3975_397525


namespace NUMINAMATH_CALUDE_negative_power_equality_l3975_397576

theorem negative_power_equality : -2010^2011 = (-2010)^2011 := by sorry

end NUMINAMATH_CALUDE_negative_power_equality_l3975_397576


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l3975_397566

/-- The probability of getting exactly k positive answers out of n questions,
    where each question has a p probability of a positive answer. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 3 positive answers out of 7 questions
    from a Magic 8 Ball, where each question has a 3/7 chance of a positive answer. -/
theorem magic_8_ball_probability :
  binomial_probability 7 3 (3/7) = 242520/823543 := by
  sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l3975_397566


namespace NUMINAMATH_CALUDE_max_profit_profit_range_l3975_397591

/-- Represents the store's pricing and sales model -/
structure Store where
  costPrice : ℝ
  maxProfitPercent : ℝ
  k : ℝ
  b : ℝ

/-- Calculates the profit given a selling price -/
def profit (s : Store) (x : ℝ) : ℝ :=
  (x - s.costPrice) * (s.k * x + s.b)

/-- Theorem stating the maximum profit and optimal selling price -/
theorem max_profit (s : Store) 
    (h1 : s.costPrice = 60)
    (h2 : s.maxProfitPercent = 0.45)
    (h3 : s.k = -1)
    (h4 : s.b = 120)
    (h5 : s.k * 65 + s.b = 55)
    (h6 : s.k * 75 + s.b = 45) :
    ∃ (maxProfit sellPrice : ℝ),
      maxProfit = 891 ∧
      sellPrice = 87 ∧
      ∀ x, s.costPrice ≤ x ∧ x ≤ s.costPrice * (1 + s.maxProfitPercent) →
        profit s x ≤ maxProfit := by
  sorry

/-- Theorem stating the selling price range for profit ≥ 500 -/
theorem profit_range (s : Store)
    (h1 : s.costPrice = 60)
    (h2 : s.maxProfitPercent = 0.45)
    (h3 : s.k = -1)
    (h4 : s.b = 120)
    (h5 : s.k * 65 + s.b = 55)
    (h6 : s.k * 75 + s.b = 45) :
    ∀ x, profit s x ≥ 500 ∧ s.costPrice ≤ x ∧ x ≤ s.costPrice * (1 + s.maxProfitPercent) →
      70 ≤ x ∧ x ≤ 87 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_profit_range_l3975_397591


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l3975_397590

/-- A three-digit number satisfying specific conditions -/
def three_digit_number (a b c : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  a + b + c = 10 ∧
  b = a + c ∧
  100 * c + 10 * b + a = 100 * a + 10 * b + c + 99

theorem unique_three_digit_number :
  ∃! (a b c : ℕ), three_digit_number a b c ∧ 100 * a + 10 * b + c = 253 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l3975_397590


namespace NUMINAMATH_CALUDE_f_min_at_one_l3975_397509

/-- The quadratic function that we're analyzing -/
def f (x : ℝ) : ℝ := (x - 1)^2 - 3

/-- Theorem stating that f reaches its minimum value when x = 1 -/
theorem f_min_at_one : ∀ x : ℝ, f x ≥ f 1 := by
  sorry

end NUMINAMATH_CALUDE_f_min_at_one_l3975_397509


namespace NUMINAMATH_CALUDE_cone_ratio_l3975_397513

theorem cone_ratio (circumference : ℝ) (volume : ℝ) :
  circumference = 28 * Real.pi →
  volume = 441 * Real.pi →
  ∃ (radius height : ℝ),
    circumference = 2 * Real.pi * radius ∧
    volume = (1/3) * Real.pi * radius^2 * height ∧
    radius / height = 14 / 9 :=
by sorry

end NUMINAMATH_CALUDE_cone_ratio_l3975_397513


namespace NUMINAMATH_CALUDE_fast_food_purchase_cost_l3975_397594

/-- The cost of a single sandwich -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda -/
def soda_cost : ℕ := 3

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 6

/-- The number of sodas purchased -/
def num_sodas : ℕ := 5

/-- The total cost of the purchase -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem fast_food_purchase_cost : total_cost = 39 := by
  sorry

end NUMINAMATH_CALUDE_fast_food_purchase_cost_l3975_397594


namespace NUMINAMATH_CALUDE_log_inequality_l3975_397558

theorem log_inequality (x y z : ℝ) 
  (hx : x = 6 * (Real.log 3 / Real.log 64))
  (hy : y = (1/3) * (Real.log 64 / Real.log 3))
  (hz : z = (3/2) * (Real.log 3 / Real.log 8)) :
  x > y ∧ y > z := by sorry

end NUMINAMATH_CALUDE_log_inequality_l3975_397558


namespace NUMINAMATH_CALUDE_max_expression_proof_l3975_397504

/-- The maximum value of c * a^b - d given the constraints --/
def max_expression : ℕ := 625

/-- The set of possible values for a, b, c, and d --/
def value_set : Finset ℕ := {0, 1, 4, 5}

/-- Proposition: The maximum value of c * a^b - d is 625, given the constraints --/
theorem max_expression_proof :
  ∀ a b c d : ℕ,
    a ∈ value_set → b ∈ value_set → c ∈ value_set → d ∈ value_set →
    a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
    c * a^b - d ≤ max_expression :=
by sorry

end NUMINAMATH_CALUDE_max_expression_proof_l3975_397504


namespace NUMINAMATH_CALUDE_quadratic_roots_bound_l3975_397505

theorem quadratic_roots_bound (a b : ℝ) (α β : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + b = 0) →  -- Equation has real roots
  (α^2 + a*α + b = 0) →           -- α is a root
  (β^2 + a*β + b = 0) →           -- β is a root
  (α ≠ β) →                       -- Roots are distinct
  (2 * abs a < 4 + b ∧ abs b < 4 ↔ abs α < 2 ∧ abs β < 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_bound_l3975_397505


namespace NUMINAMATH_CALUDE_guarantee_target_color_count_l3975_397599

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  purple : Nat

/-- The initial ball counts in the box -/
def initialBalls : BallCounts :=
  { red := 30, green := 24, yellow := 16, blue := 14, white := 12, purple := 4 }

/-- The minimum number of balls of a single color we want to guarantee -/
def targetCount : Nat := 12

/-- The number of balls we claim will guarantee the target count -/
def claimedDrawCount : Nat := 60

/-- Theorem stating that drawing the claimed number of balls guarantees
    at least the target count of a single color -/
theorem guarantee_target_color_count :
  ∀ (drawn : Nat),
    drawn ≥ claimedDrawCount →
    ∃ (color : Fin 6),
      (match color with
       | 0 => initialBalls.red
       | 1 => initialBalls.green
       | 2 => initialBalls.yellow
       | 3 => initialBalls.blue
       | 4 => initialBalls.white
       | 5 => initialBalls.purple) -
      (claimedDrawCount - drawn) ≥ targetCount :=
by sorry

end NUMINAMATH_CALUDE_guarantee_target_color_count_l3975_397599


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l3975_397534

theorem logarithm_expression_equality : 
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 - 5 ^ (Real.log 3 / Real.log 5) = -1 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l3975_397534


namespace NUMINAMATH_CALUDE_parabola_b_value_l3975_397550

/-- A parabola passing through two points -/
def Parabola (b c : ℝ) : ℝ → ℝ := fun x ↦ x^2 + b*x + c

theorem parabola_b_value :
  ∀ b c : ℝ,
  Parabola b c 1 = 2 →
  Parabola b c 5 = 2 →
  b = -6 := by
sorry

end NUMINAMATH_CALUDE_parabola_b_value_l3975_397550


namespace NUMINAMATH_CALUDE_tank_fill_time_l3975_397539

/-- Represents the time it takes to fill a tank given the rates of three pipes -/
def fill_time (rate1 rate2 rate3 : ℚ) : ℚ :=
  1 / (rate1 + rate2 + rate3)

/-- Theorem stating that given specific pipe rates, the fill time is 20 minutes -/
theorem tank_fill_time :
  fill_time (1/18) (1/60) (-1/45) = 20 := by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_l3975_397539


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l3975_397536

theorem rectangle_area_ratio (large_horizontal small_horizontal large_vertical small_vertical large_area : ℝ)
  (h_horizontal_ratio : large_horizontal / small_horizontal = 8 / 7)
  (h_vertical_ratio : large_vertical / small_vertical = 9 / 4)
  (h_large_area : large_horizontal * large_vertical = large_area)
  (h_large_area_value : large_area = 108) :
  small_horizontal * small_vertical = 42 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l3975_397536


namespace NUMINAMATH_CALUDE_triangle_proof_l3975_397572

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_proof (t : Triangle) (h1 : t.a * Real.cos t.C + Real.sqrt 3 * t.a * Real.sin t.C - t.b - t.c = 0)
                       (h2 : t.a = 2)
                       (h3 : (1/2) * t.a * t.b * Real.sin t.C = Real.sqrt 3) :
  t.A = Real.pi / 3 ∧ t.b = 2 ∧ t.c = 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_proof_l3975_397572


namespace NUMINAMATH_CALUDE_sum_greater_than_one_l3975_397598

theorem sum_greater_than_one
  (a b c d : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hd : d > 0)
  (hac : a > c)
  (hbd : b < d)
  (h1 : a + Real.sqrt b ≥ c + Real.sqrt d)
  (h2 : Real.sqrt a + b ≤ Real.sqrt c + d) :
  a + b + c + d > 1 :=
by sorry

end NUMINAMATH_CALUDE_sum_greater_than_one_l3975_397598


namespace NUMINAMATH_CALUDE_odd_function_sum_l3975_397516

/-- A function f is odd on an interval [a, b] -/
def IsOddOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x ∈ Set.Icc a b, f (-x) = -f x) ∧ a + b = 0

/-- The main theorem -/
theorem odd_function_sum (a b c : ℝ) :
  IsOddOn (fun x ↦ a * x^3 + x + c) a b →
  a + b + c + 2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_odd_function_sum_l3975_397516


namespace NUMINAMATH_CALUDE_coin_order_l3975_397589

/-- Represents the relative position of coins -/
inductive Position
| Above
| Below
| Same
| Unknown

/-- Represents a coin -/
inductive Coin
| F
| A
| B
| C
| D
| E

/-- Defines the relative position between two coins -/
def relative_position (c1 c2 : Coin) : Position := sorry

/-- Defines whether a coin is directly above another -/
def is_directly_above (c1 c2 : Coin) : Prop := 
  relative_position c1 c2 = Position.Above ∧ 
  ∀ c, c ≠ c1 ∧ c ≠ c2 → relative_position c1 c = Position.Above ∨ relative_position c c2 = Position.Above

/-- The main theorem to prove -/
theorem coin_order :
  (∀ c, c ≠ Coin.F → relative_position Coin.F c = Position.Above) ∧
  (is_directly_above Coin.A Coin.B) ∧
  (is_directly_above Coin.A Coin.C) ∧
  (relative_position Coin.A Coin.D = Position.Unknown) ∧
  (relative_position Coin.A Coin.E = Position.Unknown) ∧
  (is_directly_above Coin.D Coin.E) ∧
  (is_directly_above Coin.E Coin.B) ∧
  (∀ c, c ≠ Coin.F ∧ c ≠ Coin.A → relative_position c Coin.C = Position.Below ∨ relative_position c Coin.C = Position.Unknown) →
  (relative_position Coin.F Coin.A = Position.Above) ∧
  (relative_position Coin.A Coin.D = Position.Above) ∧
  (relative_position Coin.D Coin.E = Position.Above) ∧
  (relative_position Coin.E Coin.C = Position.Above ∨ relative_position Coin.E Coin.C = Position.Unknown) ∧
  (relative_position Coin.C Coin.B = Position.Above) := by
  sorry

end NUMINAMATH_CALUDE_coin_order_l3975_397589


namespace NUMINAMATH_CALUDE_root_equation_value_l3975_397517

theorem root_equation_value (m : ℝ) : 
  m^2 - m - 2 = 0 → m^2 - m + 2023 = 2025 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l3975_397517


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l3975_397585

/-- In a right-angled triangle with legs a and b, hypotenuse c, and altitude m
    corresponding to the hypotenuse, m + c > a + b -/
theorem right_triangle_inequality (a b c m : ℝ) 
  (h_right : a^2 + b^2 = c^2)  -- Pythagorean theorem
  (h_altitude : a * b = c * m) -- Area equality
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ m > 0) : m + c > a + b := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_inequality_l3975_397585


namespace NUMINAMATH_CALUDE_max_area_between_parabolas_l3975_397529

/-- The parabola C_a -/
def C_a (a x : ℝ) : ℝ := -2 * x^2 + 4 * a * x - 2 * a^2 + a + 1

/-- The parabola C -/
def C (x : ℝ) : ℝ := x^2 - 2 * x

/-- The difference function between C and C_a -/
def f_a (a x : ℝ) : ℝ := C x - C_a a x

/-- Theorem: The maximum area enclosed by parabolas C_a and C is 27/(4√2) -/
theorem max_area_between_parabolas :
  ∃ (a : ℝ), (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f_a a x₁ = 0 ∧ f_a a x₂ = 0) →
  (∫ (x : ℝ) in Set.Icc (min x₁ x₂) (max x₁ x₂), f_a a x) ≤ 27 / (4 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_max_area_between_parabolas_l3975_397529


namespace NUMINAMATH_CALUDE_correct_hand_in_amount_l3975_397583

/-- Calculates the amount of money Jack will hand in given the number of bills of each denomination and the amount to be left in the till -/
def money_to_hand_in (hundreds twos fifties twenties tens fives ones leave_in_till : ℕ) : ℕ :=
  let total_in_notes := 100 * hundreds + 50 * fifties + 20 * twenties + 10 * tens + 5 * fives + ones
  total_in_notes - leave_in_till

/-- Theorem stating that the amount Jack will hand in is correct given the problem conditions -/
theorem correct_hand_in_amount :
  money_to_hand_in 2 0 1 5 3 7 27 300 = 142 := by
  sorry

#eval money_to_hand_in 2 0 1 5 3 7 27 300

end NUMINAMATH_CALUDE_correct_hand_in_amount_l3975_397583


namespace NUMINAMATH_CALUDE_hexagon_not_possible_after_cut_l3975_397597

-- Define a polygon
structure Polygon :=
  (sides : ℕ)
  (sides_ge_3 : sides ≥ 3)

-- Define the operation of cutting off a corner
def cut_corner (p : Polygon) : Polygon :=
  ⟨p.sides - 1, by sorry⟩

-- Theorem statement
theorem hexagon_not_possible_after_cut (p : Polygon) :
  (cut_corner p).sides = 4 → p.sides ≠ 6 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_not_possible_after_cut_l3975_397597


namespace NUMINAMATH_CALUDE_solution_set_implies_m_value_l3975_397556

theorem solution_set_implies_m_value (m : ℝ) :
  (∀ x : ℝ, x^2 - (m + 2) * x > 0 ↔ x < 0 ∨ x > 2) →
  m = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_m_value_l3975_397556


namespace NUMINAMATH_CALUDE_simplify_radical_sum_l3975_397586

theorem simplify_radical_sum : Real.sqrt 98 + Real.sqrt 32 + (27 : Real).rpow (1/3) = 11 * Real.sqrt 2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_sum_l3975_397586


namespace NUMINAMATH_CALUDE_runner_journey_time_l3975_397570

/-- Represents the runner's journey --/
structure RunnerJourney where
  totalDistance : ℝ
  firstHalfSpeed : ℝ
  secondHalfSpeed : ℝ
  firstHalfTime : ℝ
  secondHalfTime : ℝ

/-- Theorem stating the conditions and the conclusion to be proved --/
theorem runner_journey_time (j : RunnerJourney) 
  (h1 : j.totalDistance = 40)
  (h2 : j.secondHalfSpeed = j.firstHalfSpeed / 2)
  (h3 : j.secondHalfTime = j.firstHalfTime + 12)
  (h4 : j.firstHalfTime = (j.totalDistance / 2) / j.firstHalfSpeed)
  (h5 : j.secondHalfTime = (j.totalDistance / 2) / j.secondHalfSpeed) :
  j.secondHalfTime = 24 := by
  sorry


end NUMINAMATH_CALUDE_runner_journey_time_l3975_397570


namespace NUMINAMATH_CALUDE_exists_k_for_all_m_unique_k_characterization_l3975_397579

/-- The number of elements in {k+1, k+2, ..., 2k} with exactly 3 ones in binary representation -/
def f (k : ℕ+) : ℕ := sorry

/-- There exists a k for every m such that f(k) = m -/
theorem exists_k_for_all_m (m : ℕ+) : ∃ k : ℕ+, f k = m := by sorry

/-- Characterization of m for which there's exactly one k satisfying f(k) = m -/
theorem unique_k_characterization (m : ℕ+) : 
  (∃! k : ℕ+, f k = m) ↔ ∃ n : ℕ, n ≥ 2 ∧ m = n * (n - 1) / 2 + 1 := by sorry

end NUMINAMATH_CALUDE_exists_k_for_all_m_unique_k_characterization_l3975_397579


namespace NUMINAMATH_CALUDE_min_value_expression_l3975_397541

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) :
  ∃ (m : ℝ), m = 2/675 ∧ ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → 1/a + 1/b + 1/c = 9 → 2 * a^3 * b^2 * c ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3975_397541


namespace NUMINAMATH_CALUDE_remainder_67_power_67_plus_67_mod_68_l3975_397542

theorem remainder_67_power_67_plus_67_mod_68 : (67^67 + 67) % 68 = 66 := by
  sorry

end NUMINAMATH_CALUDE_remainder_67_power_67_plus_67_mod_68_l3975_397542


namespace NUMINAMATH_CALUDE_intersection_A_B_l3975_397559

def A : Set ℝ := {x | (x + 3) * (2 - x) > 0}
def B : Set ℝ := {-5, -4, 0, 1, 4}

theorem intersection_A_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3975_397559


namespace NUMINAMATH_CALUDE_undefined_at_eleven_l3975_397506

theorem undefined_at_eleven (x : ℝ) : 
  (∃ y, (3 * x^2 + 5) / (x^2 - 22*x + 121) = y) ↔ x ≠ 11 :=
by sorry

end NUMINAMATH_CALUDE_undefined_at_eleven_l3975_397506


namespace NUMINAMATH_CALUDE_quadratic_root_k_value_l3975_397521

theorem quadratic_root_k_value (k : ℝ) : 
  (∃ x : ℝ, x^2 - k*x - 6 = 0 ∧ x = -3) → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_k_value_l3975_397521


namespace NUMINAMATH_CALUDE_no_real_solution_for_equation_l3975_397560

theorem no_real_solution_for_equation : 
  ∀ x : ℝ, ¬(5 * (2*x)^2 - 3*(2*x) + 7 = 2*(8*x^2 - 2*x + 3)) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_for_equation_l3975_397560


namespace NUMINAMATH_CALUDE_c_bounds_l3975_397571

theorem c_bounds (a b c : ℝ) (h1 : a + 2*b + c = 1) (h2 : a^2 + b^2 + c^2 = 1) :
  -2/3 ≤ c ∧ c ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_c_bounds_l3975_397571


namespace NUMINAMATH_CALUDE_group_five_frequency_l3975_397554

theorem group_five_frequency (total : ℕ) (group1 group2 group3 group4 : ℕ) 
  (h_total : total = 50)
  (h_group1 : group1 = 2)
  (h_group2 : group2 = 8)
  (h_group3 : group3 = 15)
  (h_group4 : group4 = 5) :
  (total - group1 - group2 - group3 - group4 : ℚ) / total = 0.4 := by
sorry

end NUMINAMATH_CALUDE_group_five_frequency_l3975_397554


namespace NUMINAMATH_CALUDE_expression_evaluation_l3975_397563

theorem expression_evaluation :
  let sin30 : Real := 1/2
  4 * (Real.sqrt 3 + Real.sqrt 7) / (5 * Real.sqrt (3 + sin30)) = (16 + 8 * Real.sqrt 21) / 35 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3975_397563


namespace NUMINAMATH_CALUDE_max_red_socks_is_990_l3975_397510

/-- Represents the number of socks in a drawer -/
structure SockDrawer where
  red : ℕ
  blue : ℕ
  total_le_2000 : red + blue ≤ 2000
  blue_lt_red : blue < red
  total_odd : ¬ Even (red + blue)
  prob_same_color : (red * (red - 1) + blue * (blue - 1)) = (red + blue) * (red + blue - 1) / 2

/-- The maximum number of red socks possible in the drawer -/
def max_red_socks : ℕ := 990

/-- Theorem stating that the maximum number of red socks is 990 -/
theorem max_red_socks_is_990 (drawer : SockDrawer) : drawer.red ≤ max_red_socks := by
  sorry

end NUMINAMATH_CALUDE_max_red_socks_is_990_l3975_397510


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l3975_397567

/-- The ratio of the side length of a regular pentagon to the width of a rectangle is 6/5, 
    given that both shapes have a perimeter of 60 inches and the rectangle's length is twice its width. -/
theorem pentagon_rectangle_ratio : 
  ∀ (pentagon_side rectangle_width rectangle_length : ℝ),
  pentagon_side * 5 = 60 →
  rectangle_width * 2 + rectangle_length * 2 = 60 →
  rectangle_length = 2 * rectangle_width →
  pentagon_side / rectangle_width = 6 / 5 := by
sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l3975_397567


namespace NUMINAMATH_CALUDE_coprime_elements_bound_l3975_397569

/-- The number of elements in [1, n] coprime to M -/
def h (M n : ℕ+) : ℕ := sorry

/-- The proportion of numbers in [1, M] coprime to M -/
def β (M : ℕ+) : ℚ := (h M M : ℚ) / M

/-- ω(M) is the number of distinct prime factors of M -/
def ω (M : ℕ+) : ℕ := sorry

theorem coprime_elements_bound (M : ℕ+) :
  ∃ S : Finset ℕ+,
    S.card ≥ M / 3 ∧
    ∀ n ∈ S, n ≤ M ∧
    |h M n - β M * n| ≤ Real.sqrt (β M * 2^(ω M - 3)) + 1 :=
  sorry

end NUMINAMATH_CALUDE_coprime_elements_bound_l3975_397569


namespace NUMINAMATH_CALUDE_inequality_always_true_l3975_397540

theorem inequality_always_true (x : ℝ) : (7 / 20) + |3 * x - (2 / 5)| ≥ (1 / 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_true_l3975_397540


namespace NUMINAMATH_CALUDE_colored_pencils_count_l3975_397578

def number_of_packs : ℕ := 7
def pencils_per_pack : ℕ := 10
def difference : ℕ := 3

theorem colored_pencils_count :
  let total_pencils := number_of_packs * pencils_per_pack
  let colored_pencils := total_pencils + difference
  colored_pencils = 73 := by
  sorry

end NUMINAMATH_CALUDE_colored_pencils_count_l3975_397578


namespace NUMINAMATH_CALUDE_consecutive_sets_sum_150_l3975_397546

/-- A structure representing a set of consecutive integers -/
structure ConsecutiveSet where
  start : ℕ
  length : ℕ
  sum_is_150 : (length * (2 * start + length - 1)) / 2 = 150
  length_ge_2 : length ≥ 2

/-- The theorem stating that there are exactly 5 sets of consecutive integers summing to 150 -/
theorem consecutive_sets_sum_150 :
  (∃ (sets : Finset ConsecutiveSet), sets.card = 5 ∧
    (∀ s : ConsecutiveSet, s ∈ sets ↔ 
      (s.length * (2 * s.start + s.length - 1)) / 2 = 150 ∧ 
      s.length ≥ 2)) :=
sorry

end NUMINAMATH_CALUDE_consecutive_sets_sum_150_l3975_397546


namespace NUMINAMATH_CALUDE_cos_250_over_sin_200_equals_1_l3975_397514

theorem cos_250_over_sin_200_equals_1 :
  (Real.cos (250 * π / 180)) / (Real.sin (200 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_250_over_sin_200_equals_1_l3975_397514


namespace NUMINAMATH_CALUDE_library_fine_fifth_day_l3975_397562

def fine_calculation (initial_fine : Float) (increase : Float) (days : Nat) : Float :=
  let rec calc_fine (current_fine : Float) (day : Nat) : Float :=
    if day = 0 then
      current_fine
    else
      let increased := current_fine + increase
      let doubled := current_fine * 2
      calc_fine (min increased doubled) (day - 1)
  calc_fine initial_fine days

theorem library_fine_fifth_day :
  fine_calculation 0.07 0.30 4 = 0.86 := by
  sorry

end NUMINAMATH_CALUDE_library_fine_fifth_day_l3975_397562


namespace NUMINAMATH_CALUDE_smallest_invertible_domain_l3975_397545

-- Define the function f
def f (x : ℝ) : ℝ := (x + 3)^2 - 7

-- State the theorem
theorem smallest_invertible_domain : 
  ∃ (c : ℝ), c = -3 ∧ 
  (∀ x y, x ≥ c → y ≥ c → f x = f y → x = y) ∧ 
  (∀ c' < c, ∃ x y, x ≥ c' → y ≥ c' → f x = f y ∧ x ≠ y) :=
sorry

end NUMINAMATH_CALUDE_smallest_invertible_domain_l3975_397545


namespace NUMINAMATH_CALUDE_solve_equation_l3975_397557

theorem solve_equation : ∃ y : ℝ, (2 * y) / 3 = 30 ∧ y = 45 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3975_397557


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3975_397573

theorem arithmetic_sequence_formula (x : ℝ) (a : ℕ → ℝ) :
  (a 1 = x - 1) →
  (a 2 = x + 1) →
  (a 3 = 2*x + 3) →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = a 2 - a 1) →
  (∀ n : ℕ, n ≥ 1 → a n = 2*n - 3) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3975_397573


namespace NUMINAMATH_CALUDE_translation_company_min_employees_l3975_397508

/-- The number of languages offered by the company -/
def num_languages : ℕ := 4

/-- The number of languages each employee must learn -/
def languages_per_employee : ℕ := 2

/-- The minimum number of employees with identical training -/
def min_identical_training : ℕ := 5

/-- The number of possible language combinations -/
def num_combinations : ℕ := Nat.choose num_languages languages_per_employee

/-- The minimum number of employees in the company -/
def min_employees : ℕ := 25

theorem translation_company_min_employees :
  ∀ n : ℕ, n ≥ min_employees →
    ∃ (group : Finset (Finset (Fin num_languages))),
      (∀ e ∈ group, Finset.card e = languages_per_employee) ∧
      (Finset.card group ≥ min_identical_training) :=
by sorry

end NUMINAMATH_CALUDE_translation_company_min_employees_l3975_397508


namespace NUMINAMATH_CALUDE_power_function_through_point_is_odd_l3975_397526

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem power_function_through_point_is_odd
  (f : ℝ → ℝ)
  (h1 : is_power_function f)
  (h2 : f (Real.sqrt 3 / 3) = Real.sqrt 3) :
  is_odd_function f :=
sorry

end NUMINAMATH_CALUDE_power_function_through_point_is_odd_l3975_397526


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_equation_l3975_397515

/-- Given three consecutive odd numbers where the first is 7, 
    prove that the multiple of the third number that satisfies 
    the equation is 3 --/
theorem consecutive_odd_numbers_equation (n : ℕ) : 
  let first := 7
  let second := first + 2
  let third := second + 2
  8 * first = n * third + 5 + 2 * second → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_equation_l3975_397515


namespace NUMINAMATH_CALUDE_f_2010_equals_zero_l3975_397520

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_2010_equals_zero
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (x + 3) = -f x) :
  f 2010 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_2010_equals_zero_l3975_397520


namespace NUMINAMATH_CALUDE_percentage_less_than_third_l3975_397561

theorem percentage_less_than_third (n1 n2 n3 : ℝ) : 
  n1 = 0.7 * n3 →  -- First number is 30% less than third number
  n2 = 0.9 * n1 →  -- Second number is 10% less than first number
  n2 = 0.63 * n3   -- Second number is 37% less than third number
:= by sorry

end NUMINAMATH_CALUDE_percentage_less_than_third_l3975_397561


namespace NUMINAMATH_CALUDE_raja_income_distribution_l3975_397575

theorem raja_income_distribution (monthly_income : ℝ) 
  (household_percentage : ℝ) (medicine_percentage : ℝ) (savings : ℝ) :
  monthly_income = 37500 →
  household_percentage = 35 →
  medicine_percentage = 5 →
  savings = 15000 →
  ∃ (clothes_percentage : ℝ),
    clothes_percentage = 20 ∧
    (household_percentage / 100 + medicine_percentage / 100 + clothes_percentage / 100) * monthly_income + savings = monthly_income :=
by sorry

end NUMINAMATH_CALUDE_raja_income_distribution_l3975_397575


namespace NUMINAMATH_CALUDE_square_sum_of_roots_l3975_397596

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

-- State the theorem
theorem square_sum_of_roots (a b : ℝ) (ha : f a = 1) (hb : f b = 19) : (a + b)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_roots_l3975_397596


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3975_397531

/-- Proves that a triangle with inradius 2.5 cm and area 50 cm² has a perimeter of 40 cm -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 2.5 → A = 50 → A = r * (p / 2) → p = 40 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3975_397531


namespace NUMINAMATH_CALUDE_min_value_and_k_range_l3975_397501

open Real

noncomputable def f (x : ℝ) : ℝ := log x + 2 / x

noncomputable def σ (x : ℝ) : ℝ := log x + exp x / x - x

noncomputable def g (x : ℝ) : ℝ := x^2 - x * log x + 2

theorem min_value_and_k_range :
  (∃ (x : ℝ), ∀ (y : ℝ), y > 0 → σ y ≥ σ x) ∧
  σ (1 : ℝ) = exp 1 - 1 ∧
  ∀ (k : ℝ), (∃ (a b : ℝ), 1/2 ≤ a ∧ a < b ∧
    (∀ (x : ℝ), a ≤ x ∧ x ≤ b → k * (a + 2) ≤ g x ∧ g x ≤ k * (b + 2))) →
    1 < k ∧ k ≤ (9 + 2 * log 2) / 10 :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_k_range_l3975_397501


namespace NUMINAMATH_CALUDE_bolded_area_percentage_l3975_397555

theorem bolded_area_percentage (s : ℝ) (h : s > 0) : 
  let square_area := s^2
  let total_area := 4 * square_area
  let bolded_area_1 := (1/2) * square_area
  let bolded_area_2 := (1/2) * square_area
  let bolded_area_3 := (1/8) * square_area
  let bolded_area_4 := (1/4) * square_area
  let total_bolded_area := bolded_area_1 + bolded_area_2 + bolded_area_3 + bolded_area_4
  (total_bolded_area / total_area) * 100 = 100/3
:= by sorry

end NUMINAMATH_CALUDE_bolded_area_percentage_l3975_397555


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3975_397532

theorem interest_rate_calculation (total_sum second_part : ℚ) 
  (h1 : total_sum = 2704)
  (h2 : second_part = 1664)
  (h3 : total_sum > second_part) :
  let first_part := total_sum - second_part
  let interest_first := first_part * (3/100) * 8
  let interest_second := second_part * (5/100) * 3
  interest_first = interest_second := by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3975_397532
