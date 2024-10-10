import Mathlib

namespace johns_pre_raise_earnings_l375_37514

/-- The amount John makes per week after the raise, in dollars. -/
def post_raise_earnings : ℝ := 60

/-- The percentage increase of John's earnings. -/
def percentage_increase : ℝ := 50

/-- John's weekly earnings before the raise, in dollars. -/
def pre_raise_earnings : ℝ := 40

/-- Theorem stating that John's pre-raise earnings were $40, given the conditions. -/
theorem johns_pre_raise_earnings : 
  pre_raise_earnings * (1 + percentage_increase / 100) = post_raise_earnings := by
  sorry

end johns_pre_raise_earnings_l375_37514


namespace subtraction_correction_l375_37533

theorem subtraction_correction (x : ℤ) : x - 63 = 24 → x - 36 = 51 := by
  sorry

end subtraction_correction_l375_37533


namespace wheel_diameter_l375_37568

/-- The diameter of a wheel given its revolutions and distance covered -/
theorem wheel_diameter (revolutions : ℝ) (distance : ℝ) (π : ℝ) :
  revolutions = 8.007279344858963 →
  distance = 1056 →
  π = 3.14159 →
  ∃ (diameter : ℝ), abs (diameter - 41.975) < 0.001 :=
by
  sorry

end wheel_diameter_l375_37568


namespace cat_cleaner_amount_l375_37524

/-- The amount of cleaner used for a dog stain in ounces -/
def dog_cleaner : ℝ := 6

/-- The amount of cleaner used for a rabbit stain in ounces -/
def rabbit_cleaner : ℝ := 1

/-- The total amount of cleaner used for all stains in ounces -/
def total_cleaner : ℝ := 49

/-- The number of dogs -/
def num_dogs : ℕ := 6

/-- The number of cats -/
def num_cats : ℕ := 3

/-- The number of rabbits -/
def num_rabbits : ℕ := 1

/-- The amount of cleaner used for a cat stain in ounces -/
def cat_cleaner : ℝ := 4

theorem cat_cleaner_amount :
  dog_cleaner * num_dogs + cat_cleaner * num_cats + rabbit_cleaner * num_rabbits = total_cleaner :=
by sorry

end cat_cleaner_amount_l375_37524


namespace at_least_one_basketball_l375_37537

/-- Represents the total number of balls -/
def totalBalls : ℕ := 8

/-- Represents the number of basketballs -/
def numBasketballs : ℕ := 6

/-- Represents the number of volleyballs -/
def numVolleyballs : ℕ := 2

/-- Represents the number of balls to be chosen -/
def chosenBalls : ℕ := 3

/-- Theorem stating that at least one basketball is always chosen -/
theorem at_least_one_basketball : 
  ∀ (selection : Finset (Fin totalBalls)), 
  selection.card = chosenBalls → 
  ∃ (i : Fin totalBalls), i ∈ selection ∧ i.val < numBasketballs :=
sorry

end at_least_one_basketball_l375_37537


namespace hcf_of_numbers_l375_37587

theorem hcf_of_numbers (x y : ℕ+) 
  (sum_eq : x + y = 45)
  (lcm_eq : Nat.lcm x y = 120)
  (sum_recip_eq : (1 : ℚ) / x + (1 : ℚ) / y = 11 / 120) :
  Nat.gcd x y = 1 := by
  sorry

end hcf_of_numbers_l375_37587


namespace factorial_ratio_simplification_l375_37503

theorem factorial_ratio_simplification : (Nat.factorial 10 * Nat.factorial 6 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 60 / 7 := by
  sorry

end factorial_ratio_simplification_l375_37503


namespace f_zero_at_three_l375_37511

/-- The polynomial function f(x) = 3x^4 - 2x^3 + x^2 - 4x + r -/
def f (x r : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + x^2 - 4 * x + r

/-- Theorem stating that f(3) = 0 if and only if r = -186 -/
theorem f_zero_at_three (r : ℝ) : f 3 r = 0 ↔ r = -186 := by
  sorry

end f_zero_at_three_l375_37511


namespace library_repacking_l375_37532

/-- Given a number of boxes and books per box, calculate the number of books left over when repacking into new boxes with a different number of books per box. -/
def books_left_over (initial_boxes : ℕ) (initial_books_per_box : ℕ) (new_books_per_box : ℕ) : ℕ :=
  let total_books := initial_boxes * initial_books_per_box
  total_books % new_books_per_box

/-- Prove that given 1575 boxes with 45 books each, when repacking into boxes of 50 books each, the number of books left over is 25. -/
theorem library_repacking : books_left_over 1575 45 50 = 25 := by
  sorry

end library_repacking_l375_37532


namespace only_math_is_75_l375_37551

/-- Represents the number of students in different subject combinations -/
structure StudentCounts where
  total : ℕ
  math : ℕ
  foreignLanguage : ℕ
  science : ℕ
  allThree : ℕ

/-- The actual student counts from the problem -/
def actualCounts : StudentCounts :=
  { total := 120
  , math := 85
  , foreignLanguage := 65
  , science := 75
  , allThree := 20 }

/-- Calculate the number of students taking only math -/
def onlyMathCount (counts : StudentCounts) : ℕ :=
  counts.math - (counts.total - (counts.math + counts.foreignLanguage + counts.science - counts.allThree))

/-- Theorem stating that the number of students taking only math is 75 -/
theorem only_math_is_75 : onlyMathCount actualCounts = 75 := by
  sorry

end only_math_is_75_l375_37551


namespace train_passing_jogger_time_train_passes_jogger_in_37_seconds_l375_37528

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (train_length : ℝ) 
  (initial_distance : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- The train passes the jogger in 37 seconds under the given conditions -/
theorem train_passes_jogger_in_37_seconds : 
  train_passing_jogger_time 9 45 120 250 = 37 := by
  sorry

end train_passing_jogger_time_train_passes_jogger_in_37_seconds_l375_37528


namespace quadrilateral_offset_l375_37584

/-- Given a quadrilateral with one diagonal of 50 cm, one offset of 8 cm, and an area of 450 cm²,
    the length of the other offset is 10 cm. -/
theorem quadrilateral_offset (diagonal : ℝ) (offset1 : ℝ) (area : ℝ) :
  diagonal = 50 ∧ offset1 = 8 ∧ area = 450 →
  ∃ offset2 : ℝ, offset2 = 10 ∧ area = (diagonal * (offset1 + offset2)) / 2 := by
  sorry

end quadrilateral_offset_l375_37584


namespace acrobats_count_l375_37539

/-- The number of acrobats in a parade group -/
def num_acrobats : ℕ := 10

/-- The number of elephants in a parade group -/
def num_elephants : ℕ := 20 - num_acrobats

/-- The total number of legs in the parade group -/
def total_legs : ℕ := 60

/-- The total number of heads in the parade group -/
def total_heads : ℕ := 20

/-- Theorem stating that the number of acrobats is 10 given the conditions -/
theorem acrobats_count :
  (2 * num_acrobats + 4 * num_elephants = total_legs) ∧
  (num_acrobats + num_elephants = total_heads) ∧
  (num_acrobats = 10) := by
  sorry

end acrobats_count_l375_37539


namespace fraction_invariance_l375_37573

theorem fraction_invariance (x y : ℝ) (hx : x ≠ 0) : (y + x) / x = (3*y + 3*x) / (3*x) := by
  sorry

end fraction_invariance_l375_37573


namespace blue_candy_count_l375_37591

theorem blue_candy_count (total : ℕ) (red : ℕ) (h1 : total = 3409) (h2 : red = 145) :
  total - red = 3264 := by
  sorry

end blue_candy_count_l375_37591


namespace frog_population_equality_l375_37592

theorem frog_population_equality : ∃ n : ℕ, n > 0 ∧ n = 6 ∧ ∀ m : ℕ, m > 0 → (5^(m+1) = 243 * 3^m → m ≥ n) := by
  sorry

end frog_population_equality_l375_37592


namespace remainder_17_pow_2046_mod_23_l375_37518

theorem remainder_17_pow_2046_mod_23 : 17^2046 % 23 = 22 := by
  sorry

end remainder_17_pow_2046_mod_23_l375_37518


namespace additional_sticks_needed_l375_37570

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the problem setup -/
structure ProblemSetup where
  large_rectangle : Rectangle
  total_sticks : ℕ
  num_small_rectangles : ℕ
  small_rectangle_types : List Rectangle

/-- The main theorem statement -/
theorem additional_sticks_needed 
  (setup : ProblemSetup)
  (h1 : setup.large_rectangle = ⟨8, 12⟩)
  (h2 : setup.total_sticks = 40)
  (h3 : setup.num_small_rectangles = 40)
  (h4 : setup.small_rectangle_types = [⟨1, 2⟩, ⟨1, 3⟩])
  : ∃ (additional_sticks : ℕ), additional_sticks = 116 ∧
    ∃ (small_rectangles : List Rectangle),
      small_rectangles.length = setup.num_small_rectangles ∧
      (∀ r ∈ small_rectangles, r ∈ setup.small_rectangle_types) ∧
      (small_rectangles.map (λ r => r.width * r.height)).sum = 
        setup.large_rectangle.width * setup.large_rectangle.height :=
by
  sorry


end additional_sticks_needed_l375_37570


namespace cPass_max_entries_aPass_cost_effective_l375_37525

-- Define the ticketing options
structure TicketOption where
  initialCost : ℕ
  entryCost : ℕ

-- Define the budget
def budget : ℕ := 80

-- Define the ticketing options
def noPass : TicketOption := ⟨0, 10⟩
def aPass : TicketOption := ⟨120, 0⟩
def bPass : TicketOption := ⟨60, 2⟩
def cPass : TicketOption := ⟨40, 3⟩

-- Function to calculate the number of entries for a given option and budget
def numEntries (option : TicketOption) (budget : ℕ) : ℕ :=
  if option.initialCost > budget then 0
  else (budget - option.initialCost) / option.entryCost

-- Theorem 1: C pass allows for the maximum number of entries with 80 yuan budget
theorem cPass_max_entries :
  ∀ option : TicketOption, numEntries cPass budget ≥ numEntries option budget :=
sorry

-- Theorem 2: A pass becomes more cost-effective when entering more than 30 times
theorem aPass_cost_effective (n : ℕ) (h : n > 30) :
  ∀ option : TicketOption, option.initialCost + n * option.entryCost > aPass.initialCost :=
sorry

end cPass_max_entries_aPass_cost_effective_l375_37525


namespace specific_tetrahedron_volume_l375_37565

/-- Represents a tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  -- Length of edge AB
  ab_length : ℝ
  -- Length of edge CD
  cd_length : ℝ
  -- Distance between lines AB and CD
  line_distance : ℝ
  -- Angle between lines AB and CD
  line_angle : ℝ

/-- Calculates the volume of the tetrahedron -/
def tetrahedron_volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 1/2 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    ab_length := 1,
    cd_length := Real.sqrt 3,
    line_distance := 2,
    line_angle := π / 3
  }
  tetrahedron_volume t = 1 / 2 := by
  sorry

end specific_tetrahedron_volume_l375_37565


namespace pencil_distribution_problem_l375_37547

/-- The number of ways to distribute pencils among friends -/
def distribute_pencils (total_pencils : ℕ) (num_friends : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of distributions for the given problem -/
theorem pencil_distribution_problem :
  distribute_pencils 10 4 = 58 :=
sorry

end pencil_distribution_problem_l375_37547


namespace friend_lunch_cost_l375_37554

theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 15 → difference = 1 → friend_cost = (total + difference) / 2 → friend_cost = 8 := by
  sorry

end friend_lunch_cost_l375_37554


namespace one_fifth_greater_than_decimal_l375_37540

theorem one_fifth_greater_than_decimal : 1/5 = 0.20000001 + 1/(5*10^8) := by
  sorry

end one_fifth_greater_than_decimal_l375_37540


namespace complex_equation_solution_l375_37581

theorem complex_equation_solution (z : ℂ) (h : 10 * Complex.normSq z = 3 * Complex.normSq (z + 3) + Complex.normSq (z^2 - 1) + 40) :
  z + 9 / z = (9 + Real.sqrt 61) / 2 ∨ z + 9 / z = (9 - Real.sqrt 61) / 2 := by
  sorry

end complex_equation_solution_l375_37581


namespace inequality_proof_l375_37508

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := by
  sorry

end inequality_proof_l375_37508


namespace min_reciprocal_sum_l375_37516

theorem min_reciprocal_sum (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x + y + z = 2) :
  (1/x + 1/y + 1/z) ≥ 4.5 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ x' + y' + z' = 2 ∧ 1/x' + 1/y' + 1/z' = 4.5 := by
  sorry

end min_reciprocal_sum_l375_37516


namespace ratio_of_fractions_l375_37553

theorem ratio_of_fractions (x y : ℝ) (h1 : 5 * x = 6 * y) (h2 : x * y ≠ 0) :
  (1 / 3 * x) / (1 / 5 * y) = 2 := by
sorry

end ratio_of_fractions_l375_37553


namespace balloon_distribution_difference_l375_37530

/-- Represents the number of balloons of each color brought by a person -/
structure Balloons :=
  (red : ℕ)
  (blue : ℕ)
  (green : ℕ)

/-- Calculates the total number of balloons -/
def totalBalloons (b : Balloons) : ℕ := b.red + b.blue + b.green

theorem balloon_distribution_difference :
  let allan_brought := Balloons.mk 150 75 30
  let jake_brought := Balloons.mk 100 50 45
  let allan_forgot := 25
  let allan_distributed := totalBalloons { red := allan_brought.red,
                                           blue := allan_brought.blue - allan_forgot,
                                           green := allan_brought.green }
  let jake_distributed := totalBalloons jake_brought
  allan_distributed - jake_distributed = 35 := by sorry

end balloon_distribution_difference_l375_37530


namespace seven_eighths_of_64_l375_37549

theorem seven_eighths_of_64 : (7 / 8 : ℚ) * 64 = 56 := by
  sorry

end seven_eighths_of_64_l375_37549


namespace line_intersecting_parabola_l375_37566

/-- The equation of a line that intersects a parabola at two points 8 units apart vertically -/
theorem line_intersecting_parabola (m b : ℝ) (h1 : b ≠ 0) :
  (∃ k : ℝ, abs ((k^2 + 4*k + 4) - (m*k + b)) = 8) →
  (9 = 2*m + b) →
  (m = 2 ∧ b = 5) :=
by sorry

end line_intersecting_parabola_l375_37566


namespace quadratic_problem_l375_37546

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_problem (a b c : ℝ) (f : ℝ → ℝ) (h_f : f = QuadraticFunction a b c) :
  (∀ x, f x ≤ 4) ∧ -- The maximum value of f(x) is 4
  (f 2 = 4) ∧ -- The maximum occurs at x = 2
  (f 0 = -20) ∧ -- The graph passes through (0, -20)
  (∃ m, f 5 = m) -- The graph passes through (5, m)
  → f 5 = -50 := by sorry

end quadratic_problem_l375_37546


namespace last_number_not_one_l375_37522

def board_sum : ℕ := (2012 * 2013) / 2

theorem last_number_not_one :
  ∀ (operations : ℕ) (final_number : ℕ),
    (operations < 2011 → final_number ≠ 1) ∧
    (operations = 2011 → final_number % 2 = 0) :=
by sorry

end last_number_not_one_l375_37522


namespace student_count_l375_37561

theorem student_count (bags : ℕ) (nuts_per_bag : ℕ) (nuts_per_student : ℕ) : 
  bags = 65 → nuts_per_bag = 15 → nuts_per_student = 75 → 
  (bags * nuts_per_bag) / nuts_per_student = 13 := by
  sorry

end student_count_l375_37561


namespace prime_sequence_l375_37575

theorem prime_sequence (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) := by
sorry

end prime_sequence_l375_37575


namespace solve_for_y_l375_37500

theorem solve_for_y (x y : ℚ) (h1 : x - y = 20) (h2 : 3 * (x + y) = 15) : y = -15/2 := by
  sorry

end solve_for_y_l375_37500


namespace quadratic_inequality_range_l375_37574

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 < 0) → (a < -2 ∨ a > 2) := by sorry

end quadratic_inequality_range_l375_37574


namespace not_strictly_monotone_sequence_l375_37504

/-- d(k) denotes the number of natural divisors of a natural number k -/
def d (k : ℕ) : ℕ := (Finset.filter (· ∣ k) (Finset.range (k + 1))).card

/-- The sequence {d(n^2+1)}_{n=n_0}^∞ is not strictly monotone -/
theorem not_strictly_monotone_sequence (n_0 : ℕ) :
  ∃ m n : ℕ, m > n ∧ n ≥ n_0 ∧ d (m^2 + 1) ≤ d (n^2 + 1) :=
sorry

end not_strictly_monotone_sequence_l375_37504


namespace negation_of_existential_absolute_value_l375_37595

theorem negation_of_existential_absolute_value (x : ℝ) :
  (¬ ∃ x : ℝ, |x| < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) :=
by sorry

end negation_of_existential_absolute_value_l375_37595


namespace line_in_first_third_quadrants_positive_slope_l375_37523

/-- A line passing through the first and third quadrants -/
structure LineInFirstThirdQuadrants where
  k : ℝ
  k_neq_zero : k ≠ 0
  passes_through_first_third : ∀ x y : ℝ, y = k * x → 
    ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0))

/-- Theorem: If a line y = kx passes through the first and third quadrants, then k > 0 -/
theorem line_in_first_third_quadrants_positive_slope 
  (line : LineInFirstThirdQuadrants) : line.k > 0 := by
  sorry

end line_in_first_third_quadrants_positive_slope_l375_37523


namespace opposite_of_negative_fraction_l375_37557

theorem opposite_of_negative_fraction :
  -(-(1 / 2023)) = 1 / 2023 := by sorry

end opposite_of_negative_fraction_l375_37557


namespace sum_of_squares_and_square_of_sum_l375_37585

theorem sum_of_squares_and_square_of_sum : (3 + 9)^2 + (3^2 + 9^2) = 234 := by
  sorry

end sum_of_squares_and_square_of_sum_l375_37585


namespace log_inequality_l375_37567

theorem log_inequality (x : ℝ) : 
  0 < x → x < 4 → (Real.log x / Real.log 9 ≥ (Real.log (Real.sqrt (1 - x / 4)) / Real.log 3)^2 ↔ x = 2 ∨ (4/5 ≤ x ∧ x < 4)) :=
by sorry

end log_inequality_l375_37567


namespace function_characterization_l375_37527

theorem function_characterization 
  (f : ℕ → ℕ) 
  (h1 : ∀ x y : ℕ, (x + y) ∣ (f x + f y))
  (h2 : ∀ x : ℕ, x ≥ 1395 → x^3 ≥ 2 * f x) :
  ∃ k : ℕ, k ≤ 1395^2 / 2 ∧ ∀ n : ℕ, f n = k * n :=
sorry

end function_characterization_l375_37527


namespace class_average_theorem_l375_37542

theorem class_average_theorem (group1_percent : Real) (group1_avg : Real)
                              (group2_percent : Real) (group2_avg : Real)
                              (group3_percent : Real) (group3_avg : Real) :
  group1_percent = 0.45 →
  group1_avg = 0.95 →
  group2_percent = 0.50 →
  group2_avg = 0.78 →
  group3_percent = 1 - group1_percent - group2_percent →
  group3_avg = 0.60 →
  round ((group1_percent * group1_avg + group2_percent * group2_avg + group3_percent * group3_avg) * 100) = 85 :=
by
  sorry

#check class_average_theorem

end class_average_theorem_l375_37542


namespace perpendicular_vectors_l375_37521

/-- Given vectors a and b in ℝ², prove that if (a + kb) ⊥ (a - kb), then k = ±√5 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (2, -1))
  (h2 : b = (-Real.sqrt 3 / 2, -1 / 2))
  (h3 : (a.1 + k * b.1, a.2 + k * b.2) • (a.1 - k * b.1, a.2 - k * b.2) = 0) :
  k = Real.sqrt 5 ∨ k = -Real.sqrt 5 := by
  sorry

end perpendicular_vectors_l375_37521


namespace fraction_simplification_l375_37569

theorem fraction_simplification : (3^2016 + 3^2014) / (3^2016 - 3^2014) = 5/4 := by
  sorry

end fraction_simplification_l375_37569


namespace length_of_DE_l375_37538

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the angles
variable (angle_BAE angle_CBE angle_DCE : ℝ)

-- Define the side lengths
variable (AE AB BC CD : ℝ)

-- Define t
variable (t : ℝ)

-- State the theorem
theorem length_of_DE (h1 : angle_BAE = 90) (h2 : angle_CBE = 90) (h3 : angle_DCE = 90)
                     (h4 : AE = Real.sqrt 5) (h5 : AB = Real.sqrt 4) (h6 : BC = Real.sqrt 3)
                     (h7 : CD = Real.sqrt t) (h8 : t = 4) :
  Real.sqrt ((CD^2) + (Real.sqrt ((BC^2) + (Real.sqrt (AB^2 + AE^2))^2))^2) = 4 := by
  sorry

end length_of_DE_l375_37538


namespace associates_hired_l375_37541

theorem associates_hired (initial_partners initial_associates : ℕ) 
  (new_associates : ℕ) (hired_associates : ℕ) : 
  initial_partners = 18 →
  initial_partners * 63 = 2 * initial_associates →
  (initial_partners) * 34 = (initial_associates + hired_associates) →
  hired_associates = 45 := by sorry

end associates_hired_l375_37541


namespace radiator_water_fraction_l375_37578

/-- Calculates the fraction of water remaining in a radiator after a given number of replacements -/
def waterFraction (initialVolume : ℚ) (replacementVolume : ℚ) (numReplacements : ℕ) : ℚ :=
  (1 - replacementVolume / initialVolume) ^ numReplacements

theorem radiator_water_fraction :
  let initialVolume : ℚ := 25
  let replacementVolume : ℚ := 5
  let numReplacements : ℕ := 5
  waterFraction initialVolume replacementVolume numReplacements = 1024 / 3125 := by
  sorry

#eval waterFraction 25 5 5

end radiator_water_fraction_l375_37578


namespace dryer_weight_l375_37501

def bridge_weight_limit : ℕ := 20000
def empty_truck_weight : ℕ := 12000
def soda_crates : ℕ := 20
def soda_crate_weight : ℕ := 50
def dryer_count : ℕ := 3
def loaded_truck_weight : ℕ := 24000

theorem dryer_weight (h1 : bridge_weight_limit = 20000)
                     (h2 : empty_truck_weight = 12000)
                     (h3 : soda_crates = 20)
                     (h4 : soda_crate_weight = 50)
                     (h5 : dryer_count = 3)
                     (h6 : loaded_truck_weight = 24000) :
  let soda_weight := soda_crates * soda_crate_weight
  let produce_weight := 2 * soda_weight
  let truck_soda_produce_weight := empty_truck_weight + soda_weight + produce_weight
  let total_dryer_weight := loaded_truck_weight - truck_soda_produce_weight
  total_dryer_weight / dryer_count = 3000 := by
sorry

end dryer_weight_l375_37501


namespace product_of_roots_l375_37589

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 17 → ∃ y : ℝ, (x + 3) * (x - 4) = 17 ∧ (x * y = -29) := by
  sorry

end product_of_roots_l375_37589


namespace unbounded_solution_set_l375_37559

/-- The set of points (x, y) satisfying the given system of inequalities is unbounded -/
theorem unbounded_solution_set :
  ∃ (S : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x, y) ∈ S ↔ 
      ((abs x + x)^2 + (abs y + y)^2 ≤ 4 ∧ 3*y + x ≤ 0)) ∧
    ¬(∃ (M : ℝ), ∀ (p : ℝ × ℝ), p ∈ S → ‖p‖ ≤ M) :=
by sorry

end unbounded_solution_set_l375_37559


namespace total_animals_on_yacht_l375_37571

theorem total_animals_on_yacht (cows foxes zebras sheep : ℕ) : 
  cows = 20 → 
  foxes = 15 → 
  zebras = 3 * foxes → 
  sheep = 20 → 
  cows + foxes + zebras + sheep = 100 := by
  sorry

end total_animals_on_yacht_l375_37571


namespace new_sequence_common_difference_l375_37535

theorem new_sequence_common_difference 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h : ∀ n : ℕ, a (n + 1) = a n + d) :
  let b : ℕ → ℝ := λ n => a n + a (n + 3)
  ∀ n : ℕ, b (n + 1) = b n + 2 * d :=
by sorry

end new_sequence_common_difference_l375_37535


namespace cones_problem_l375_37544

-- Define the radii of the three cones
def r1 (r : ℝ) : ℝ := 2 * r
def r2 (r : ℝ) : ℝ := 3 * r
def r3 (r : ℝ) : ℝ := 10 * r

-- Define the radius of the smaller base of the truncated cone
def R : ℝ := 15

-- Define the distances between the centers of the bases of cones
def d12 (r : ℝ) : ℝ := 5 * r
def d13 (r : ℝ) : ℝ := 12 * r
def d23 (r : ℝ) : ℝ := 13 * r

-- Define the distances from the center of the truncated cone to the centers of the other cones
def dC1 (r : ℝ) : ℝ := r1 r + R
def dC2 (r : ℝ) : ℝ := r2 r + R
def dC3 (r : ℝ) : ℝ := r3 r + R

-- Theorem statement
theorem cones_problem (r : ℝ) (h_pos : r > 0) :
  225 * (r1 r + R)^2 = (30 * r - 10 * R)^2 + (30 * r - 3 * R)^2 → r = 29 := by
  sorry

end cones_problem_l375_37544


namespace two_plus_three_equals_twentysix_l375_37563

/-- Defines the sequence operation for two consecutive terms -/
def sequenceOperation (a b : ℕ) : ℕ := (a + b)^2 + 1

/-- Theorem stating that 2 + 3 in the given sequence equals 26 -/
theorem two_plus_three_equals_twentysix :
  sequenceOperation 2 3 = 26 := by
  sorry

end two_plus_three_equals_twentysix_l375_37563


namespace f_has_extrema_l375_37594

/-- The function f(x) = 2 - x^2 - x^3 -/
def f (x : ℝ) : ℝ := 2 - x^2 - x^3

/-- Theorem stating that f has both a maximum and a minimum value -/
theorem f_has_extrema : 
  (∃ a : ℝ, ∀ x : ℝ, f x ≤ f a) ∧ (∃ b : ℝ, ∀ x : ℝ, f x ≥ f b) :=
sorry

end f_has_extrema_l375_37594


namespace inequality_proof_l375_37590

theorem inequality_proof (x : ℝ) (n : ℕ) (a : ℝ) 
  (h1 : x > 0) (h2 : n > 0) (h3 : x + a / x^n ≥ n + 1) : 
  a = n^n := by
sorry

end inequality_proof_l375_37590


namespace log_equation_solution_l375_37558

theorem log_equation_solution : 
  ∃ y : ℝ, (Real.log y - 3 * Real.log 5 = -3) ∧ (y = 0.125) :=
by sorry

end log_equation_solution_l375_37558


namespace system_solution_l375_37550

theorem system_solution (x y m : ℝ) 
  (eq1 : 2*x + y = 1) 
  (eq2 : x + 2*y = 2) 
  (eq3 : x + y = 2*m - 1) : 
  m = 1 := by sorry

end system_solution_l375_37550


namespace dave_apps_left_l375_37520

/-- The number of files Dave has left on his phone -/
def files_left : ℕ := 5

/-- The difference between the number of apps and files Dave has left -/
def app_file_difference : ℕ := 7

/-- The number of apps Dave has left on his phone -/
def apps_left : ℕ := files_left + app_file_difference

theorem dave_apps_left : apps_left = 12 := by
  sorry

end dave_apps_left_l375_37520


namespace grandma_backpacks_l375_37586

def backpack_problem (original_price : ℝ) (discount_rate : ℝ) (monogram_cost : ℝ) (total_cost : ℝ) : Prop :=
  let discounted_price := original_price * (1 - discount_rate)
  let final_price := discounted_price + monogram_cost
  let num_grandchildren := total_cost / final_price
  num_grandchildren = 5

theorem grandma_backpacks :
  backpack_problem 20 0.2 12 140 := by
  sorry

end grandma_backpacks_l375_37586


namespace p_squared_plus_eight_composite_l375_37517

theorem p_squared_plus_eight_composite (p : ℕ) (h_prime : Nat.Prime p) (h_not_three : p ≠ 3) :
  ¬(Nat.Prime (p^2 + 8)) := by
  sorry

end p_squared_plus_eight_composite_l375_37517


namespace intersection_of_A_and_B_l375_37502

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (2 - x)}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | -1 ≤ x ∧ x < 2} := by
  sorry

end intersection_of_A_and_B_l375_37502


namespace unique_solution_l375_37534

/-- Represents the intersection point of two lines --/
structure IntersectionPoint where
  x : ℤ
  y : ℤ

/-- Checks if a given point satisfies both line equations --/
def is_valid_intersection (m : ℕ) (p : IntersectionPoint) : Prop :=
  13 * p.x + 11 * p.y = 700 ∧ p.y = m * p.x - 1

/-- Main theorem: m = 6 is the only solution --/
theorem unique_solution : 
  ∃! (m : ℕ), ∃ (p : IntersectionPoint), is_valid_intersection m p :=
sorry

end unique_solution_l375_37534


namespace conic_is_hyperbola_l375_37512

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop := x^2 + 2*x - 8*y^2 = 0

/-- Definition of a hyperbola -/
def is_hyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b h k : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  ∀ x y, f x y ↔ (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1

/-- Theorem stating that the given equation represents a hyperbola -/
theorem conic_is_hyperbola : is_hyperbola conic_equation := by
  sorry

end conic_is_hyperbola_l375_37512


namespace opposite_of_negative_2023_l375_37598

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end opposite_of_negative_2023_l375_37598


namespace ten_percent_relation_l375_37572

/-- If 10% of s is equal to t, then s equals 10t -/
theorem ten_percent_relation (s t : ℝ) (h : (10 : ℝ) / 100 * s = t) : s = 10 * t := by
  sorry

end ten_percent_relation_l375_37572


namespace halfway_fraction_l375_37577

theorem halfway_fraction :
  ∃ (n d : ℕ), d ≠ 0 ∧ (n : ℚ) / d > 1 / 2 ∧
  (n : ℚ) / d = (3 / 4 + 5 / 7) / 2 ∧
  n = 41 ∧ d = 56 := by
  sorry

end halfway_fraction_l375_37577


namespace personal_income_tax_example_l375_37579

/-- Calculate the personal income tax for a citizen given their salary and prize information --/
def personal_income_tax (salary_jan_jun : ℕ) (salary_jul_dec : ℕ) (prize_value : ℕ) : ℕ :=
  let salary_tax_rate : ℚ := 13 / 100
  let prize_tax_rate : ℚ := 35 / 100
  let non_taxable_prize : ℕ := 4000
  let total_salary : ℕ := salary_jan_jun * 6 + salary_jul_dec * 6
  let salary_tax : ℕ := (total_salary * salary_tax_rate).floor.toNat
  let taxable_prize : ℕ := max (prize_value - non_taxable_prize) 0
  let prize_tax : ℕ := (taxable_prize * prize_tax_rate).floor.toNat
  salary_tax + prize_tax

/-- Theorem stating that the personal income tax for the given scenario is 39540 rubles --/
theorem personal_income_tax_example : 
  personal_income_tax 23000 25000 10000 = 39540 := by
  sorry

end personal_income_tax_example_l375_37579


namespace cookie_brownie_difference_l375_37507

/-- Represents the daily cookie and brownie activity -/
structure DailyActivity where
  eaten_cookies : ℕ
  eaten_brownies : ℕ
  baked_cookies : ℕ
  baked_brownies : ℕ

/-- Calculates the final number of cookies and brownies after a week -/
def final_counts (initial_cookies : ℕ) (initial_brownies : ℕ) (activities : List DailyActivity) : ℕ × ℕ :=
  activities.foldl
    (fun (acc : ℕ × ℕ) (day : DailyActivity) =>
      (acc.1 - day.eaten_cookies + day.baked_cookies,
       acc.2 - day.eaten_brownies + day.baked_brownies))
    (initial_cookies, initial_brownies)

/-- The theorem to be proved -/
theorem cookie_brownie_difference :
  let initial_cookies := 60
  let initial_brownies := 10
  let activities : List DailyActivity := [
    ⟨2, 1, 10, 0⟩,
    ⟨4, 2, 0, 4⟩,
    ⟨3, 1, 5, 2⟩,
    ⟨5, 1, 0, 0⟩,
    ⟨4, 3, 8, 0⟩,
    ⟨3, 2, 0, 1⟩,
    ⟨2, 1, 0, 5⟩
  ]
  let (final_cookies, final_brownies) := final_counts initial_cookies initial_brownies activities
  final_cookies - final_brownies = 49 := by
  sorry

end cookie_brownie_difference_l375_37507


namespace paths_on_4x10_grid_with_forbidden_segments_l375_37529

/-- Represents a grid with forbidden segments -/
structure Grid where
  height : ℕ
  width : ℕ
  forbidden_segments : ℕ

/-- Calculates the number of paths on a grid with forbidden segments -/
def count_paths (g : Grid) : ℕ :=
  let total_paths := Nat.choose (g.height + g.width) g.height
  let forbidden_paths := g.forbidden_segments * (Nat.choose (g.height + g.width / 2 - 2) (g.height - 2) * Nat.choose (g.width / 2 + 2) 2)
  total_paths - forbidden_paths

/-- Theorem stating the number of paths on a 4x10 grid with two forbidden segments -/
theorem paths_on_4x10_grid_with_forbidden_segments :
  count_paths { height := 4, width := 10, forbidden_segments := 2 } = 161 := by
  sorry

end paths_on_4x10_grid_with_forbidden_segments_l375_37529


namespace angle_with_supplement_four_times_complement_l375_37597

theorem angle_with_supplement_four_times_complement (x : ℝ) :
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end angle_with_supplement_four_times_complement_l375_37597


namespace same_height_antonio_maria_l375_37560

-- Define the type for height comparisons
inductive HeightComparison
  | Taller : HeightComparison
  | Shorter : HeightComparison
  | Same : HeightComparison

-- Define the siblings
inductive Sibling
  | Luiza : Sibling
  | Maria : Sibling
  | Antonio : Sibling
  | Julio : Sibling

-- Define the height comparison function
def compareHeight : Sibling → Sibling → HeightComparison := sorry

-- State the theorem
theorem same_height_antonio_maria :
  (compareHeight Sibling.Luiza Sibling.Antonio = HeightComparison.Taller) →
  (compareHeight Sibling.Antonio Sibling.Julio = HeightComparison.Taller) →
  (compareHeight Sibling.Maria Sibling.Luiza = HeightComparison.Shorter) →
  (compareHeight Sibling.Julio Sibling.Maria = HeightComparison.Shorter) →
  (compareHeight Sibling.Antonio Sibling.Maria = HeightComparison.Same) :=
by
  sorry

end same_height_antonio_maria_l375_37560


namespace even_sum_sufficient_not_necessary_l375_37599

/-- A function is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_sum_sufficient_not_necessary :
  (∀ f g : ℝ → ℝ, IsEven f ∧ IsEven g → IsEven (fun x ↦ f x + g x)) ∧
  (∃ f g : ℝ → ℝ, ¬(IsEven f ∧ IsEven g) ∧ IsEven (fun x ↦ f x + g x)) :=
by sorry

end even_sum_sufficient_not_necessary_l375_37599


namespace root_sum_theorem_l375_37526

theorem root_sum_theorem (a b c : ℝ) : 
  a^3 - 24*a^2 + 50*a - 42 = 0 →
  b^3 - 24*b^2 + 50*b - 42 = 0 →
  c^3 - 24*c^2 + 50*c - 42 = 0 →
  (a / (1/a + b*c)) + (b / (1/b + c*a)) + (c / (1/c + a*b)) = 476/43 := by
sorry

end root_sum_theorem_l375_37526


namespace trigonometric_sum_divisibility_l375_37583

theorem trigonometric_sum_divisibility (n : ℕ) :
  ∃ k : ℤ, (2 * Real.sin (π / 7 : ℝ))^(2*n) + 
           (2 * Real.sin (2*π / 7 : ℝ))^(2*n) + 
           (2 * Real.sin (3*π / 7 : ℝ))^(2*n) = 
           k * (7 : ℝ)^(Int.floor (n / 3 : ℝ)) := by
  sorry

end trigonometric_sum_divisibility_l375_37583


namespace quadratic_equation_roots_l375_37505

theorem quadratic_equation_roots (m n : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + n = 0 ∧ x₂^2 + m*x₂ + n = 0) →
  (n = 3 - m ∧ (∀ x : ℝ, x^2 + m*x + n = 0 → x < 0) → 2 ≤ m ∧ m < 3) ∧
  (∃ t : ℝ, ∀ m n : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + n = 0 ∧ x₂^2 + m*x₂ + n = 0) →
    t ≤ (m-1)^2 + (n-1)^2 + (m-n)^2 ∧
    t = 9/8 ∧
    ∀ t' : ℝ, (∀ m n : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + n = 0 ∧ x₂^2 + m*x₂ + n = 0) →
      t' ≤ (m-1)^2 + (n-1)^2 + (m-n)^2) → t' ≤ t) :=
by sorry

end quadratic_equation_roots_l375_37505


namespace leisurely_morning_time_l375_37596

/-- Represents the time taken for each part of Aiden's morning routine -/
structure MorningRoutine where
  prep : ℝ  -- Preparation time
  bus : ℝ   -- Bus ride time
  walk : ℝ  -- Walking time

/-- Calculates the total time for a given morning routine -/
def totalTime (r : MorningRoutine) : ℝ := r.prep + r.bus + r.walk

/-- Represents the conditions given in the problem -/
axiom typical_morning : ∃ r : MorningRoutine, totalTime r = 120

axiom rushed_morning : ∃ r : MorningRoutine, 
  0.5 * r.prep + 1.25 * r.bus + 0.5 * r.walk = 96

/-- Theorem stating the time taken on the leisurely morning -/
theorem leisurely_morning_time : 
  ∀ r : MorningRoutine, 
  totalTime r = 120 → 
  0.5 * r.prep + 1.25 * r.bus + 0.5 * r.walk = 96 → 
  1.25 * r.prep + 0.75 * r.bus + 1.25 * r.walk = 126 := by
  sorry

end leisurely_morning_time_l375_37596


namespace canteen_leak_rate_l375_37509

/-- Proves that the canteen leak rate is 1 cup per hour given the hiking conditions -/
theorem canteen_leak_rate
  (total_distance : ℝ)
  (initial_water : ℝ)
  (hike_duration : ℝ)
  (remaining_water : ℝ)
  (last_mile_consumption : ℝ)
  (first_three_miles_rate : ℝ)
  (h1 : total_distance = 4)
  (h2 : initial_water = 6)
  (h3 : hike_duration = 2)
  (h4 : remaining_water = 1)
  (h5 : last_mile_consumption = 1)
  (h6 : first_three_miles_rate = 0.6666666666666666)
  : (initial_water - remaining_water - (first_three_miles_rate * 3 + last_mile_consumption)) / hike_duration = 1 := by
  sorry

#check canteen_leak_rate

end canteen_leak_rate_l375_37509


namespace fair_coin_expectation_l375_37556

/-- A fair coin is a coin with probability 1/2 for both heads and tails -/
def fairCoin (p : ℝ) : Prop := p = 1/2

/-- The expected value of heads for a single toss of a fair coin -/
def expectedValueSingleToss (p : ℝ) (h : fairCoin p) : ℝ := p

/-- The number of tosses -/
def numTosses : ℕ := 5

/-- The mathematical expectation of heads for multiple tosses of a fair coin -/
def expectedValueMultipleTosses (p : ℝ) (h : fairCoin p) : ℝ :=
  (expectedValueSingleToss p h) * numTosses

theorem fair_coin_expectation (p : ℝ) (h : fairCoin p) :
  expectedValueMultipleTosses p h = 5/2 := by sorry

end fair_coin_expectation_l375_37556


namespace train_speed_problem_l375_37510

/-- The speed of train B given the conditions of the problem -/
theorem train_speed_problem (length_A length_B : ℝ) (speed_A : ℝ) (crossing_time : ℝ) :
  length_A = 150 ∧ 
  length_B = 150 ∧ 
  speed_A = 54 ∧ 
  crossing_time = 12 →
  (length_A + length_B) / crossing_time * 3.6 - speed_A = 36 := by
  sorry

#check train_speed_problem

end train_speed_problem_l375_37510


namespace tim_books_l375_37543

theorem tim_books (sam_books : ℕ) (total_books : ℕ) (h1 : sam_books = 52) (h2 : total_books = 96) :
  total_books - sam_books = 44 := by
  sorry

end tim_books_l375_37543


namespace unique_zero_point_between_consecutive_integers_l375_37593

open Real

noncomputable def f (a x : ℝ) : ℝ := a * (x^2 + 2/x) - log x

theorem unique_zero_point_between_consecutive_integers (a : ℝ) (h : a > 0) :
  ∃ (x₀ m n : ℝ), 
    (∀ x ≠ x₀, f a x ≠ 0) ∧ 
    (f a x₀ = 0) ∧
    (m < x₀ ∧ x₀ < n) ∧
    (n = m + 1) ∧
    (m + n = 5) := by
  sorry

end unique_zero_point_between_consecutive_integers_l375_37593


namespace expression_equality_l375_37515

theorem expression_equality (θ : Real) 
  (h1 : π / 4 < θ) (h2 : θ < π / 2) : 
  2 * Real.cos θ + Real.sqrt (1 - 2 * Real.sin (π - θ) * Real.cos θ) = Real.sin θ + Real.cos θ := by
  sorry

end expression_equality_l375_37515


namespace sum_of_four_numbers_l375_37519

theorem sum_of_four_numbers : 1357 + 3571 + 5713 + 7135 = 17776 := by
  sorry

end sum_of_four_numbers_l375_37519


namespace rowing_distance_problem_l375_37545

/-- Proves that given a man who can row 7.5 km/hr in still water, in a river flowing at 1.5 km/hr,
    if it takes him 50 minutes to row to a place and back, the distance to that place is 3 km. -/
theorem rowing_distance_problem (man_speed : ℝ) (river_speed : ℝ) (total_time : ℝ) :
  man_speed = 7.5 →
  river_speed = 1.5 →
  total_time = 50 / 60 →
  ∃ (distance : ℝ),
    distance / (man_speed - river_speed) + distance / (man_speed + river_speed) = total_time ∧
    distance = 3 :=
by sorry

end rowing_distance_problem_l375_37545


namespace equation_solution_l375_37548

/-- The overall substitution method for solving quadratic equations -/
def overall_substitution_method (a b c : ℝ) : Set ℝ :=
  { x | ∃ y, y^2 + b*y + c = 0 ∧ a*x + b = y }

/-- The equation (2x-5)^2 - 2(2x-5) - 3 = 0 has solutions x₁ = 2 and x₂ = 4 -/
theorem equation_solution : 
  overall_substitution_method 2 (-5) (-3) = {2, 4} := by
  sorry

#check equation_solution

end equation_solution_l375_37548


namespace inequality_chain_l375_37580

theorem inequality_chain (x : ℝ) 
  (h1 : 0 < x) (h2 : x < 1) 
  (a b c : ℝ) 
  (ha : a = x^2) 
  (hb : b = 1/x) 
  (hc : c = Real.sqrt x) : 
  b > c ∧ c > a := by sorry

end inequality_chain_l375_37580


namespace smallest_x_absolute_value_equation_l375_37536

theorem smallest_x_absolute_value_equation : 
  ∃ x : ℝ, x = -8.6 ∧ ∀ y : ℝ, |5 * y + 9| = 34 → y ≥ x := by
  sorry

end smallest_x_absolute_value_equation_l375_37536


namespace triangle_shape_l375_37588

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the condition
def satisfiesCondition (t : Triangle) : Prop :=
  (Real.cos t.A) / (Real.cos t.B) = t.b / t.a

-- Define isosceles triangle
def isIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.C = t.A

-- Define right triangle
def isRight (t : Triangle) : Prop :=
  t.A = Real.pi / 2 ∨ t.B = Real.pi / 2 ∨ t.C = Real.pi / 2

-- Theorem statement
theorem triangle_shape (t : Triangle) :
  satisfiesCondition t → isIsosceles t ∨ isRight t :=
by
  sorry


end triangle_shape_l375_37588


namespace current_calculation_l375_37562

-- Define the variables and their types
variable (Q I R t : ℝ)

-- Define the theorem
theorem current_calculation 
  (heat_equation : Q = I^2 * R * t)
  (resistance : R = 5)
  (heat_generated : Q = 30)
  (time : t = 1) :
  I = Real.sqrt 6 := by sorry

end current_calculation_l375_37562


namespace odd_digits_base4_157_l375_37506

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Theorem: The number of odd digits in the base-4 representation of 157₁₀ is 2 -/
theorem odd_digits_base4_157 : countOddDigits (toBase4 157) = 2 := by
  sorry

end odd_digits_base4_157_l375_37506


namespace cost_difference_l375_37576

-- Define the parameters
def batches : ℕ := 4
def ounces_per_batch : ℕ := 12
def blueberry_carton_size : ℕ := 6
def raspberry_carton_size : ℕ := 8
def blueberry_price : ℚ := 5
def raspberry_price : ℚ := 3

-- Define the total ounces needed
def total_ounces : ℕ := batches * ounces_per_batch

-- Define the number of cartons needed for each fruit
def blueberry_cartons : ℕ := (total_ounces + blueberry_carton_size - 1) / blueberry_carton_size
def raspberry_cartons : ℕ := (total_ounces + raspberry_carton_size - 1) / raspberry_carton_size

-- Define the total cost for each fruit
def blueberry_cost : ℚ := blueberry_price * blueberry_cartons
def raspberry_cost : ℚ := raspberry_price * raspberry_cartons

-- Theorem to prove
theorem cost_difference : blueberry_cost - raspberry_cost = 22 := by
  sorry

end cost_difference_l375_37576


namespace triangle_sum_equals_22_l375_37513

/-- The triangle operation defined as 2a - b + c -/
def triangle_op (a b c : ℤ) : ℤ := 2*a - b + c

/-- The vertices of the first triangle -/
def triangle1 : List ℤ := [3, 7, 5]

/-- The vertices of the second triangle -/
def triangle2 : List ℤ := [6, 2, 8]

theorem triangle_sum_equals_22 : 
  triangle_op triangle1[0] triangle1[1] triangle1[2] + 
  triangle_op triangle2[0] triangle2[1] triangle2[2] = 22 := by
sorry

end triangle_sum_equals_22_l375_37513


namespace fraction_of_complex_l375_37582

def complex_i : ℂ := Complex.I

theorem fraction_of_complex (z : ℂ) (h : z = 1 + complex_i) : 2 / z = 1 - complex_i := by
  sorry

end fraction_of_complex_l375_37582


namespace min_value_abc_l375_37531

theorem min_value_abc (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + 2*a*b + 2*a*c + 4*b*c = 16) :
  ∃ m : ℝ, ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
  x^2 + 2*x*y + 2*x*z + 4*y*z = 16 → m ≤ x + y + z :=
sorry

end min_value_abc_l375_37531


namespace arithmetic_sequence_properties_l375_37564

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  h1 : a 7 = 1  -- 7th term is 1
  h2 : S 4 = -32  -- Sum of first 4 terms is -32
  h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1  -- Constant difference property

/-- Properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 2 * n - 13) ∧
  (∀ n : ℕ, seq.S n = (n - 6)^2 - 36) ∧
  (∀ n : ℕ, seq.S n ≥ -36) ∧
  (∃ n : ℕ, seq.S n = -36) :=
by sorry

end arithmetic_sequence_properties_l375_37564


namespace number_divided_by_004_l375_37552

theorem number_divided_by_004 :
  ∃ x : ℝ, x / 0.04 = 100.9 ∧ x = 4.036 := by
  sorry

end number_divided_by_004_l375_37552


namespace expression_evaluation_l375_37555

theorem expression_evaluation : 5 * 7 + 9 * 4 - (15 / 3)^2 = 46 := by
  sorry

end expression_evaluation_l375_37555
