import Mathlib

namespace distribute_5_balls_4_boxes_l1537_153787

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_5_balls_4_boxes : distribute_balls 5 4 = 56 := by sorry

end distribute_5_balls_4_boxes_l1537_153787


namespace soda_cans_purchased_l1537_153734

/-- The number of cans of soda that can be purchased for a given amount of money -/
theorem soda_cans_purchased (S Q D : ℚ) (h1 : S > 0) (h2 : Q > 0) (h3 : D ≥ 0) :
  let cans_per_quarter := S / Q
  let quarters_per_dollar := 4
  let cans_per_dollar := cans_per_quarter * quarters_per_dollar
  cans_per_dollar * D = 4 * D * S / Q :=
by sorry

end soda_cans_purchased_l1537_153734


namespace pencils_per_box_l1537_153796

theorem pencils_per_box (total_pencils : ℕ) (num_boxes : ℕ) (pencils_per_box : ℕ) 
  (h1 : total_pencils = 27)
  (h2 : num_boxes = 3)
  (h3 : total_pencils = num_boxes * pencils_per_box) :
  pencils_per_box = 9 := by
  sorry

end pencils_per_box_l1537_153796


namespace parking_lot_problem_l1537_153738

theorem parking_lot_problem (total_vehicles : ℕ) (total_wheels : ℕ) 
  (h1 : total_vehicles = 24) 
  (h2 : total_wheels = 86) : 
  ∃ (cars motorcycles : ℕ), 
    cars + motorcycles = total_vehicles ∧ 
    4 * cars + 3 * motorcycles = total_wheels ∧ 
    motorcycles = 10 := by
  sorry

end parking_lot_problem_l1537_153738


namespace certain_number_problem_l1537_153745

theorem certain_number_problem (x : ℝ) (certain_number : ℝ) 
  (h1 : certain_number * x = 675)
  (h2 : x = 27) : 
  certain_number = 25 := by
  sorry

end certain_number_problem_l1537_153745


namespace sequence_problem_l1537_153719

theorem sequence_problem (m : ℕ+) (a : ℕ → ℝ) 
  (h0 : a 0 = 37)
  (h1 : a 1 = 72)
  (hm : a m = 0)
  (h_rec : ∀ k : ℕ, 1 ≤ k → k < m → a (k + 1) = a (k - 1) - 3 / a k) :
  m = 889 := by
  sorry

end sequence_problem_l1537_153719


namespace equilateral_triangle_most_stable_l1537_153794

-- Define the shapes
inductive Shape
| EquilateralTriangle
| Square
| Parallelogram
| Trapezoid

-- Define stability as a function of shape properties
def stability (s : Shape) : ℝ :=
  match s with
  | Shape.EquilateralTriangle => 1
  | Shape.Square => 0.9
  | Shape.Parallelogram => 0.7
  | Shape.Trapezoid => 0.5

-- Define a predicate for being the most stable
def is_most_stable (s : Shape) : Prop :=
  ∀ t : Shape, stability s ≥ stability t

-- Theorem statement
theorem equilateral_triangle_most_stable :
  is_most_stable Shape.EquilateralTriangle :=
sorry

end equilateral_triangle_most_stable_l1537_153794


namespace certain_number_proof_l1537_153769

theorem certain_number_proof (k : ℤ) (x : ℝ) 
  (h1 : x * (10 : ℝ)^(k : ℝ) > 100)
  (h2 : ∀ m : ℝ, m < 4.9956356288922485 → x * (10 : ℝ)^m ≤ 100) :
  x = 0.00101 := by
  sorry

end certain_number_proof_l1537_153769


namespace average_of_r_s_t_l1537_153793

theorem average_of_r_s_t (r s t : ℝ) (h : (5 / 2) * (r + s + t) = 25) :
  (r + s + t) / 3 = 10 / 3 := by
  sorry

end average_of_r_s_t_l1537_153793


namespace wolf_nobel_count_l1537_153732

/-- Represents the number of scientists with different prize combinations -/
structure ScientistCounts where
  total : ℕ
  wolf : ℕ
  nobel : ℕ
  wolfNobel : ℕ

/-- The conditions of the workshop -/
def workshopConditions (s : ScientistCounts) : Prop :=
  s.total = 50 ∧
  s.wolf = 31 ∧
  s.nobel = 29 ∧
  s.total - s.wolf = (s.nobel - s.wolfNobel) + (s.total - s.wolf - (s.nobel - s.wolfNobel)) + 3

/-- The theorem stating that 18 Wolf Prize laureates were also Nobel Prize laureates -/
theorem wolf_nobel_count (s : ScientistCounts) :
  workshopConditions s → s.wolfNobel = 18 := by
  sorry

end wolf_nobel_count_l1537_153732


namespace specific_pyramid_volume_l1537_153775

/-- A right pyramid with a square base -/
structure RightPyramid where
  base_area : ℝ
  total_surface_area : ℝ
  triangular_face_area : ℝ

/-- The volume of a right pyramid with a square base -/
def volume (p : RightPyramid) : ℝ := sorry

/-- Theorem: The volume of the specific right pyramid is 310.5√207 cubic units -/
theorem specific_pyramid_volume :
  ∀ (p : RightPyramid),
    p.total_surface_area = 486 ∧
    p.triangular_face_area = p.base_area / 3 →
    volume p = 310.5 * Real.sqrt 207 :=
by sorry

end specific_pyramid_volume_l1537_153775


namespace parabola_sum_l1537_153791

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- Check if a point is on the parabola -/
def contains_point (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2 + p.b * x + p.c

/-- Check if the parabola has a vertical axis of symmetry -/
def has_vertical_axis (p : Parabola) : Prop := sorry

theorem parabola_sum (p : Parabola) :
  vertex p = (3, 7) →
  has_vertical_axis p →
  contains_point p 0 4 →
  p.a + p.b + p.c = 5.666 := by sorry

end parabola_sum_l1537_153791


namespace sandwich_interval_is_40_minutes_l1537_153763

/-- Represents the Sandwich Shop's operations -/
structure SandwichShop where
  hours_per_day : ℕ
  peppers_per_day : ℕ
  peppers_per_sandwich : ℕ

/-- Calculates the interval between sandwiches in minutes -/
def sandwich_interval (shop : SandwichShop) : ℕ :=
  let sandwiches_per_day := shop.peppers_per_day / shop.peppers_per_sandwich
  let minutes_per_day := shop.hours_per_day * 60
  minutes_per_day / sandwiches_per_day

/-- The theorem stating the interval between sandwiches is 40 minutes -/
theorem sandwich_interval_is_40_minutes :
  ∀ (shop : SandwichShop),
    shop.hours_per_day = 8 →
    shop.peppers_per_day = 48 →
    shop.peppers_per_sandwich = 4 →
    sandwich_interval shop = 40 :=
by
  sorry


end sandwich_interval_is_40_minutes_l1537_153763


namespace age_ratio_nine_years_ago_l1537_153718

def henry_present_age : ℕ := 29
def jill_present_age : ℕ := 19

theorem age_ratio_nine_years_ago :
  (henry_present_age - 9) / (jill_present_age - 9) = 2 :=
by sorry

end age_ratio_nine_years_ago_l1537_153718


namespace sin_30_degrees_l1537_153749

/-- Sine of 30 degrees is equal to 1/2 -/
theorem sin_30_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end sin_30_degrees_l1537_153749


namespace geometric_sequence_increasing_condition_l1537_153773

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define an increasing sequence
def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_increasing_condition
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_positive : a 1 > 0) :
  (is_increasing_sequence a → a 1 ^ 2 < a 2 ^ 2) ∧
  ¬(a 1 ^ 2 < a 2 ^ 2 → is_increasing_sequence a) :=
by sorry

end geometric_sequence_increasing_condition_l1537_153773


namespace expenditure_recording_l1537_153759

/-- Given that income is recorded as positive and an income of 20 yuan is recorded as +20 yuan,
    prove that an expenditure of 75 yuan should be recorded as -75 yuan. -/
theorem expenditure_recording (income_recording : ℤ → ℤ) (h : income_recording 20 = 20) :
  income_recording (-75) = -75 := by
  sorry

end expenditure_recording_l1537_153759


namespace sugar_water_and_triangle_inequality_l1537_153727

theorem sugar_water_and_triangle_inequality 
  (a b m : ℝ) 
  (hab : b > a) (ha : a > 0) (hm : m > 0) 
  (A B C : ℝ) 
  (hABC : A > 0 ∧ B > 0 ∧ C > 0) 
  (hAcute : A < B + C ∧ B < C + A ∧ C < A + B) : 
  (a / b < (a + m) / (b + m)) ∧ 
  (A / (B + C) + B / (C + A) + C / (A + B) < 2) := by
  sorry

end sugar_water_and_triangle_inequality_l1537_153727


namespace problem_statement_l1537_153735

theorem problem_statement (a b : ℝ) : 
  ({1, a, b/a} : Set ℝ) = {0, a^2, a+b} → a^2015 + b^2016 = -1 := by
  sorry

end problem_statement_l1537_153735


namespace pascal_triangle_32nd_row_31st_element_l1537_153703

theorem pascal_triangle_32nd_row_31st_element : Nat.choose 32 30 = 496 := by
  sorry

end pascal_triangle_32nd_row_31st_element_l1537_153703


namespace seating_arrangements_l1537_153736

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def choose (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

theorem seating_arrangements (total_people : ℕ) (restricted_people : ℕ) 
  (h1 : total_people = 10) 
  (h2 : restricted_people = 3) : 
  factorial total_people - 
  (factorial (total_people - restricted_people + 1) * factorial restricted_people + 
   restricted_people * choose (total_people - restricted_people + 1) 1 * 
   factorial (total_people - restricted_people) - 
   restricted_people * (factorial (total_people - restricted_people + 1) * 
   factorial restricted_people)) = 3507840 := by
  sorry

end seating_arrangements_l1537_153736


namespace division_remainder_proof_l1537_153781

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 139 →
  divisor = 19 →
  quotient = 7 →
  dividend = divisor * quotient + remainder →
  remainder = 6 := by
sorry

end division_remainder_proof_l1537_153781


namespace donut_combinations_l1537_153744

/-- The number of ways to choose k items from n types with repetition. -/
def choose_with_repetition (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) k

/-- The number of donut types available. -/
def num_donut_types : ℕ := 6

/-- The number of remaining donuts to be chosen. -/
def remaining_donuts : ℕ := 2

/-- The total number of donuts in the order. -/
def total_donuts : ℕ := 8

/-- The number of donuts already accounted for (2 each of 3 specific kinds). -/
def accounted_donuts : ℕ := 6

theorem donut_combinations :
  choose_with_repetition num_donut_types remaining_donuts = 21 ∧
  total_donuts = accounted_donuts + remaining_donuts :=
by sorry

end donut_combinations_l1537_153744


namespace five_a_value_l1537_153798

theorem five_a_value (a : ℝ) (h : 5 * (a - 3) = 25) : 5 * a = 40 := by
  sorry

end five_a_value_l1537_153798


namespace sum_100th_group_value_l1537_153747

/-- The sum of the three numbers in the 100th group of the sequence (n, n^2, n^3) -/
def sum_100th_group : ℕ := 100 + 100^2 + 100^3

/-- Theorem stating that the sum of the 100th group is 1010100 -/
theorem sum_100th_group_value : sum_100th_group = 1010100 := by
  sorry

end sum_100th_group_value_l1537_153747


namespace factorization_of_difference_of_squares_l1537_153754

theorem factorization_of_difference_of_squares (a b : ℝ) :
  36 * a^2 - 4 * b^2 = 4 * (3*a + b) * (3*a - b) := by
  sorry

end factorization_of_difference_of_squares_l1537_153754


namespace farm_legs_count_l1537_153785

/-- Calculates the total number of legs for animals in a farm -/
def total_legs (total_animals : ℕ) (chickens : ℕ) (chicken_legs : ℕ) (buffalo_legs : ℕ) : ℕ :=
  let buffalos := total_animals - chickens
  chickens * chicken_legs + buffalos * buffalo_legs

/-- Theorem: In a farm with 13 animals, where 4 are chickens and the rest are buffalos,
    the total number of animal legs is 44, given that chickens have 2 legs each and
    buffalos have 4 legs each. -/
theorem farm_legs_count :
  total_legs 13 4 2 4 = 44 := by
  sorry


end farm_legs_count_l1537_153785


namespace square_area_error_l1537_153737

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.04)
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 8.16 := by
sorry

end square_area_error_l1537_153737


namespace hen_count_l1537_153733

theorem hen_count (total_heads : ℕ) (total_feet : ℕ) (hen_heads : ℕ) (hen_feet : ℕ) (cow_heads : ℕ) (cow_feet : ℕ) 
  (h1 : total_heads = 44)
  (h2 : total_feet = 128)
  (h3 : hen_heads = 1)
  (h4 : hen_feet = 2)
  (h5 : cow_heads = 1)
  (h6 : cow_feet = 4) :
  ∃ (num_hens : ℕ), num_hens = 24 ∧ 
    num_hens * hen_heads + (total_heads - num_hens) * cow_heads = total_heads ∧
    num_hens * hen_feet + (total_heads - num_hens) * cow_feet = total_feet :=
by sorry

end hen_count_l1537_153733


namespace function_properties_l1537_153704

theorem function_properties (f : ℝ → ℝ) 
  (h1 : ∀ x, f (2 - x) = -f (2 + x)) 
  (h2 : ∀ x, f (x + 2) = -f x) : 
  (f 0 = 0) ∧ 
  (∀ x, f (x + 4) = f x) ∧ 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (2 + x) = -f (2 - x)) := by
sorry

end function_properties_l1537_153704


namespace divisibility_problem_l1537_153710

theorem divisibility_problem (n : ℕ) (h1 : n > 0) (h2 : 100 % n = 3) (h3 : 197 % n = 3) : n = 97 := by
  sorry

end divisibility_problem_l1537_153710


namespace condition_equivalence_l1537_153709

theorem condition_equivalence (x : ℝ) : x > 0 ↔ x + 1/x ≥ 2 := by sorry

end condition_equivalence_l1537_153709


namespace arithmetic_mean_problem_l1537_153758

theorem arithmetic_mean_problem (x : ℝ) :
  (x + 3*x + 1000 + 3000) / 4 = 2018 ↔ x = 1018 := by
  sorry

end arithmetic_mean_problem_l1537_153758


namespace complement_of_union_M_N_l1537_153778

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 4}
def N : Set ℕ := {4, 5}

theorem complement_of_union_M_N :
  (M ∪ N)ᶜ = {1, 6} := by sorry

end complement_of_union_M_N_l1537_153778


namespace max_value_of_g_l1537_153753

-- Define the interval [0,1]
def interval : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

-- Define the function y = ax
def f (a : ℝ) : ℝ → ℝ := λ x ↦ a * x

-- Define the function y = 3ax - 1
def g (a : ℝ) : ℝ → ℝ := λ x ↦ 3 * a * x - 1

-- State the theorem
theorem max_value_of_g (a : ℝ) :
  (∃ (max min : ℝ), (∀ x ∈ interval, f a x ≤ max) ∧
                    (∀ x ∈ interval, min ≤ f a x) ∧
                    max + min = 3) →
  (∃ max : ℝ, (∀ x ∈ interval, g a x ≤ max) ∧
              (∃ y ∈ interval, g a y = max) ∧
              max = 5) :=
by sorry

end max_value_of_g_l1537_153753


namespace girls_from_maple_grove_l1537_153767

/-- Represents the number of students in different categories -/
structure StudentCounts where
  total : Nat
  girls : Nat
  boys : Nat
  pinecrest : Nat
  mapleGrove : Nat
  boysPinecrest : Nat

/-- The theorem stating that 40 girls are from Maple Grove School -/
theorem girls_from_maple_grove (s : StudentCounts)
  (h_total : s.total = 150)
  (h_girls : s.girls = 90)
  (h_boys : s.boys = 60)
  (h_pinecrest : s.pinecrest = 80)
  (h_mapleGrove : s.mapleGrove = 70)
  (h_boysPinecrest : s.boysPinecrest = 30)
  (h_total_sum : s.total = s.girls + s.boys)
  (h_school_sum : s.total = s.pinecrest + s.mapleGrove)
  : s.girls - (s.pinecrest - s.boysPinecrest) = 40 := by
  sorry


end girls_from_maple_grove_l1537_153767


namespace min_value_a_l1537_153788

theorem min_value_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (1/2), x^2 + 2*a*x + 1 ≥ 0) ↔ a ≥ -5/4 :=
sorry

end min_value_a_l1537_153788


namespace hard_round_points_is_five_l1537_153762

/-- A math contest with three rounds -/
structure MathContest where
  easy_correct : ℕ
  easy_points : ℕ
  avg_correct : ℕ
  avg_points : ℕ
  hard_correct : ℕ
  total_points : ℕ

/-- Kim's performance in the math contest -/
def kim_contest : MathContest := {
  easy_correct := 6
  easy_points := 2
  avg_correct := 2
  avg_points := 3
  hard_correct := 4
  total_points := 38
}

/-- Calculate the points per correct answer in the hard round -/
def hard_round_points (contest : MathContest) : ℕ :=
  (contest.total_points - (contest.easy_correct * contest.easy_points + contest.avg_correct * contest.avg_points)) / contest.hard_correct

/-- Theorem: The points per correct answer in the hard round is 5 -/
theorem hard_round_points_is_five : hard_round_points kim_contest = 5 := by
  sorry


end hard_round_points_is_five_l1537_153762


namespace inequality_solution_range_l1537_153700

/-- The inequality x^2 + ax - 2 < 0 has solutions within [2, 4] if and only if a ∈ (-∞, -1) -/
theorem inequality_solution_range (a : ℝ) :
  (∃ x ∈ Set.Icc 2 4, x^2 + a*x - 2 < 0) ↔ a < -1 := by
  sorry

end inequality_solution_range_l1537_153700


namespace integer_list_mean_l1537_153724

theorem integer_list_mean (m : ℤ) : 
  let ones := m + 1
  let twos := m + 2
  let threes := m + 3
  let fours := m + 4
  let fives := m + 5
  let total_count := ones + twos + threes + fours + fives
  let sum := ones * 1 + twos * 2 + threes * 3 + fours * 4 + fives * 5
  (sum : ℚ) / total_count = 19 / 6 → m = 9 := by
sorry

end integer_list_mean_l1537_153724


namespace error_percentage_l1537_153782

theorem error_percentage (x : ℝ) (h : x > 0) :
  ∃ ε > 0, abs ((x^2 - x/8) / x^2 * 100 - 88) < ε :=
sorry

end error_percentage_l1537_153782


namespace reciprocal_and_opposite_sum_l1537_153760

theorem reciprocal_and_opposite_sum (a b c d : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposite numbers
  : 3 * a * b + 2 * c + 2 * d = 3 := by
  sorry

end reciprocal_and_opposite_sum_l1537_153760


namespace exam_mean_score_l1537_153746

theorem exam_mean_score (a b : ℕ) (mean_a mean_b : ℝ) :
  a > 0 ∧ b > 0 →
  mean_a = 90 →
  mean_b = 78 →
  a = (5 : ℝ) / 7 * b →
  ∃ (max_score_a : ℝ), max_score_a = 100 ∧ max_score_a ≥ mean_b + 20 →
  (mean_a * a + mean_b * b) / (a + b) = 83 :=
by sorry

end exam_mean_score_l1537_153746


namespace chernov_has_gray_hair_l1537_153731

-- Define the three people
inductive Person : Type
| Sedov : Person
| Chernov : Person
| Ryzhov : Person

-- Define the hair colors
inductive HairColor : Type
| Gray : HairColor
| Red : HairColor
| Black : HairColor

-- Define the sports ranks
inductive SportsRank : Type
| MasterOfSports : SportsRank
| CandidateMaster : SportsRank
| FirstRank : SportsRank

-- Define the function that assigns a hair color to each person
def hairColor : Person → HairColor := sorry

-- Define the function that assigns a sports rank to each person
def sportsRank : Person → SportsRank := sorry

-- State the theorem
theorem chernov_has_gray_hair :
  -- No person's hair color matches their surname
  (hairColor Person.Sedov ≠ HairColor.Gray) ∧
  (hairColor Person.Chernov ≠ HairColor.Black) ∧
  (hairColor Person.Ryzhov ≠ HairColor.Red) ∧
  -- One person is gray-haired, one is red-haired, and one is black-haired
  (∃! p : Person, hairColor p = HairColor.Gray) ∧
  (∃! p : Person, hairColor p = HairColor.Red) ∧
  (∃! p : Person, hairColor p = HairColor.Black) ∧
  -- The black-haired person made the statement
  (∃ p : Person, hairColor p = HairColor.Black ∧ p ≠ Person.Sedov ∧ p ≠ Person.Chernov) ∧
  -- The Master of Sports confirmed the statement
  (sportsRank Person.Sedov = SportsRank.MasterOfSports) ∧
  (sportsRank Person.Chernov = SportsRank.CandidateMaster) ∧
  (sportsRank Person.Ryzhov = SportsRank.FirstRank) →
  -- Conclusion: Chernov has gray hair
  hairColor Person.Chernov = HairColor.Gray :=
by
  sorry


end chernov_has_gray_hair_l1537_153731


namespace max_area_equilateral_triangle_in_rectangle_max_area_equilateral_triangle_proof_l1537_153705

/-- The maximum area of an equilateral triangle inscribed in a 10x11 rectangle -/
theorem max_area_equilateral_triangle_in_rectangle : ℝ :=
  let rectangle_width := 10
  let rectangle_height := 11
  let max_area := 221 * Real.sqrt 3 - 330
  max_area

/-- Proof that the maximum area of an equilateral triangle inscribed in a 10x11 rectangle is 221√3 - 330 -/
theorem max_area_equilateral_triangle_proof : 
  max_area_equilateral_triangle_in_rectangle = 221 * Real.sqrt 3 - 330 := by
  sorry

end max_area_equilateral_triangle_in_rectangle_max_area_equilateral_triangle_proof_l1537_153705


namespace quadratic_inequality_always_negative_l1537_153777

theorem quadratic_inequality_always_negative :
  ∀ x : ℝ, -7 * x^2 + 4 * x - 6 < 0 := by
  sorry

end quadratic_inequality_always_negative_l1537_153777


namespace fibonacci_factorial_last_two_digits_sum_l1537_153725

def fibonacci_factorial_series : List Nat :=
  [1, 1, 2, 3, 5, 8, 13, 21, 34]

def last_two_digits (n : Nat) : Nat :=
  n % 100

def sum_last_two_digits (series : List Nat) : Nat :=
  (series.map (λ x => last_two_digits (Nat.factorial x))).sum

theorem fibonacci_factorial_last_two_digits_sum :
  sum_last_two_digits fibonacci_factorial_series = 50 := by
  sorry

end fibonacci_factorial_last_two_digits_sum_l1537_153725


namespace national_park_trees_l1537_153714

theorem national_park_trees (num_pines : ℕ) (num_redwoods : ℕ) : 
  num_pines = 600 →
  num_redwoods = num_pines + (num_pines * 20 / 100) →
  num_pines + num_redwoods = 1320 := by
sorry

end national_park_trees_l1537_153714


namespace soccer_attendance_difference_l1537_153774

theorem soccer_attendance_difference (seattle_estimate chicago_estimate : ℕ) 
  (seattle_actual chicago_actual : ℝ) : 
  seattle_estimate = 40000 →
  chicago_estimate = 50000 →
  seattle_actual ≥ 0.85 * seattle_estimate ∧ seattle_actual ≤ 1.15 * seattle_estimate →
  chicago_actual ≥ chicago_estimate / 1.15 ∧ chicago_actual ≤ chicago_estimate / 0.85 →
  ∃ (max_diff : ℕ), max_diff = 25000 ∧ 
    ∀ (diff : ℝ), diff = chicago_actual - seattle_actual → 
      diff ≤ max_diff ∧ 
      (max_diff - 500 < diff ∨ diff < max_diff + 500) :=
by sorry

end soccer_attendance_difference_l1537_153774


namespace largest_multiple_with_negation_constraint_l1537_153728

theorem largest_multiple_with_negation_constraint : 
  ∀ n : ℤ, n % 12 = 0 ∧ -n > -150 → n ≤ 144 := by sorry

end largest_multiple_with_negation_constraint_l1537_153728


namespace constant_ratio_problem_l1537_153730

theorem constant_ratio_problem (x y : ℝ) (k : ℝ) :
  (∀ x y, (4 * x - 5) / (2 * y + 20) = k) →
  (4 * 4 - 5) / (2 * 5 + 20) = k →
  (4 * 9 - 5) / (2 * (355 / 11) + 20) = k :=
by sorry

end constant_ratio_problem_l1537_153730


namespace regular_triangular_prism_edge_length_l1537_153770

/-- A regular triangular prism with edge length a and volume 16√3 has a = 4 -/
theorem regular_triangular_prism_edge_length (a : ℝ) : 
  a > 0 →  -- Ensure a is positive
  (1/4 : ℝ) * a^3 * Real.sqrt 3 = 16 * Real.sqrt 3 → 
  a = 4 := by
  sorry

#check regular_triangular_prism_edge_length

end regular_triangular_prism_edge_length_l1537_153770


namespace factorial_sum_theorem_l1537_153721

def is_solution (x y : ℕ) (z : ℤ) : Prop :=
  (Nat.factorial x + Nat.factorial y = 16 * z + 2017) ∧
  z % 2 ≠ 0

theorem factorial_sum_theorem :
  ∀ x y : ℕ, ∀ z : ℤ,
    is_solution x y z →
    ((x = 1 ∧ y = 6 ∧ z = -81) ∨
     (x = 6 ∧ y = 1 ∧ z = -81) ∨
     (x = 1 ∧ y = 7 ∧ z = 189) ∨
     (x = 7 ∧ y = 1 ∧ z = 189)) :=
by
  sorry

#check factorial_sum_theorem

end factorial_sum_theorem_l1537_153721


namespace ten_point_square_impossibility_l1537_153755

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of ten points in a plane -/
def TenPoints := Fin 10 → Point

/-- Predicate to check if four points lie on the boundary of some square -/
def FourPointsOnSquare (p₁ p₂ p₃ p₄ : Point) : Prop := sorry

/-- Predicate to check if all points in a set lie on the boundary of some square -/
def AllPointsOnSquare (points : TenPoints) : Prop := sorry

/-- The main theorem -/
theorem ten_point_square_impossibility (points : TenPoints) 
  (h : ∀ (a b c d : Fin 10), a ≠ b → b ≠ c → c ≠ d → d ≠ a → 
    FourPointsOnSquare (points a) (points b) (points c) (points d)) :
  ¬ AllPointsOnSquare points :=
sorry

end ten_point_square_impossibility_l1537_153755


namespace sequence_existence_l1537_153740

theorem sequence_existence (a b : ℕ) (h1 : b > a) (h2 : a > 1) (h3 : ¬(a ∣ b))
  (b_seq : ℕ → ℕ) (h4 : ∀ n, b_seq (n + 1) ≥ 2 * b_seq n) :
  ∃ a_seq : ℕ → ℕ,
    (∀ n, a_seq (n + 1) - a_seq n = a ∨ a_seq (n + 1) - a_seq n = b) ∧
    (∀ m l, a_seq m + a_seq l ∉ Set.range b_seq) :=
sorry

end sequence_existence_l1537_153740


namespace fraction_order_l1537_153712

theorem fraction_order : 
  let f1 := (4 : ℚ) / 3
  let f2 := (4 : ℚ) / 5
  let f3 := (4 : ℚ) / 6
  let f4 := (3 : ℚ) / 5
  let f5 := (6 : ℚ) / 5
  let f6 := (2 : ℚ) / 5
  (f6 < f4) ∧ (f4 < f3) ∧ (f3 < f2) ∧ (f2 < f5) ∧ (f5 < f1) := by
sorry

end fraction_order_l1537_153712


namespace cost_price_is_60_l1537_153756

/-- The cost price of a single ball, given the selling price of multiple balls and the loss incurred. -/
def cost_price_of_ball (selling_price : ℕ) (num_balls_sold : ℕ) (num_balls_loss : ℕ) : ℕ :=
  selling_price / (num_balls_sold - num_balls_loss)

/-- Theorem stating that the cost price of a ball is 60 under given conditions. -/
theorem cost_price_is_60 :
  cost_price_of_ball 720 17 5 = 60 := by
  sorry

end cost_price_is_60_l1537_153756


namespace least_possible_difference_l1537_153723

theorem least_possible_difference (x y z N : ℤ) (h1 : x < y) (h2 : y < z) 
  (h3 : y - x > 5) (h4 : Even x) (h5 : Odd y) (h6 : Odd z) (h7 : ∃ k : ℤ, x = 5 * k) 
  (h8 : y^2 + z^2 = N) (h9 : N > 0) : 
  (∀ w : ℤ, w ≥ 0 → z - x ≥ w + 9) ∧ (z - x = 9) :=
sorry

end least_possible_difference_l1537_153723


namespace max_product_with_constraints_l1537_153768

theorem max_product_with_constraints :
  ∀ a b : ℕ,
  a + b = 100 →
  a % 3 = 2 →
  b % 7 = 5 →
  ∀ x y : ℕ,
  x + y = 100 →
  x % 3 = 2 →
  y % 7 = 5 →
  a * b ≤ 2491 ∧ (∃ a b : ℕ, a + b = 100 ∧ a % 3 = 2 ∧ b % 7 = 5 ∧ a * b = 2491) :=
by
  sorry

end max_product_with_constraints_l1537_153768


namespace quadratic_factorization_l1537_153726

theorem quadratic_factorization (b : ℤ) : 
  (∃ (c d e f : ℤ), (35 : ℤ) * x ^ 2 + b * x + 35 = (c * x + d) * (e * x + f)) →
  (∃ (k : ℤ), b = 2 * k) ∧ 
  ¬(∀ (k : ℤ), ∃ (c d e f : ℤ), (35 : ℤ) * x ^ 2 + (2 * k) * x + 35 = (c * x + d) * (e * x + f)) :=
by sorry

end quadratic_factorization_l1537_153726


namespace c_is_largest_l1537_153707

/-- Given that a - 1 = b + 2 = c - 3 = d + 4, prove that c is the largest among a, b, c, and d -/
theorem c_is_largest (a b c d : ℝ) (h : a - 1 = b + 2 ∧ b + 2 = c - 3 ∧ c - 3 = d + 4) :
  c = max a (max b (max c d)) := by
  sorry

end c_is_largest_l1537_153707


namespace angle_at_5pm_l1537_153771

/-- The angle between the hour and minute hands of a clock at a given hour -/
def clockAngle (hour : ℝ) : ℝ := 30 * hour

/-- Proposition: The angle between the minute hand and hour hand is 150° at 5 pm -/
theorem angle_at_5pm : clockAngle 5 = 150 := by sorry

end angle_at_5pm_l1537_153771


namespace jills_total_earnings_l1537_153722

/-- Calculates Jill's earnings over three months based on specific working conditions. -/
def jills_earnings (days_per_month : ℕ) (first_month_rate : ℕ) : ℕ :=
  let second_month_rate := 2 * first_month_rate
  let first_month := days_per_month * first_month_rate
  let second_month := days_per_month * second_month_rate
  let third_month := (days_per_month / 2) * second_month_rate
  first_month + second_month + third_month

/-- Theorem stating that Jill's earnings over three months equal $1,200 -/
theorem jills_total_earnings : 
  jills_earnings 30 10 = 1200 := by
  sorry

#eval jills_earnings 30 10

end jills_total_earnings_l1537_153722


namespace sports_club_members_l1537_153790

theorem sports_club_members (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ) :
  total = 30 →
  badminton = 18 →
  tennis = 19 →
  both = 9 →
  total - (badminton + tennis - both) = 2 := by
  sorry

end sports_club_members_l1537_153790


namespace convex_curve_triangle_inequalities_l1537_153702

/-- A convex curve in a metric space -/
class ConvexCurve (α : Type*) [MetricSpace α]

/-- The distance between two convex curves -/
def curve_distance {α : Type*} [MetricSpace α] (A B : ConvexCurve α) : ℝ := sorry

/-- Triangle inequalities for distances between convex curves -/
theorem convex_curve_triangle_inequalities
  {α : Type*} [MetricSpace α]
  (A B C : ConvexCurve α) :
  let AB := curve_distance A B
  let BC := curve_distance B C
  let AC := curve_distance A C
  (AB + BC ≥ AC) ∧ (AC + BC ≥ AB) ∧ (AB + AC ≥ BC) :=
by sorry

end convex_curve_triangle_inequalities_l1537_153702


namespace conference_handshakes_l1537_153711

theorem conference_handshakes (n : ℕ) (h : n = 12) : n.choose 2 = 66 := by
  sorry

end conference_handshakes_l1537_153711


namespace total_trips_is_seven_l1537_153739

/-- Calculates the number of trips needed to carry a given number of trays -/
def trips_needed (trays_per_trip : ℕ) (num_trays : ℕ) : ℕ :=
  (num_trays + trays_per_trip - 1) / trays_per_trip

/-- Proves that the total number of trips needed is 7 -/
theorem total_trips_is_seven (trays_per_trip : ℕ) (table1_trays : ℕ) (table2_trays : ℕ)
    (h1 : trays_per_trip = 3)
    (h2 : table1_trays = 15)
    (h3 : table2_trays = 5) :
    trips_needed trays_per_trip table1_trays + trips_needed trays_per_trip table2_trays = 7 := by
  sorry

#eval trips_needed 3 15 + trips_needed 3 5

end total_trips_is_seven_l1537_153739


namespace right_triangle_area_right_triangle_area_is_625_div_3_l1537_153780

/-- A right triangle XYZ in the xy-plane with specific properties -/
structure RightTriangle where
  /-- The length of the hypotenuse XY -/
  hypotenuse_length : ℝ
  /-- The y-intercept of the line containing the median through X -/
  median_x_intercept : ℝ
  /-- The slope of the line containing the median through Y -/
  median_y_slope : ℝ
  /-- The y-intercept of the line containing the median through Y -/
  median_y_intercept : ℝ
  /-- Condition: The hypotenuse length is 50 -/
  hypotenuse_cond : hypotenuse_length = 50
  /-- Condition: The median through X lies on y = x + 5 -/
  median_x_cond : median_x_intercept = 5
  /-- Condition: The median through Y lies on y = 3x + 6 -/
  median_y_cond : median_y_slope = 3 ∧ median_y_intercept = 6

/-- The theorem stating that the area of the specific right triangle is 625/3 -/
theorem right_triangle_area (t : RightTriangle) : ℝ :=
  625 / 3

/-- The main theorem to be proved -/
theorem right_triangle_area_is_625_div_3 (t : RightTriangle) :
  right_triangle_area t = 625 / 3 := by
  sorry

end right_triangle_area_right_triangle_area_is_625_div_3_l1537_153780


namespace total_cost_calculation_l1537_153716

def land_acres : ℕ := 30
def land_cost_per_acre : ℕ := 20
def house_cost : ℕ := 120000
def cow_count : ℕ := 20
def cow_cost_per_unit : ℕ := 1000
def chicken_count : ℕ := 100
def chicken_cost_per_unit : ℕ := 5
def solar_installation_hours : ℕ := 6
def solar_installation_cost_per_hour : ℕ := 100
def solar_equipment_cost : ℕ := 6000

theorem total_cost_calculation :
  land_acres * land_cost_per_acre +
  house_cost +
  cow_count * cow_cost_per_unit +
  chicken_count * chicken_cost_per_unit +
  solar_installation_hours * solar_installation_cost_per_hour +
  solar_equipment_cost = 147700 := by
  sorry

end total_cost_calculation_l1537_153716


namespace total_amount_paid_l1537_153795

/-- Represents the purchase of a fruit with its quantity and price per kg -/
structure FruitPurchase where
  quantity : ℕ
  price_per_kg : ℕ

/-- Calculates the total cost of a fruit purchase -/
def total_cost (purchase : FruitPurchase) : ℕ :=
  purchase.quantity * purchase.price_per_kg

/-- Represents Tom's fruit shopping -/
def fruit_shopping : List FruitPurchase :=
  [
    { quantity := 8, price_per_kg := 70 },  -- Apples
    { quantity := 9, price_per_kg := 65 },  -- Mangoes
    { quantity := 5, price_per_kg := 50 },  -- Oranges
    { quantity := 3, price_per_kg := 30 }   -- Bananas
  ]

/-- Theorem: The total amount Tom paid for all fruits is $1485 -/
theorem total_amount_paid : (fruit_shopping.map total_cost).sum = 1485 := by
  sorry

end total_amount_paid_l1537_153795


namespace limit_proof_l1537_153742

/-- The limit of (3x^2 + 5x - 2) / (x + 2) as x approaches -2 is -7 -/
theorem limit_proof (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x : ℝ, x ≠ -2 → |x + 2| < δ →
    |(3*x^2 + 5*x - 2) / (x + 2) + 7| < ε :=
by
  use ε/3
  sorry

end limit_proof_l1537_153742


namespace smallest_x_for_equation_l1537_153751

theorem smallest_x_for_equation : 
  ∃ (x : ℕ), x > 0 ∧ 
  (∃ (y : ℕ), y > 0 ∧ (3 : ℚ) / 4 = y / (210 + x)) ∧
  (∀ (x' : ℕ), x' > 0 → x' < x → 
    ¬∃ (y : ℕ), y > 0 ∧ (3 : ℚ) / 4 = y / (210 + x')) ∧
  x = 2 := by
sorry

end smallest_x_for_equation_l1537_153751


namespace range_f_and_a_condition_l1537_153715

/-- The function f(x) = 3|x-1| + |3x+1| -/
def f (x : ℝ) : ℝ := 3 * abs (x - 1) + abs (3 * x + 1)

/-- The function g(x) = |x+2| + |x-a| -/
def g (a : ℝ) (x : ℝ) : ℝ := abs (x + 2) + abs (x - a)

/-- The set A, which is the range of f -/
def A : Set ℝ := Set.range f

/-- The set B, which is the range of g for a given a -/
def B (a : ℝ) : Set ℝ := Set.range (g a)

theorem range_f_and_a_condition (a : ℝ) :
  (A = Set.Ici 4) ∧ (A ∪ B a = B a) → a ∈ Set.Icc (-6) 2 := by
  sorry

end range_f_and_a_condition_l1537_153715


namespace f_of_f_of_3_l1537_153720

def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 4

theorem f_of_f_of_3 : f (f 3) = 692 := by
  sorry

end f_of_f_of_3_l1537_153720


namespace unique_solution_quadratic_l1537_153766

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x, k * x^2 - 3 * x + 2 = 0) → (k = 0 ∨ k = 9/8) := by
  sorry

end unique_solution_quadratic_l1537_153766


namespace cistern_emptied_l1537_153799

/-- Represents the emptying rate of a pipe in terms of fraction of cistern per minute -/
structure PipeRate where
  fraction : ℚ
  time : ℚ

/-- Calculates the rate at which a pipe empties a cistern -/
def emptyingRate (p : PipeRate) : ℚ :=
  p.fraction / p.time

/-- Calculates the total emptying rate of multiple pipes -/
def totalRate (pipes : List PipeRate) : ℚ :=
  pipes.map emptyingRate |> List.sum

/-- Theorem: Given the specified pipes and time, the entire cistern will be emptied -/
theorem cistern_emptied (pipeA pipeB pipeC : PipeRate) 
    (h1 : pipeA = { fraction := 3/4, time := 12 })
    (h2 : pipeB = { fraction := 1/2, time := 15 })
    (h3 : pipeC = { fraction := 1/3, time := 10 })
    (time : ℚ)
    (h4 : time = 8) :
    totalRate [pipeA, pipeB, pipeC] * time ≥ 1 := by
  sorry


end cistern_emptied_l1537_153799


namespace remainder_divisibility_l1537_153783

theorem remainder_divisibility (N : ℤ) : 
  ∃ k : ℤ, N = 45 * k + 31 → ∃ m : ℤ, N = 15 * m + 1 :=
by sorry

end remainder_divisibility_l1537_153783


namespace harmonic_sum_inequality_l1537_153776

theorem harmonic_sum_inequality : 1 + 1/2 + 1/3 < 2 := by
  sorry

end harmonic_sum_inequality_l1537_153776


namespace square_ratio_proof_l1537_153717

theorem square_ratio_proof : ∃ (a b c : ℕ), 
  (300 : ℚ) / 75 = (a * Real.sqrt b / c)^2 ∧ a + b + c = 4 :=
by sorry

end square_ratio_proof_l1537_153717


namespace quadratic_roots_condition_l1537_153729

theorem quadratic_roots_condition (b c : ℝ) :
  (c < 0 → ∃ x : ℂ, x^2 + b*x + c = 0) ∧
  ¬(∃ x : ℂ, x^2 + b*x + c = 0 → c < 0) :=
sorry

end quadratic_roots_condition_l1537_153729


namespace domain_transformation_l1537_153772

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem domain_transformation (h : Set.Icc (-3 : ℝ) 3 = {x | ∃ y, f (2*y - 1) = x}) :
  {x | ∃ y, f y = x} = Set.Icc (-7 : ℝ) 5 := by
  sorry

end domain_transformation_l1537_153772


namespace g_of_2_equals_5_l1537_153779

/-- Given a function g(x) = x^3 - 2x + 1, prove that g(2) = 5 -/
theorem g_of_2_equals_5 :
  let g : ℝ → ℝ := fun x ↦ x^3 - 2*x + 1
  g 2 = 5 := by
  sorry

end g_of_2_equals_5_l1537_153779


namespace events_mutually_exclusive_not_contradictory_l1537_153741

/-- Represents the total number of products -/
def total_products : ℕ := 5

/-- Represents the number of qualified products -/
def qualified_products : ℕ := 3

/-- Represents the number of unqualified products -/
def unqualified_products : ℕ := 2

/-- Represents the number of products randomly selected -/
def selected_products : ℕ := 2

/-- Event A: Exactly 1 unqualified product is selected -/
def event_A : Prop := sorry

/-- Event B: Exactly 2 qualified products are selected -/
def event_B : Prop := sorry

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutually_exclusive (e1 e2 : Prop) : Prop := ¬(e1 ∧ e2)

/-- Two events are contradictory if one must occur when the other does not -/
def contradictory (e1 e2 : Prop) : Prop := (e1 ↔ ¬e2)

theorem events_mutually_exclusive_not_contradictory :
  mutually_exclusive event_A event_B ∧ ¬contradictory event_A event_B := by sorry

end events_mutually_exclusive_not_contradictory_l1537_153741


namespace line_passes_through_fixed_point_l1537_153786

/-- The line equation passing through a fixed point for all values of k -/
def line_equation (k x y : ℝ) : Prop :=
  (k + 2) * x + (1 - k) * y - 4 * k - 5 = 0

/-- Theorem stating that the line passes through the point (3, -1) for all k -/
theorem line_passes_through_fixed_point :
  ∀ k : ℝ, line_equation k 3 (-1) :=
by
  sorry

end line_passes_through_fixed_point_l1537_153786


namespace smallest_lcm_with_gcd_3_l1537_153748

theorem smallest_lcm_with_gcd_3 (k l : ℕ) : 
  k ≥ 1000 ∧ k ≤ 9999 ∧ l ≥ 1000 ∧ l ≤ 9999 ∧ Nat.gcd k l = 3 →
  Nat.lcm k l ≥ 335670 ∧ ∃ (k₀ l₀ : ℕ), k₀ ≥ 1000 ∧ k₀ ≤ 9999 ∧ l₀ ≥ 1000 ∧ l₀ ≤ 9999 ∧ 
  Nat.gcd k₀ l₀ = 3 ∧ Nat.lcm k₀ l₀ = 335670 :=
by sorry

end smallest_lcm_with_gcd_3_l1537_153748


namespace power_five_mod_six_l1537_153765

theorem power_five_mod_six : 5^2023 % 6 = 5 := by
  sorry

end power_five_mod_six_l1537_153765


namespace largest_prime_divisor_factorial_sum_l1537_153757

theorem largest_prime_divisor_factorial_sum : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (Nat.factorial 13 + Nat.factorial 14) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 13 + Nat.factorial 14) → q ≤ p :=
by sorry

end largest_prime_divisor_factorial_sum_l1537_153757


namespace product_of_roots_plus_one_l1537_153761

theorem product_of_roots_plus_one (p q r : ℝ) : 
  (p^3 - 15*p^2 + 25*p - 10 = 0) →
  (q^3 - 15*q^2 + 25*q - 10 = 0) →
  (r^3 - 15*r^2 + 25*r - 10 = 0) →
  (1 + p) * (1 + q) * (1 + r) = 51 := by
sorry

end product_of_roots_plus_one_l1537_153761


namespace smallest_a_for_integer_sqrt_8a_l1537_153706

theorem smallest_a_for_integer_sqrt_8a : 
  (∃ (a : ℕ), a > 0 ∧ ∃ (n : ℕ), n^2 = 8*a) → 
  (∀ (a : ℕ), a > 0 → (∃ (n : ℕ), n^2 = 8*a) → a ≥ 2) ∧
  (∃ (n : ℕ), n^2 = 8*2) :=
sorry

end smallest_a_for_integer_sqrt_8a_l1537_153706


namespace circle_and_tangents_l1537_153752

-- Define the points
def A : ℝ × ℝ := (-1, -3)
def B : ℝ × ℝ := (5, 5)
def M : ℝ × ℝ := (-3, 2)

-- Define the circle O
def O : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 - 1)^2 = 25}

-- Define the tangent lines
def tangent_lines : Set (Set (ℝ × ℝ)) := 
  {{p | p.1 = -3}, {p | 12 * p.1 - 5 * p.2 + 46 = 0}}

theorem circle_and_tangents :
  (∀ p ∈ O, (p.1 - 2)^2 + (p.2 - 1)^2 = 25) ∧
  (∀ l ∈ tangent_lines, ∃ p ∈ O, p ∈ l ∧ 
    (∀ q ∈ O, q ≠ p → q ∉ l)) ∧
  (∀ l ∈ tangent_lines, M ∈ l) :=
sorry

end circle_and_tangents_l1537_153752


namespace area_of_trapezoid_DBCE_l1537_153708

-- Define the triangle ABC
structure Triangle (ABC : Type) where
  AB : ℝ
  AC : ℝ
  area : ℝ

-- Define the smallest triangle
def SmallestTriangle : Triangle Unit :=
  { AB := 1, AC := 1, area := 2 }

-- Define the triangle ADE
def TriangleADE : Triangle Unit :=
  { AB := 1, AC := 1, area := 5 * SmallestTriangle.area }

-- Define the triangle ABC
def TriangleABC : Triangle Unit :=
  { AB := 1, AC := 1, area := 80 }

-- Define the trapezoid DBCE
def TrapezoidDBCE : ℝ := TriangleABC.area - TriangleADE.area

-- Theorem statement
theorem area_of_trapezoid_DBCE : TrapezoidDBCE = 70 := by
  sorry

end area_of_trapezoid_DBCE_l1537_153708


namespace min_isosceles_right_triangles_10x100_l1537_153750

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle

/-- Returns the minimum number of isosceles right triangles needed to cover a rectangle -/
def minIsoscelesRightTriangles (r : Rectangle) : ℕ := sorry

/-- The theorem statement -/
theorem min_isosceles_right_triangles_10x100 :
  minIsoscelesRightTriangles ⟨100, 10⟩ = 11 := by sorry

end min_isosceles_right_triangles_10x100_l1537_153750


namespace f_composition_equals_pi_plus_one_l1537_153713

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

-- State the theorem
theorem f_composition_equals_pi_plus_one :
  f (f (f (-1))) = Real.pi + 1 := by
  sorry

end f_composition_equals_pi_plus_one_l1537_153713


namespace incenter_vector_ratio_l1537_153792

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define vector operations
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vec_scale (a : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (a * v.1, a * v.2)

-- Main theorem
theorem incenter_vector_ratio (t : Triangle) 
  (h1 : dist t.A t.B = 6)
  (h2 : dist t.B t.C = 7)
  (h3 : dist t.A t.C = 4)
  (O : ℝ × ℝ)
  (hO : O = incenter t)
  (p q : ℝ)
  (h4 : vec_add (vec_scale (-1) O) t.A = vec_add (vec_scale p (vec_add (vec_scale (-1) t.A) t.B)) 
                                                 (vec_scale q (vec_add (vec_scale (-1) t.A) t.C)))
  : p / q = 2 / 3 := by
  sorry


end incenter_vector_ratio_l1537_153792


namespace train_crossing_time_l1537_153764

/-- Proves that a train 130 m long, moving at 144 km/hr, takes 3.25 seconds to cross an electric pole -/
theorem train_crossing_time : 
  let train_length : ℝ := 130 -- meters
  let train_speed_kmh : ℝ := 144 -- km/hr
  let train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600 -- Convert km/hr to m/s
  let crossing_time : ℝ := train_length / train_speed_ms
  crossing_time = 3.25 := by
sorry


end train_crossing_time_l1537_153764


namespace number_of_advertisements_number_of_advertisements_proof_l1537_153743

/-- The number of advertisements shown during a race, given their duration, 
    cost per minute, and total transmission cost. -/
theorem number_of_advertisements (ad_duration : ℕ) (cost_per_minute : ℕ) (total_cost : ℕ) : ℕ :=
  5
where
  ad_duration := 3
  cost_per_minute := 4000
  total_cost := 60000

/-- Proof of the theorem -/
theorem number_of_advertisements_proof :
  number_of_advertisements 3 4000 60000 = 5 := by
  sorry

end number_of_advertisements_number_of_advertisements_proof_l1537_153743


namespace exists_irrational_in_interval_l1537_153784

theorem exists_irrational_in_interval :
  ∃ x : ℝ, x ∈ Set.Ioo 0.3 0.4 ∧ Irrational x ∧ x * (x + 1) * (x + 2) = 1 := by
  sorry

end exists_irrational_in_interval_l1537_153784


namespace probability_same_length_l1537_153789

/-- The set of all sides and diagonals of a regular hexagon -/
def S : Finset ℕ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of segments in S -/
def total_segments : ℕ := num_sides + num_diagonals

/-- The number of ways to choose 2 segments from S -/
def total_choices : ℕ := (total_segments.choose 2)

/-- The number of ways to choose 2 sides -/
def side_choices : ℕ := (num_sides.choose 2)

/-- The number of ways to choose 2 diagonals -/
def diagonal_choices : ℕ := (num_diagonals.choose 2)

/-- The total number of favorable outcomes (choosing two segments of the same length) -/
def favorable_outcomes : ℕ := side_choices + diagonal_choices

/-- The probability of selecting two segments of the same length from S -/
theorem probability_same_length : 
  (favorable_outcomes : ℚ) / total_choices = 17 / 35 :=
sorry

end probability_same_length_l1537_153789


namespace platform_length_l1537_153701

/-- Given a train of length 300 meters that crosses a platform in 38 seconds
    and a signal pole in 18 seconds, prove that the length of the platform
    is approximately 333.46 meters. -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_time = 38)
  (h3 : pole_time = 18) :
  ∃ (platform_length : ℝ), abs (platform_length - 333.46) < 0.01 :=
by sorry

end platform_length_l1537_153701


namespace train_crossing_platform_time_l1537_153797

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_platform_time 
  (train_length : ℝ) 
  (signal_pole_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 200) 
  (h2 : signal_pole_time = 42) 
  (h3 : platform_length = 38.0952380952381) : 
  (train_length + platform_length) / (train_length / signal_pole_time) = 50 := by
  sorry

#check train_crossing_platform_time

end train_crossing_platform_time_l1537_153797
