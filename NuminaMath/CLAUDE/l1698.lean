import Mathlib

namespace journey_average_mpg_l1698_169826

/-- Represents a car's journey with odometer readings and gas fill-ups -/
structure CarJourney where
  initial_odometer : ℕ
  initial_gas : ℕ
  intermediate_odometer : ℕ
  intermediate_gas : ℕ
  final_odometer : ℕ
  final_gas : ℕ

/-- Calculates the average miles per gallon for a car journey -/
def average_mpg (journey : CarJourney) : ℚ :=
  let total_distance : ℕ := journey.final_odometer - journey.initial_odometer
  let total_gas : ℕ := journey.initial_gas + journey.intermediate_gas + journey.final_gas
  (total_distance : ℚ) / total_gas

/-- Theorem stating that the average mpg for the given journey is 15.2 -/
theorem journey_average_mpg :
  let journey : CarJourney := {
    initial_odometer := 35200,
    initial_gas := 10,
    intermediate_odometer := 35480,
    intermediate_gas := 15,
    final_odometer := 35960,
    final_gas := 25
  }
  average_mpg journey = 152 / 10 := by sorry

end journey_average_mpg_l1698_169826


namespace fraction_of_fraction_l1698_169867

theorem fraction_of_fraction : (5 / 12) / (3 / 4) = 5 / 9 := by sorry

end fraction_of_fraction_l1698_169867


namespace xiaomas_calculation_l1698_169829

theorem xiaomas_calculation (square : ℤ) (h : 40 + square = 35) : 40 / square = -8 := by
  sorry

end xiaomas_calculation_l1698_169829


namespace divisibility_by_480_l1698_169873

theorem divisibility_by_480 (a : ℤ) 
  (h1 : ¬ (4 ∣ a)) 
  (h2 : a % 10 = 4) : 
  480 ∣ (a * (a^2 - 1) * (a^2 - 4)) := by
  sorry

end divisibility_by_480_l1698_169873


namespace expression_simplification_l1698_169868

theorem expression_simplification (x y : ℝ) (h : x * y ≠ 0) :
  (x^2 + 2 / x^2) * (y^2 + 2 / y^2) + (x^2 - 2 / y^2) * (y^2 - 2 / x^2) = 2 + 8 / (x^2 * y^2) := by
  sorry

end expression_simplification_l1698_169868


namespace percentage_of_indian_women_l1698_169846

theorem percentage_of_indian_women (total_men : ℕ) (total_women : ℕ) (total_children : ℕ)
  (indian_men_percentage : ℚ) (indian_children_percentage : ℚ) (non_indian_percentage : ℚ)
  (h_total_men : total_men = 700)
  (h_total_women : total_women = 500)
  (h_total_children : total_children = 800)
  (h_indian_men : indian_men_percentage = 20 / 100)
  (h_indian_children : indian_children_percentage = 10 / 100)
  (h_non_indian : non_indian_percentage = 79 / 100) :
  (((1 - non_indian_percentage) * (total_men + total_women + total_children)
    - indian_men_percentage * total_men
    - indian_children_percentage * total_children)
   / total_women) = 40 / 100 :=
by sorry

end percentage_of_indian_women_l1698_169846


namespace largest_binomial_term_l1698_169808

theorem largest_binomial_term (n : ℕ) (x : ℝ) (h1 : n = 500) (h2 : x = 0.3) :
  let A : ℕ → ℝ := λ k => (n.choose k) * x^k
  ∃ k : ℕ, k = 125 ∧ ∀ j : ℕ, j ≤ n → A k ≥ A j :=
by sorry

end largest_binomial_term_l1698_169808


namespace spherical_segment_height_l1698_169853

/-- The height of a spherical segment given a right-angled triangle inscribed in its base -/
theorem spherical_segment_height
  (S : ℝ) -- Area of the inscribed right-angled triangle
  (α : ℝ) -- Acute angle of the inscribed right-angled triangle
  (β : ℝ) -- Central angle of the segment's arc in axial section
  (h_S_pos : S > 0)
  (h_α_pos : 0 < α)
  (h_α_lt_pi_2 : α < π / 2)
  (h_β_pos : 0 < β)
  (h_β_lt_pi : β < π) :
  ∃ (height : ℝ), height = Real.sqrt (S / Real.sin (2 * α)) * Real.tan (β / 4) :=
sorry

end spherical_segment_height_l1698_169853


namespace two_roots_condition_l1698_169801

-- Define the quadratic equation
def quadratic_equation (x a : ℝ) : Prop := x^2 - 2*x + a = 0

-- Define the condition for having two distinct real roots
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation x a ∧ quadratic_equation y a

-- Statement of the theorem
theorem two_roots_condition (a : ℝ) :
  has_two_distinct_roots a ↔ a < 1 :=
sorry

end two_roots_condition_l1698_169801


namespace grid_paths_count_l1698_169851

/-- The number of paths from (0, 0) to (n, n) on an n × n grid,
    moving only 1 up or 1 right at a time -/
def gridPaths (n : ℕ) : ℕ :=
  Nat.choose (2 * n) n

/-- Theorem stating that the number of paths on an n × n grid
    from (0, 0) to (n, n), moving only 1 up or 1 right at a time,
    is equal to (2n choose n) -/
theorem grid_paths_count (n : ℕ) :
  gridPaths n = Nat.choose (2 * n) n := by
  sorry

end grid_paths_count_l1698_169851


namespace problem_statement_l1698_169824

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  (ab ≤ 1/8) ∧ (2/(a+1) + 1/b ≥ 3 + 2*Real.sqrt 2) := by
  sorry

end problem_statement_l1698_169824


namespace greatest_common_factor_three_digit_palindromes_l1698_169897

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

end greatest_common_factor_three_digit_palindromes_l1698_169897


namespace worker_efficiency_l1698_169885

/-- Given two workers A and B, where A is twice as efficient as B, and they complete a work together in 12 days, prove that A can complete the work alone in 18 days. -/
theorem worker_efficiency (work_rate_A work_rate_B : ℝ) (total_time : ℝ) :
  work_rate_A = 2 * work_rate_B →
  work_rate_A + work_rate_B = 1 / total_time →
  total_time = 12 →
  1 / work_rate_A = 18 := by
  sorry

end worker_efficiency_l1698_169885


namespace part1_part2_l1698_169847

-- Definition of arithmetic sequence sum
def S (a1 : ℚ) (d : ℚ) (n : ℕ) : ℚ := n * a1 + (n * (n - 1) / 2) * d

-- Part 1
theorem part1 : ∃! k : ℕ+, S (3/2) 1 (k^2) = (S (3/2) 1 k)^2 := by sorry

-- Part 2
theorem part2 : ∀ a1 d : ℚ, 
  (∀ k : ℕ+, S a1 d (k^2) = (S a1 d k)^2) ↔ 
  ((a1 = 0 ∧ d = 0) ∨ (a1 = 1 ∧ d = 0) ∨ (a1 = 1 ∧ d = 2)) := by sorry

end part1_part2_l1698_169847


namespace monotone_increasing_interval_l1698_169896

noncomputable def f (x : ℝ) : ℝ := (Real.cos x + Real.sin x) * Real.cos (x - Real.pi / 2)

theorem monotone_increasing_interval (k : ℤ) :
  StrictMonoOn f { x | k * Real.pi - Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + 3 * Real.pi / 8 } :=
sorry

end monotone_increasing_interval_l1698_169896


namespace investment_value_after_one_year_l1698_169809

def initial_investment : ℝ := 900
def num_stocks : ℕ := 3
def stock_a_multiplier : ℝ := 2
def stock_b_multiplier : ℝ := 2
def stock_c_multiplier : ℝ := 0.5

theorem investment_value_after_one_year :
  let investment_per_stock := initial_investment / num_stocks
  let stock_a_value := investment_per_stock * stock_a_multiplier
  let stock_b_value := investment_per_stock * stock_b_multiplier
  let stock_c_value := investment_per_stock * stock_c_multiplier
  stock_a_value + stock_b_value + stock_c_value = 1350 := by
  sorry

end investment_value_after_one_year_l1698_169809


namespace computer_price_is_150_l1698_169878

/-- The price per computer in a factory with given production and earnings -/
def price_per_computer (daily_production : ℕ) (weekly_earnings : ℕ) : ℚ :=
  weekly_earnings / (daily_production * 7)

/-- Theorem stating that the price per computer is $150 -/
theorem computer_price_is_150 :
  price_per_computer 1500 1575000 = 150 := by
  sorry

end computer_price_is_150_l1698_169878


namespace factor_theorem_quadratic_l1698_169845

theorem factor_theorem_quadratic (k : ℚ) : 
  (∀ m : ℚ, (m - 8) ∣ (m^2 - k*m - 24)) → k = 5 := by
  sorry

end factor_theorem_quadratic_l1698_169845


namespace min_cuts_for_20_gons_l1698_169899

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

end min_cuts_for_20_gons_l1698_169899


namespace tan_half_sum_l1698_169889

theorem tan_half_sum (a b : Real) 
  (h1 : Real.cos a + Real.cos b = 3/5)
  (h2 : Real.sin a + Real.sin b = 1/5) :
  Real.tan ((a + b) / 2) = 1/3 := by
sorry

end tan_half_sum_l1698_169889


namespace exponential_strictly_increasing_l1698_169820

theorem exponential_strictly_increasing :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → (2 : ℝ) ^ x₁ < (2 : ℝ) ^ x₂ := by
  sorry

end exponential_strictly_increasing_l1698_169820


namespace sparrows_to_cardinals_ratio_l1698_169821

/-- The number of cardinals Camille saw -/
def cardinals : ℕ := 3

/-- The number of robins Camille saw -/
def robins : ℕ := 4 * cardinals

/-- The number of blue jays Camille saw -/
def blue_jays : ℕ := 2 * cardinals

/-- The total number of birds Camille saw -/
def total_birds : ℕ := 31

/-- The number of sparrows Camille saw -/
def sparrows : ℕ := total_birds - (cardinals + robins + blue_jays)

theorem sparrows_to_cardinals_ratio :
  (sparrows : ℚ) / cardinals = 10 / 3 := by sorry

end sparrows_to_cardinals_ratio_l1698_169821


namespace tobys_money_l1698_169877

/-- 
Proves that if Toby gives 1/7 of his money to each of his two brothers 
and is left with $245, then the initial amount of money he received was $343.
-/
theorem tobys_money (initial_amount : ℚ) : 
  (initial_amount * (1 - 2 * (1 / 7)) = 245) → initial_amount = 343 := by
  sorry

end tobys_money_l1698_169877


namespace isosceles_triangle_area_bounds_l1698_169869

-- Define the area function S
noncomputable def S (α : Real) : Real :=
  let β := α / 2
  let r := Real.sqrt (1 / (2 * Real.tan β))
  let a := r * (1 + Real.sin β) / Real.cos β
  let b := if β ≤ Real.pi / 4 then r * Real.sin (2 * β) else r
  (b / a) ^ 2

-- State the theorem
theorem isosceles_triangle_area_bounds :
  ∀ α : Real, Real.pi / 3 ≤ α ∧ α ≤ 2 * Real.pi / 3 →
    (1 / 4 : Real) ≥ S α ∧ S α ≥ 7 - 4 * Real.sqrt 3 := by
  sorry

end isosceles_triangle_area_bounds_l1698_169869


namespace division_problem_l1698_169813

theorem division_problem (divisor quotient remainder dividend : ℕ) 
  (h1 : divisor = 21)
  (h2 : remainder = 7)
  (h3 : dividend = 301)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 14 := by
sorry

end division_problem_l1698_169813


namespace min_sum_squares_min_sum_squares_achieved_l1698_169836

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : 2*x₁ + 3*x₂ + 4*x₃ = 120) : 
  x₁^2 + x₂^2 + x₃^2 ≥ 14400/29 := by
  sorry

theorem min_sum_squares_achieved (ε : ℝ) (h_pos : ε > 0) : 
  ∃ x₁ x₂ x₃ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ 
  2*x₁ + 3*x₂ + 4*x₃ = 120 ∧ 
  x₁^2 + x₂^2 + x₃^2 < 14400/29 + ε := by
  sorry

end min_sum_squares_min_sum_squares_achieved_l1698_169836


namespace contrapositive_relation_l1698_169882

theorem contrapositive_relation (p r s : Prop) :
  (¬p ↔ r) → (r → s) → (s ↔ (¬p → p)) :=
by sorry

end contrapositive_relation_l1698_169882


namespace area_gray_quadrilateral_l1698_169888

/-- The Stomachion puzzle square --/
def stomachion_square : ℝ := 12

/-- The length of side AB in the gray quadrilateral --/
def side_AB : ℝ := 6

/-- The height of triangle ABD --/
def height_ABD : ℝ := 3

/-- The length of side BC in the gray quadrilateral --/
def side_BC : ℝ := 3

/-- The height of triangle BCD --/
def height_BCD : ℝ := 2

/-- The area of the gray quadrilateral ABCD in the Stomachion puzzle --/
theorem area_gray_quadrilateral : 
  (1/2 * side_AB * height_ABD) + (1/2 * side_BC * height_BCD) = 12 := by
  sorry

end area_gray_quadrilateral_l1698_169888


namespace photos_per_remaining_page_l1698_169875

theorem photos_per_remaining_page (total_photos : ℕ) (total_pages : ℕ) 
  (first_15_photos : ℕ) (next_15_photos : ℕ) (following_10_photos : ℕ) :
  total_photos = 500 →
  total_pages = 60 →
  first_15_photos = 3 →
  next_15_photos = 4 →
  following_10_photos = 5 →
  (17 : ℕ) = (total_photos - (15 * first_15_photos + 15 * next_15_photos + 10 * following_10_photos)) / (total_pages - 40) :=
by sorry

end photos_per_remaining_page_l1698_169875


namespace sheila_mwf_hours_l1698_169822

/-- Represents Sheila's work schedule and earnings --/
structure SheilaWork where
  mwf_hours : ℕ  -- Hours worked on Monday, Wednesday, Friday
  tt_hours : ℕ   -- Hours worked on Tuesday, Thursday
  hourly_rate : ℕ -- Hourly rate in dollars
  weekly_earnings : ℕ -- Total weekly earnings in dollars

/-- Theorem stating Sheila's work hours on Monday, Wednesday, and Friday --/
theorem sheila_mwf_hours (s : SheilaWork) 
  (h1 : s.tt_hours = 6)
  (h2 : s.hourly_rate = 11)
  (h3 : s.weekly_earnings = 396)
  (h4 : s.weekly_earnings = s.hourly_rate * (3 * s.mwf_hours + 2 * s.tt_hours)) :
  s.mwf_hours = 8 := by
  sorry

end sheila_mwf_hours_l1698_169822


namespace equation_rearrangement_l1698_169860

theorem equation_rearrangement (x : ℝ) : (x - 5 = 3*x + 7) ↔ (x - 3*x = 7 + 5) := by
  sorry

end equation_rearrangement_l1698_169860


namespace sqrt_of_square_root_7_minus_3_squared_l1698_169804

theorem sqrt_of_square_root_7_minus_3_squared (x : ℝ) :
  Real.sqrt ((Real.sqrt 7 - 3) ^ 2) = 3 - Real.sqrt 7 :=
by sorry

end sqrt_of_square_root_7_minus_3_squared_l1698_169804


namespace marys_animal_count_l1698_169872

/-- The number of animals Mary thought were in the petting zoo -/
def marys_count (actual_count : ℕ) (double_counted : ℕ) (forgotten : ℕ) : ℕ :=
  actual_count + double_counted - forgotten

/-- Theorem stating that Mary thought there were 60 animals in the petting zoo -/
theorem marys_animal_count :
  marys_count 56 7 3 = 60 := by
  sorry

end marys_animal_count_l1698_169872


namespace isosceles_right_triangle_on_parabola_l1698_169816

/-- Given points A and B on the parabola y = -2x^2 forming an isosceles right triangle ABO 
    with O at the origin, prove that the length of OA (equal to OB) is √5 when a = 1. -/
theorem isosceles_right_triangle_on_parabola :
  ∀ (a : ℝ), 
  let A : ℝ × ℝ := (a, -2 * a^2)
  let B : ℝ × ℝ := (-a, -2 * a^2)
  let O : ℝ × ℝ := (0, 0)
  -- A and B are on the parabola y = -2x^2
  (A.2 = -2 * A.1^2 ∧ B.2 = -2 * B.1^2) →
  -- ABO is an isosceles right triangle with right angle at O
  (Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2) = Real.sqrt ((B.1 - O.1)^2 + (B.2 - O.2)^2)) →
  (A.1 - O.1)^2 + (A.2 - O.2)^2 + (B.1 - O.1)^2 + (B.2 - O.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 →
  -- When a = 1, the length of OA (equal to OB) is √5
  a = 1 → Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2) = Real.sqrt 5 :=
by
  sorry


end isosceles_right_triangle_on_parabola_l1698_169816


namespace house_distance_proof_l1698_169828

/-- Represents the position of a house on a straight street -/
structure HousePosition :=
  (position : ℝ)

/-- The distance between two houses -/
def distance (a b : HousePosition) : ℝ :=
  |a.position - b.position|

theorem house_distance_proof
  (A B V G : HousePosition)
  (h1 : distance A B = 600)
  (h2 : distance V G = 600)
  (h3 : distance A G = 3 * distance B V) :
  distance A G = 900 ∨ distance A G = 1800 := by
  sorry


end house_distance_proof_l1698_169828


namespace sport_to_standard_ratio_l1698_169812

/-- The ratio of flavoring to corn syrup to water in the standard formulation -/
def standard_ratio : Fin 3 → ℚ
| 0 => 1
| 1 => 12
| 2 => 30

/-- The ratio of flavoring to corn syrup in the sport formulation is three times that of standard -/
def sport_ratio_multiplier : ℚ := 3

/-- The amount of corn syrup in the sport formulation (in ounces) -/
def sport_corn_syrup : ℚ := 7

/-- The amount of water in the sport formulation (in ounces) -/
def sport_water : ℚ := 105

/-- The ratio of flavoring to water in the sport formulation compared to the standard formulation -/
theorem sport_to_standard_ratio : 
  (sport_corn_syrup / sport_ratio_multiplier / sport_water) / 
  (standard_ratio 0 / standard_ratio 2) = 1 / 2 := by
  sorry

end sport_to_standard_ratio_l1698_169812


namespace mushroom_count_l1698_169825

/-- The number of vegetables Maria needs to cut for her stew -/
def vegetable_counts (potatoes : ℕ) : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ :=
  let carrots := 6 * potatoes
  let onions := 2 * carrots
  let green_beans := onions / 3
  let bell_peppers := 4 * green_beans
  let mushrooms := 3 * bell_peppers
  (potatoes, carrots, onions, green_beans, bell_peppers, mushrooms)

/-- Theorem stating the number of mushrooms Maria needs to cut -/
theorem mushroom_count (potatoes : ℕ) (h : potatoes = 3) :
  (vegetable_counts potatoes).2.2.2.2.2 = 144 := by
  sorry

end mushroom_count_l1698_169825


namespace sixth_result_proof_l1698_169865

theorem sixth_result_proof (total_results : ℕ) (first_group : ℕ) (last_group : ℕ)
  (total_average : ℚ) (first_average : ℚ) (last_average : ℚ)
  (h1 : total_results = 11)
  (h2 : first_group = 6)
  (h3 : last_group = 6)
  (h4 : total_average = 60)
  (h5 : first_average = 58)
  (h6 : last_average = 63) :
  ∃ (sixth_result : ℚ), sixth_result = 66 := by
sorry

end sixth_result_proof_l1698_169865


namespace max_excellent_videos_l1698_169887

/-- A micro-video with likes and expert score -/
structure MicroVideo where
  likes : ℕ
  expertScore : ℕ

/-- Determines if one video is not inferior to another -/
def notInferior (a b : MicroVideo) : Prop :=
  a.likes ≥ b.likes ∨ a.expertScore ≥ b.expertScore

/-- Determines if a video is excellent among a list of videos -/
def isExcellent (v : MicroVideo) (videos : List MicroVideo) : Prop :=
  ∀ u ∈ videos, notInferior v u

/-- The main theorem to prove -/
theorem max_excellent_videos (videos : List MicroVideo) 
  (h : videos.length = 5) :
  ∃ (excellentVideos : List MicroVideo), 
    excellentVideos.length ≤ 5 ∧ 
    ∀ v ∈ excellentVideos, isExcellent v videos ∧
    ∀ v ∈ videos, isExcellent v videos → v ∈ excellentVideos :=
  sorry

end max_excellent_videos_l1698_169887


namespace doctor_lawyer_ratio_l1698_169856

theorem doctor_lawyer_ratio (total : ℕ) (avg_all avg_doc avg_law : ℚ) 
  (h_total : total = 50)
  (h_avg_all : avg_all = 50)
  (h_avg_doc : avg_doc = 45)
  (h_avg_law : avg_law = 60) :
  ∃ (num_doc num_law : ℕ),
    num_doc + num_law = total ∧
    (avg_doc * num_doc + avg_law * num_law : ℚ) / total = avg_all ∧
    2 * num_law = num_doc :=
by sorry

end doctor_lawyer_ratio_l1698_169856


namespace polynomial_derivative_sum_l1698_169870

theorem polynomial_derivative_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x - 3)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 10 :=
by
  sorry

end polynomial_derivative_sum_l1698_169870


namespace no_valid_placement_l1698_169861

-- Define the chessboard
def Chessboard := Fin 8 × Fin 8

-- Define the piece types
inductive Piece
| Rook
| Knight
| Bishop

-- Define the placement function type
def Placement := Chessboard → Piece

-- Define the attack relations
def rook_attacks (a b : Chessboard) : Prop :=
  (a.1 = b.1 ∨ a.2 = b.2) ∧ a ≠ b

def knight_attacks (a b : Chessboard) : Prop :=
  (abs (a.1 - b.1) = 1 ∧ abs (a.2 - b.2) = 2) ∨
  (abs (a.1 - b.1) = 2 ∧ abs (a.2 - b.2) = 1)

def bishop_attacks (a b : Chessboard) : Prop :=
  abs (a.1 - b.1) = abs (a.2 - b.2) ∧ a ≠ b

-- Define the validity of a placement
def valid_placement (p : Placement) : Prop :=
  ∀ a b : Chessboard,
    (p a = Piece.Rook ∧ rook_attacks a b → p b = Piece.Knight) ∧
    (p a = Piece.Knight ∧ knight_attacks a b → p b = Piece.Bishop) ∧
    (p a = Piece.Bishop ∧ bishop_attacks a b → p b = Piece.Rook)

-- Theorem statement
theorem no_valid_placement : ¬∃ p : Placement, valid_placement p :=
  sorry

end no_valid_placement_l1698_169861


namespace abs_sum_inequality_l1698_169830

theorem abs_sum_inequality (x y z : ℝ) : 
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| := by
  sorry

end abs_sum_inequality_l1698_169830


namespace square_difference_minus_difference_l1698_169854

theorem square_difference_minus_difference (a b : ℤ) : 
  ((a + b)^2 - (a - b)^2) - (a - b) = 4*a*b - (a - b) := by
  sorry

end square_difference_minus_difference_l1698_169854


namespace polynomial_factorization_l1698_169844

theorem polynomial_factorization (a b c : ℂ) :
  let ω : ℂ := Complex.exp ((2 * Real.pi * Complex.I) / 3)
  a^3 + b^3 + c^3 - 3*a*b*c = (a + b + c) * (a + ω*b + ω^2*c) * (a + ω^2*b + ω*c) := by
  sorry

end polynomial_factorization_l1698_169844


namespace prob_not_blue_twelve_sided_die_l1698_169814

-- Define the die
structure Die :=
  (sides : ℕ)
  (red_faces : ℕ)
  (yellow_faces : ℕ)
  (green_faces : ℕ)
  (blue_faces : ℕ)

-- Define the specific die from the problem
def twelve_sided_die : Die :=
  { sides := 12
  , red_faces := 5
  , yellow_faces := 4
  , green_faces := 2
  , blue_faces := 1 }

-- Define the probability of not rolling a blue face
def prob_not_blue (d : Die) : ℚ :=
  (d.sides - d.blue_faces) / d.sides

-- Theorem statement
theorem prob_not_blue_twelve_sided_die :
  prob_not_blue twelve_sided_die = 11 / 12 := by
  sorry

end prob_not_blue_twelve_sided_die_l1698_169814


namespace sqrt_mixed_number_to_fraction_l1698_169803

theorem sqrt_mixed_number_to_fraction :
  Real.sqrt (8 + 9 / 16) = Real.sqrt 137 / 4 :=
by sorry

end sqrt_mixed_number_to_fraction_l1698_169803


namespace log_intersection_and_exponential_inequality_l1698_169858

-- Define the natural logarithm function
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Define the inverse function of f (exponential function)
noncomputable def g (x : ℝ) : ℝ := Real.exp x

theorem log_intersection_and_exponential_inequality :
  (∃! x : ℝ, f x = x - 1) ∧
  (∀ m n : ℝ, m < n → (g n - g m) / (n - m) > g ((m + n) / 2)) :=
by sorry

end log_intersection_and_exponential_inequality_l1698_169858


namespace discount_tax_equivalence_l1698_169880

theorem discount_tax_equivalence (original_price : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) :
  let discounted_price := original_price * (1 - discount_rate)
  let taxed_price := original_price * (1 + tax_rate)
  discounted_price * (1 + tax_rate) = taxed_price * (1 - discount_rate) :=
by sorry

#check discount_tax_equivalence 90 0.2 0.06

end discount_tax_equivalence_l1698_169880


namespace unique_solution_implies_a_half_l1698_169843

/-- Given a positive real number a, if the equation x² - 2ax - 2a ln x = 0
    has a unique solution in the interval (0, +∞), then a = 1/2. -/
theorem unique_solution_implies_a_half (a : ℝ) (ha : a > 0) :
  (∃! x : ℝ, x > 0 ∧ x^2 - 2*a*x - 2*a*(Real.log x) = 0) → a = 1/2 := by
  sorry

end unique_solution_implies_a_half_l1698_169843


namespace abs_c_value_l1698_169848

def polynomial (a b c : ℤ) (x : ℂ) : ℂ :=
  a * x^4 + b * x^3 + c * x^2 + b * x + a

theorem abs_c_value (a b c : ℤ) : 
  polynomial a b c (3 - Complex.I) = 0 →
  Int.gcd a (Int.gcd b c) = 1 →
  |c| = 129 :=
sorry

end abs_c_value_l1698_169848


namespace line_point_sum_l1698_169837

/-- The line equation y = -5/3x + 15 -/
def line_equation (x y : ℝ) : Prop := y = -5/3 * x + 15

/-- Point P is on the x-axis -/
def P : ℝ × ℝ := (9, 0)

/-- Point Q is on the y-axis -/
def Q : ℝ × ℝ := (0, 15)

/-- Point T is on line segment PQ -/
def T_on_PQ (r s : ℝ) : Prop := ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
  r = t * P.1 + (1 - t) * Q.1 ∧ 
  s = t * P.2 + (1 - t) * Q.2

/-- The area of triangle POQ is 4 times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop := 
  abs ((P.1 - 0) * (Q.2 - 0) - (Q.1 - 0) * (P.2 - 0)) / 2 = 
  4 * abs ((P.1 - 0) * (s - 0) - (r - 0) * (P.2 - 0)) / 2

/-- Theorem statement -/
theorem line_point_sum (r s : ℝ) : 
  line_equation r s → T_on_PQ r s → area_condition r s → r + s = 10.5 := by
  sorry

end line_point_sum_l1698_169837


namespace double_inverse_g_10_l1698_169849

-- Define the function g
def g (x : ℝ) : ℝ := x^2 + 2*x + 1

-- Define the inverse function of g
noncomputable def g_inv (y : ℝ) : ℝ := Real.sqrt y - 1

-- Theorem statement
theorem double_inverse_g_10 :
  g_inv (g_inv 10) = Real.sqrt (Real.sqrt 10 - 1) - 1 :=
by sorry

end double_inverse_g_10_l1698_169849


namespace floor_plus_self_eq_14_4_l1698_169857

theorem floor_plus_self_eq_14_4 :
  ∃! r : ℝ, ⌊r⌋ + r = 14.4 := by sorry

end floor_plus_self_eq_14_4_l1698_169857


namespace monkeys_on_different_ladders_l1698_169859

/-- Represents a ladder in the system -/
structure Ladder where
  id : Nat

/-- Represents a monkey in the system -/
structure Monkey where
  id : Nat
  currentLadder : Ladder

/-- Represents a rope connecting two ladders -/
structure Rope where
  ladder1 : Ladder
  ladder2 : Ladder
  height1 : Nat
  height2 : Nat

/-- Represents the state of the system -/
structure MonkeyLadderSystem where
  n : Nat
  ladders : List Ladder
  monkeys : List Monkey
  ropes : List Rope

/-- Predicate to check if all monkeys are on different ladders -/
def allMonkeysOnDifferentLadders (system : MonkeyLadderSystem) : Prop :=
  ∀ m1 m2 : Monkey, m1 ∈ system.monkeys → m2 ∈ system.monkeys → m1 ≠ m2 →
    m1.currentLadder ≠ m2.currentLadder

/-- The main theorem stating that all monkeys end up on different ladders -/
theorem monkeys_on_different_ladders (system : MonkeyLadderSystem) 
    (h1 : system.n > 0)
    (h2 : system.ladders.length = system.n)
    (h3 : system.monkeys.length = system.n)
    (h4 : ∀ m : Monkey, m ∈ system.monkeys → m.currentLadder ∈ system.ladders)
    (h5 : ∀ r : Rope, r ∈ system.ropes → r.ladder1 ∈ system.ladders ∧ r.ladder2 ∈ system.ladders)
    (h6 : ∀ r : Rope, r ∈ system.ropes → r.ladder1 ≠ r.ladder2)
    (h7 : ∀ r1 r2 : Rope, r1 ∈ system.ropes → r2 ∈ system.ropes → r1 ≠ r2 →
      (r1.ladder1 = r2.ladder1 → r1.height1 ≠ r2.height1) ∧
      (r1.ladder2 = r2.ladder2 → r1.height2 ≠ r2.height2))
    : allMonkeysOnDifferentLadders system :=
  sorry

end monkeys_on_different_ladders_l1698_169859


namespace polygon_interior_angles_sum_l1698_169834

theorem polygon_interior_angles_sum (n : ℕ) : 
  (180 * (n - 2) = 1980) → (180 * ((n + 4) - 2) = 2700) := by
  sorry

end polygon_interior_angles_sum_l1698_169834


namespace number_solution_l1698_169898

theorem number_solution : ∃ x : ℝ, 3 * x - 5 = 40 ∧ x = 15 := by
  sorry

end number_solution_l1698_169898


namespace profit_in_scientific_notation_l1698_169807

theorem profit_in_scientific_notation :
  (74.5 : ℝ) * 1000000000 = 7.45 * (10 : ℝ)^9 :=
by sorry

end profit_in_scientific_notation_l1698_169807


namespace complex_arithmetic_simplification_l1698_169892

theorem complex_arithmetic_simplification :
  57.6 * (8 / 5) + 28.8 * (184 / 5) - 14.4 * 80 + 12.5 = 12.5 := by
  sorry

end complex_arithmetic_simplification_l1698_169892


namespace rationalize_denominator_l1698_169895

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (3 : ℝ) / (2 * Real.sqrt 7 + 3 * Real.sqrt 13) = (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    B < D ∧
    A = -6 ∧ B = 7 ∧ C = -9 ∧ D = 13 ∧ E = 89 ∧
    Int.gcd (Int.gcd A C) E = 1 :=
by sorry

end rationalize_denominator_l1698_169895


namespace quadratic_factorization_l1698_169864

theorem quadratic_factorization (m n : ℝ) : 
  (∃ (x : ℝ), x^2 - m*x + n = 0) ∧ 
  (3 : ℝ)^2 - m*(3 : ℝ) + n = 0 ∧ 
  (-4 : ℝ)^2 - m*(-4 : ℝ) + n = 0 →
  ∀ (x : ℝ), x^2 - m*x + n = (x - 3)*(x + 4) :=
by sorry

end quadratic_factorization_l1698_169864


namespace unique_solution_to_diophantine_equation_l1698_169815

theorem unique_solution_to_diophantine_equation :
  ∃! (x y z n : ℕ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧ n ≥ 2 ∧
    z ≤ 5 * 2^(2*n) ∧
    x^(2*n + 1) - y^(2*n + 1) = x * y * z + 2^(2*n + 1) ∧
    x = 3 ∧ y = 1 ∧ z = 70 ∧ n = 2 := by
  sorry

end unique_solution_to_diophantine_equation_l1698_169815


namespace square_sum_value_l1698_169840

theorem square_sum_value (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 := by
  sorry

end square_sum_value_l1698_169840


namespace fermat_numbers_not_cubes_l1698_169893

theorem fermat_numbers_not_cubes : ∀ (n : ℕ), ¬ ∃ (k : ℤ), 2^(2^n) + 1 = k^3 := by
  sorry

end fermat_numbers_not_cubes_l1698_169893


namespace simple_interest_time_l1698_169818

/-- Proves that the time for simple interest is 3 years under given conditions -/
theorem simple_interest_time (P : ℝ) (r : ℝ) (compound_principal : ℝ) (compound_time : ℝ) : 
  P = 1400.0000000000014 →
  r = 0.10 →
  compound_principal = 4000 →
  compound_time = 2 →
  P * r * 3 = (compound_principal * ((1 + r) ^ compound_time - 1)) / 2 →
  3 = (((compound_principal * ((1 + r) ^ compound_time - 1)) / 2) / (P * r)) :=
by sorry

end simple_interest_time_l1698_169818


namespace only_set_b_is_right_triangle_l1698_169879

-- Define the sets of numbers
def set_a : List ℕ := [2, 3, 4]
def set_b : List ℕ := [3, 4, 5]
def set_c : List ℕ := [5, 6, 7]
def set_d : List ℕ := [7, 8, 9]

-- Define a function to check if a set of three numbers satisfies the Pythagorean theorem
def is_right_triangle (sides : List ℕ) : Prop :=
  match sides with
  | [a, b, c] => a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2
  | _ => False

-- Theorem statement
theorem only_set_b_is_right_triangle :
  ¬(is_right_triangle set_a) ∧
  (is_right_triangle set_b) ∧
  ¬(is_right_triangle set_c) ∧
  ¬(is_right_triangle set_d) :=
by sorry

end only_set_b_is_right_triangle_l1698_169879


namespace compounded_ratio_is_two_to_one_l1698_169832

/-- The compounded ratio of three given ratios -/
def compounded_ratio (r1 r2 r3 : Rat × Rat) : Rat × Rat :=
  let (a1, b1) := r1
  let (a2, b2) := r2
  let (a3, b3) := r3
  (a1 * a2 * a3, b1 * b2 * b3)

/-- The given ratios -/
def ratio1 : Rat × Rat := (2, 3)
def ratio2 : Rat × Rat := (6, 11)
def ratio3 : Rat × Rat := (11, 2)

/-- The theorem stating that the compounded ratio of the given ratios is 2:1 -/
theorem compounded_ratio_is_two_to_one :
  compounded_ratio ratio1 ratio2 ratio3 = (2, 1) := by
  sorry

end compounded_ratio_is_two_to_one_l1698_169832


namespace smallest_base_perfect_square_l1698_169862

theorem smallest_base_perfect_square : 
  ∃ (b : ℕ), b > 4 ∧ 
  (∃ (n : ℕ), 3 * b + 4 = n^2) ∧
  (∀ (k : ℕ), k > 4 ∧ k < b → ¬∃ (m : ℕ), 3 * k + 4 = m^2) ∧
  b = 7 := by
sorry

end smallest_base_perfect_square_l1698_169862


namespace jake_has_fewer_than_19_peaches_l1698_169876

/-- The number of peaches each person has -/
structure PeachCount where
  steven : ℕ
  jill : ℕ
  jake : ℕ

/-- The given conditions -/
def peach_conditions (p : PeachCount) : Prop :=
  p.steven = 19 ∧
  p.jill = 6 ∧
  p.steven = p.jill + 13 ∧
  p.jake < p.steven

/-- Theorem: Jake has fewer than 19 peaches -/
theorem jake_has_fewer_than_19_peaches (p : PeachCount) 
  (h : peach_conditions p) : p.jake < 19 := by
  sorry

end jake_has_fewer_than_19_peaches_l1698_169876


namespace nested_radical_value_l1698_169810

/-- The value of the infinite nested radical √(15 + √(15 + √(15 + ...))) -/
noncomputable def nestedRadical : ℝ :=
  Real.sqrt (15 + Real.sqrt (15 + Real.sqrt (15 + Real.sqrt (15 + Real.sqrt 15))))

/-- Theorem stating that the nested radical equals 5 -/
theorem nested_radical_value : nestedRadical = 5 := by
  sorry

end nested_radical_value_l1698_169810


namespace inequality_proof_l1698_169839

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^2 + b^2 + c^2 = 14) : a^5 + (1/8)*b^5 + (1/27)*c^5 ≥ 14 := by
  sorry

end inequality_proof_l1698_169839


namespace parabola_point_value_l1698_169805

theorem parabola_point_value (a b : ℝ) : 
  (a * (-2)^2 + b * (-2) + 5 = 9) → (2*a - b + 6 = 8) := by
  sorry

end parabola_point_value_l1698_169805


namespace f_min_value_h_unique_zero_l1698_169841

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 / Real.exp x
noncomputable def h (x : ℝ) : ℝ := g x - f (-1) x

-- Theorem for part 1
theorem f_min_value (a : ℝ) (h : a > 0) :
  ∃ (x : ℝ), x > 0 ∧ IsLocalMin (f a) x ∧ f a x = a - a * Real.log a :=
sorry

-- Theorem for part 2
theorem h_unique_zero :
  ∃! (x : ℝ), x ∈ Set.Ioo 0 1 ∧ h x = 0 :=
sorry

end f_min_value_h_unique_zero_l1698_169841


namespace tiffany_homework_problems_l1698_169883

/-- The total number of problems Tiffany had to complete -/
def total_problems (math_pages reading_pages science_pages history_pages : ℕ)
                   (math_problems_per_page reading_problems_per_page science_problems_per_page history_problems_per_page : ℕ) : ℕ :=
  math_pages * math_problems_per_page +
  reading_pages * reading_problems_per_page +
  science_pages * science_problems_per_page +
  history_pages * history_problems_per_page

/-- Theorem stating that the total number of problems is 46 -/
theorem tiffany_homework_problems :
  total_problems 6 4 3 2 3 3 4 2 = 46 := by
  sorry

end tiffany_homework_problems_l1698_169883


namespace largest_prime_factor_of_9911_l1698_169850

theorem largest_prime_factor_of_9911 : ∃ p : ℕ, 
  Nat.Prime p ∧ p ∣ 9911 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 9911 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_9911_l1698_169850


namespace other_sales_percentage_l1698_169890

/-- The percentage of sales for notebooks -/
def notebook_sales : ℝ := 42

/-- The percentage of sales for markers -/
def marker_sales : ℝ := 26

/-- The total percentage of sales -/
def total_sales : ℝ := 100

/-- Theorem: The percentage of sales that were not notebooks or markers is 32% -/
theorem other_sales_percentage :
  total_sales - (notebook_sales + marker_sales) = 32 := by sorry

end other_sales_percentage_l1698_169890


namespace nancy_total_games_l1698_169802

/-- The number of games Nancy attended this month -/
def games_this_month : ℕ := 9

/-- The number of games Nancy attended last month -/
def games_last_month : ℕ := 8

/-- The number of games Nancy plans to attend next month -/
def games_next_month : ℕ := 7

/-- The total number of games Nancy would attend -/
def total_games : ℕ := games_this_month + games_last_month + games_next_month

theorem nancy_total_games : total_games = 24 := by
  sorry

end nancy_total_games_l1698_169802


namespace range_of_x_l1698_169863

theorem range_of_x (x : ℝ) : 
  (0 ≤ x ∧ x < 2 * Real.pi) → 
  (Real.sqrt (1 - Real.sin (2 * x)) = Real.sin x - Real.cos x) →
  (x ∈ Set.Icc (Real.pi / 4) (5 * Real.pi / 4)) :=
by sorry

end range_of_x_l1698_169863


namespace twentyFourthDigitOfSum_l1698_169831

-- Define the decimal representation of a rational number
def decimalRepresentation (q : ℚ) : ℕ → ℕ := sorry

-- Define the sum of two decimal representations
def sumDecimalRepresentations (f g : ℕ → ℕ) : ℕ → ℕ := sorry

-- The main theorem
theorem twentyFourthDigitOfSum :
  let f := decimalRepresentation (1/9 : ℚ)
  let g := decimalRepresentation (1/4 : ℚ)
  let sum := sumDecimalRepresentations f g
  sum 24 = 1 := by sorry

end twentyFourthDigitOfSum_l1698_169831


namespace shaded_area_of_overlapping_sectors_l1698_169823

/-- The area of the shaded region formed by two overlapping sectors of a circle -/
theorem shaded_area_of_overlapping_sectors (r : ℝ) (θ : ℝ) (h_r : r = 15) (h_θ : θ = 45 * π / 180) :
  let sector_area := θ / (2 * π) * π * r^2
  let triangle_area := r^2 * Real.sin θ / 2
  2 * (sector_area - triangle_area) = 56.25 * π - 112.5 * Real.sqrt 2 := by
  sorry

end shaded_area_of_overlapping_sectors_l1698_169823


namespace store_revenue_l1698_169881

def shirt_price : ℝ := 10
def jeans_price : ℝ := 2 * shirt_price
def jacket_price : ℝ := 3 * jeans_price
def discount_rate : ℝ := 0.1

def num_shirts : ℕ := 20
def num_jeans : ℕ := 10
def num_jackets : ℕ := 15

def total_revenue : ℝ :=
  (num_shirts : ℝ) * shirt_price +
  (num_jeans : ℝ) * jeans_price +
  (num_jackets : ℝ) * jacket_price * (1 - discount_rate)

theorem store_revenue :
  total_revenue = 1210 := by sorry

end store_revenue_l1698_169881


namespace coronavirus_diameter_scientific_notation_l1698_169842

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem coronavirus_diameter_scientific_notation :
  toScientificNotation 0.00000012 = ScientificNotation.mk 1.2 (-7) (by norm_num) :=
sorry

end coronavirus_diameter_scientific_notation_l1698_169842


namespace morks_tax_rate_l1698_169852

/-- Given the tax rates and income ratio of Mork and Mindy, prove Mork's tax rate --/
theorem morks_tax_rate (r : ℝ) : 
  (r * 1 + 0.3 * 4) / 5 = 0.32 → r = 0.4 := by sorry

end morks_tax_rate_l1698_169852


namespace car_price_theorem_l1698_169884

def asking_price_proof (P : ℝ) : Prop :=
  let first_offer := (9/10) * P
  let second_offer := P - 320
  (first_offer - second_offer = 200) → (P = 1200)

theorem car_price_theorem :
  ∀ P : ℝ, asking_price_proof P :=
sorry

end car_price_theorem_l1698_169884


namespace frustum_area_relation_l1698_169819

/-- A frustum with base areas S₁ and S₂, and midsection area S₀ -/
structure Frustum where
  S₁ : ℝ
  S₂ : ℝ
  S₀ : ℝ
  h_positive : S₁ > 0 ∧ S₂ > 0 ∧ S₀ > 0

theorem frustum_area_relation (f : Frustum) : 2 * Real.sqrt f.S₀ = Real.sqrt f.S₁ + Real.sqrt f.S₂ := by
  sorry

end frustum_area_relation_l1698_169819


namespace sine_inequality_l1698_169871

theorem sine_inequality (y : Real) :
  (y ∈ Set.Icc 0 (Real.pi / 2)) ↔
  (∀ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), Real.sin (x + y) ≤ Real.sin x + Real.sin y) :=
by sorry

end sine_inequality_l1698_169871


namespace homework_time_ratio_l1698_169817

theorem homework_time_ratio :
  ∀ (geog_time : ℝ) (sci_time : ℝ),
    geog_time > 0 →
    sci_time = (60 + geog_time) / 2 →
    60 + geog_time + sci_time = 135 →
    geog_time / 60 = 1 / 2 := by
  sorry

end homework_time_ratio_l1698_169817


namespace target_hit_probability_l1698_169874

theorem target_hit_probability (p_A p_B p_C : ℝ) 
  (h_A : p_A = 1/2) 
  (h_B : p_B = 1/3) 
  (h_C : p_C = 1/4) : 
  1 - (1 - p_A) * (1 - p_B) * (1 - p_C) = 3/4 := by
  sorry

end target_hit_probability_l1698_169874


namespace line_passes_through_parabola_vertex_l1698_169838

/-- The number of values of 'a' for which the line y = x + a passes through
    the vertex of the parabola y = x^2 + a^2 is exactly 2. -/
theorem line_passes_through_parabola_vertex :
  let line := λ (x a : ℝ) => x + a
  let parabola := λ (x a : ℝ) => x^2 + a^2
  let vertex := λ (a : ℝ) => (0, a^2)
  ∃! (s : Finset ℝ), (∀ a ∈ s, line 0 a = (vertex a).2) ∧ s.card = 2 :=
by sorry

end line_passes_through_parabola_vertex_l1698_169838


namespace expression_simplification_l1698_169866

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (x + 1) / x / (x - (1 + x^2) / (2 * x)) = Real.sqrt 2 := by
  sorry

end expression_simplification_l1698_169866


namespace necessary_but_not_sufficient_l1698_169894

theorem necessary_but_not_sufficient :
  let p := fun x : ℝ => x^2 - 2*x ≥ 3
  let q := fun x : ℝ => -1 < x ∧ x < 2
  (∀ x, q x → ¬(p x)) ∧ 
  (∃ x, ¬(p x) ∧ ¬(q x)) :=
by sorry

end necessary_but_not_sufficient_l1698_169894


namespace solution_set_f_leq_15_max_a_for_inequality_l1698_169835

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Theorem for the solution set of f(x) ≤ 15
theorem solution_set_f_leq_15 :
  {x : ℝ | f x ≤ 15} = Set.Icc (-8 : ℝ) 7 := by sorry

-- Theorem for the maximum value of a
theorem max_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, -x^2 + a ≤ f x) ↔ a ≤ 5 := by sorry

end solution_set_f_leq_15_max_a_for_inequality_l1698_169835


namespace new_person_weight_l1698_169800

/-- Given a group of 8 persons where replacing one person weighing 35 kg
    with a new person increases the average weight by 5 kg,
    prove that the weight of the new person is 75 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 5 →
  replaced_weight = 35 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 75 :=
by sorry

end new_person_weight_l1698_169800


namespace abs_inequality_solution_set_l1698_169811

theorem abs_inequality_solution_set (x : ℝ) : 
  abs (2 * x - 1) < abs x + 1 ↔ 0 < x ∧ x < 2 := by
  sorry

end abs_inequality_solution_set_l1698_169811


namespace mean_median_difference_l1698_169855

theorem mean_median_difference (x : ℕ) : 
  let s := [x, x + 2, x + 4, x + 7, x + 27]
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 27)) / 5
  let median := x + 4
  mean = median + 4 := by
  sorry

end mean_median_difference_l1698_169855


namespace variance_of_surviving_trees_l1698_169891

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

end variance_of_surviving_trees_l1698_169891


namespace least_distinct_values_in_list_l1698_169833

/-- Given a list of 2520 positive integers with a unique mode occurring exactly 12 times,
    the least number of distinct values in the list is 229. -/
theorem least_distinct_values_in_list :
  ∀ (L : List ℕ+) (mode : ℕ+),
    L.length = 2520 →
    (∃! x, x ∈ L ∧ L.count x = 12) →
    (∃ x, x ∈ L ∧ L.count x = 12) →
    (∀ x, x ∈ L → L.count x ≤ 12) →
    L.toFinset.card ≥ 229 :=
by sorry

end least_distinct_values_in_list_l1698_169833


namespace cubic_roots_proof_l1698_169886

theorem cubic_roots_proof (k : ℝ) (p q r : ℝ) : 
  (2 * p^3 + k * p^2 - 6 * p - 3 = 0) →
  (p = 3) →
  (p + q + r = 5) →
  (p * q * r = -6) →
  ({q, r} : Set ℝ) = {1 + Real.sqrt 3, 1 - Real.sqrt 3} :=
by sorry

end cubic_roots_proof_l1698_169886


namespace point_quadrant_relation_l1698_169827

/-- If P(a,b) is in the second quadrant, then Q(-b,a-3) is in the third quadrant -/
theorem point_quadrant_relation (a b : ℝ) : 
  (a < 0 ∧ b > 0) → (-b < 0 ∧ a - 3 < 0) := by
  sorry

end point_quadrant_relation_l1698_169827


namespace polynomial_coefficients_sum_l1698_169806

theorem polynomial_coefficients_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x : ℝ, (2*x - 1)^5 + (x + 2)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  |a₀| + |a₂| + |a₄| = 110 := by
sorry

end polynomial_coefficients_sum_l1698_169806
