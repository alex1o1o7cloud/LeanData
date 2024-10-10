import Mathlib

namespace impossible_to_flip_all_l68_6828

/-- Represents the color of a button's face -/
inductive ButtonColor
| White
| Black

/-- Represents a configuration of buttons in a circle -/
def ButtonConfiguration := List ButtonColor

/-- Represents a move in the game -/
inductive Move
| FlipAdjacent (i : Nat)  -- Flip two adjacent buttons at position i and i+1
| FlipSeparated (i : Nat) -- Flip two buttons at position i and i+2

/-- The initial configuration of buttons -/
def initial_config : ButtonConfiguration :=
  [ButtonColor.Black] ++ List.replicate 2021 ButtonColor.White

/-- Applies a move to a button configuration -/
def apply_move (config : ButtonConfiguration) (move : Move) : ButtonConfiguration :=
  sorry

/-- Checks if all buttons have been flipped from their initial state -/
def all_flipped (config : ButtonConfiguration) : Prop :=
  sorry

/-- The main theorem stating it's impossible to flip all buttons -/
theorem impossible_to_flip_all (moves : List Move) :
  ¬(all_flipped (moves.foldl apply_move initial_config)) :=
sorry

end impossible_to_flip_all_l68_6828


namespace find_n_l68_6899

theorem find_n : ∃ n : ℤ, (11 : ℝ) ^ (4 * n) = (1 / 11 : ℝ) ^ (n - 30) → n = 6 := by
  sorry

end find_n_l68_6899


namespace product_five_cubed_sum_l68_6868

theorem product_five_cubed_sum (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 5^3 →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 31 := by sorry

end product_five_cubed_sum_l68_6868


namespace parallel_line_through_point_l68_6810

/-- Given a line L1 with equation 3x - y = 6 and a point P (-2, 3),
    prove that the line L2 with equation y = 3x + 9 is parallel to L1 and passes through P. -/
theorem parallel_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 3 * x - y = 6
  let P : ℝ × ℝ := (-2, 3)
  let L2 : ℝ → ℝ → Prop := λ x y => y = 3 * x + 9
  (∀ x y, L1 x y ↔ y = 3 * x - 6) →  -- L1 in slope-intercept form
  L2 P.1 P.2 ∧                      -- L2 passes through P
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → y₂ - y₁ = 3 * (x₂ - x₁)) →  -- Slope of L1 is 3
  (∀ x₁ y₁ x₂ y₂, L2 x₁ y₁ → L2 x₂ y₂ → y₂ - y₁ = 3 * (x₂ - x₁))    -- Slope of L2 is 3
  :=
by sorry

end parallel_line_through_point_l68_6810


namespace roger_lawn_mowing_earnings_l68_6845

theorem roger_lawn_mowing_earnings :
  ∀ (total_lawns : ℕ) (forgotten_lawns : ℕ) (total_earnings : ℕ),
    total_lawns = 14 →
    forgotten_lawns = 8 →
    total_earnings = 54 →
    (total_earnings : ℚ) / ((total_lawns - forgotten_lawns) : ℚ) = 9 := by
  sorry

end roger_lawn_mowing_earnings_l68_6845


namespace stratified_sampling_theorem_l68_6894

/-- Represents a college with a name and number of officers -/
structure College where
  name : String
  officers : Nat

/-- Represents the result of stratified sampling -/
structure SamplingResult where
  m : Nat
  n : Nat
  s : Nat

/-- Calculates the stratified sampling result -/
def stratifiedSampling (colleges : List College) (totalSample : Nat) : SamplingResult :=
  sorry

/-- Calculates the probability of selecting two officers from the same college -/
def probabilitySameCollege (result : SamplingResult) : Rat :=
  sorry

theorem stratified_sampling_theorem (m n s : College) (h1 : m.officers = 36) (h2 : n.officers = 24) (h3 : s.officers = 12) :
  let colleges := [m, n, s]
  let result := stratifiedSampling colleges 6
  result.m = 3 ∧ result.n = 2 ∧ result.s = 1 ∧ probabilitySameCollege result = 4/15 := by
  sorry

end stratified_sampling_theorem_l68_6894


namespace courtyard_path_ratio_l68_6844

theorem courtyard_path_ratio :
  ∀ (t p : ℝ),
  t > 0 →
  p > 0 →
  (400 * t^2) / (400 * (t + 2*p)^2) = 1/4 →
  p/t = 1/2 := by
sorry

end courtyard_path_ratio_l68_6844


namespace fraction_simplification_l68_6874

theorem fraction_simplification (d : ℝ) : 
  (5 + 4*d) / 9 - 3 + 1/3 = (4*d - 19) / 9 := by
  sorry

end fraction_simplification_l68_6874


namespace x_equals_two_l68_6862

/-- The sum of digits for all four-digit numbers formed by 1, 4, 5, and x -/
def sumOfDigits (x : ℕ) : ℕ :=
  if x = 0 then
    24 * (1 + 4 + 5)
  else
    24 * (1 + 4 + 5 + x)

/-- Theorem stating that x must be 2 given the conditions -/
theorem x_equals_two :
  ∃! x : ℕ, x ≤ 9 ∧ sumOfDigits x = 288 :=
sorry

end x_equals_two_l68_6862


namespace div_power_equals_power_diff_l68_6827

theorem div_power_equals_power_diff (a : ℝ) (h : a ≠ 0) : a^3 / a^2 = a := by
  sorry

end div_power_equals_power_diff_l68_6827


namespace ellipse_hyperbola_ratio_l68_6814

/-- Given an ellipse and a hyperbola with common foci, prove that the ratio of their minor axes is √3 -/
theorem ellipse_hyperbola_ratio (a₁ b₁ a₂ b₂ c : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  a₁ > b₁ ∧ b₁ > 0 ∧ a₂ > 0 ∧ b₂ > 0 →
  P.1^2 / a₁^2 + P.2^2 / b₁^2 = 1 →
  P.1^2 / a₂^2 - P.2^2 / b₂^2 = 1 →
  (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 4 * c^2 →
  a₁^2 - b₁^2 = c^2 →
  a₂^2 + b₂^2 = c^2 →
  Real.cos (Real.arccos ((((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2).sqrt + ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2).sqrt)^2 - 4*c^2) /
    (2 * ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2).sqrt * ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2).sqrt)) = 1/2 →
  b₁ / b₂ = Real.sqrt 3 := by
sorry

end ellipse_hyperbola_ratio_l68_6814


namespace max_area_difference_l68_6812

/-- Definition of the ellipse -/
def Ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- One focus of the ellipse -/
def Focus : ℝ × ℝ := (-1, 0)

/-- S₁ is the area of triangle ABD -/
noncomputable def S₁ (A B D : ℝ × ℝ) : ℝ := sorry

/-- S₂ is the area of triangle ABC -/
noncomputable def S₂ (A B C : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the maximum difference between S₁ and S₂ -/
theorem max_area_difference :
  ∃ (A B C D : ℝ × ℝ),
    Ellipse A.1 A.2 ∧ Ellipse B.1 B.2 ∧ Ellipse C.1 C.2 ∧ Ellipse D.1 D.2 ∧
    (∀ (E : ℝ × ℝ), Ellipse E.1 E.2 → |S₁ A B D - S₂ A B C| ≤ Real.sqrt 3) ∧
    (∃ (F G : ℝ × ℝ), Ellipse F.1 F.2 ∧ Ellipse G.1 G.2 ∧ |S₁ A B F - S₂ A B G| = Real.sqrt 3) :=
sorry

end max_area_difference_l68_6812


namespace sum_of_reciprocals_l68_6817

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 375) :
  1 / x + 1 / y = 8 / 75 := by
  sorry

end sum_of_reciprocals_l68_6817


namespace average_age_after_leaving_l68_6856

theorem average_age_after_leaving (initial_people : ℕ) (initial_average : ℚ) 
  (leaving_age : ℕ) (remaining_people : ℕ) :
  initial_people = 6 →
  initial_average = 28 →
  leaving_age = 22 →
  remaining_people = 5 →
  (initial_people * initial_average - leaving_age) / remaining_people = 29.2 := by
  sorry

end average_age_after_leaving_l68_6856


namespace reflection_of_M_l68_6801

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The point M -/
def M : ℝ × ℝ := (5, 2)

/-- Theorem: The reflection of M(5, 2) across the x-axis is (5, -2) -/
theorem reflection_of_M : reflect_x M = (5, -2) := by sorry

end reflection_of_M_l68_6801


namespace parabola_has_one_x_intercept_l68_6881

/-- The equation of the parabola -/
def parabola_equation (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 4

/-- A point (x, y) is on the parabola if it satisfies the equation -/
def on_parabola (x y : ℝ) : Prop := x = parabola_equation y

/-- An x-intercept is a point on the parabola where y = 0 -/
def is_x_intercept (x : ℝ) : Prop := on_parabola x 0

/-- The theorem stating that the parabola has exactly one x-intercept -/
theorem parabola_has_one_x_intercept : ∃! x : ℝ, is_x_intercept x := by sorry

end parabola_has_one_x_intercept_l68_6881


namespace pulley_system_theorem_l68_6838

/-- Represents the configuration of three pulleys --/
structure PulleySystem where
  r : ℝ  -- radius of pulleys
  d12 : ℝ  -- distance between O₁ and O₂
  d13 : ℝ  -- distance between O₁ and O₃
  h : ℝ  -- height of O₃ above the plane of O₁ and O₂

/-- Calculates the possible belt lengths for the pulley system --/
def beltLengths (p : PulleySystem) : Set ℝ :=
  { 32 + 4 * Real.pi, 22 + 2 * Real.sqrt 97 + 4 * Real.pi }

/-- Checks if a given cord length is always sufficient --/
def isAlwaysSufficient (p : PulleySystem) (cordLength : ℝ) : Prop :=
  ∀ l ∈ beltLengths p, l ≤ cordLength

theorem pulley_system_theorem (p : PulleySystem) 
    (h1 : p.r = 2)
    (h2 : p.d12 = 12)
    (h3 : p.d13 = 10)
    (h4 : p.h = 8) :
    (beltLengths p = { 32 + 4 * Real.pi, 22 + 2 * Real.sqrt 97 + 4 * Real.pi }) ∧
    (¬ isAlwaysSufficient p 54) := by
  sorry

end pulley_system_theorem_l68_6838


namespace sqrt_nine_minus_two_power_zero_plus_abs_negative_one_equals_three_l68_6883

theorem sqrt_nine_minus_two_power_zero_plus_abs_negative_one_equals_three :
  Real.sqrt 9 - 2^(0 : ℕ) + |(-1 : ℝ)| = 3 := by sorry

end sqrt_nine_minus_two_power_zero_plus_abs_negative_one_equals_three_l68_6883


namespace cookie_problem_l68_6895

/-- The number of guests who did not come to Brenda's mother's cookie event -/
def guests_not_came (total_guests : ℕ) (total_cookies : ℕ) (cookies_per_guest : ℕ) : ℕ :=
  total_guests - (total_cookies / cookies_per_guest)

theorem cookie_problem :
  guests_not_came 10 18 18 = 9 := by
  sorry

end cookie_problem_l68_6895


namespace minimal_coloring_exists_l68_6866

/-- Define the function f for a given set M and subset A -/
def f (M : Finset ℕ) (A : Finset ℕ) : Finset ℕ :=
  M.filter (fun x => (A.filter (fun a => x % a = 0)).card % 2 = 1)

/-- The main theorem -/
theorem minimal_coloring_exists :
  ∀ (M : Finset ℕ), M.card = 2017 →
  ∃ (c : Finset ℕ → Bool),
    ∀ (A : Finset ℕ), A ⊆ M →
      A ≠ f M A → c A ≠ c (f M A) :=
by sorry

end minimal_coloring_exists_l68_6866


namespace next_multiple_year_l68_6824

theorem next_multiple_year : ∀ n : ℕ, 
  n > 2016 ∧ 
  n % 6 = 0 ∧ 
  n % 8 = 0 ∧ 
  n % 9 = 0 → 
  n ≥ 2088 :=
by
  sorry

end next_multiple_year_l68_6824


namespace conference_drinks_l68_6882

theorem conference_drinks (total : ℕ) (coffee : ℕ) (juice : ℕ) (both : ℕ) :
  total = 30 →
  coffee = 15 →
  juice = 18 →
  both = 7 →
  total - (coffee + juice - both) = 4 :=
by sorry

end conference_drinks_l68_6882


namespace sqrt_product_equality_l68_6820

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end sqrt_product_equality_l68_6820


namespace inequality_solution_l68_6855

noncomputable def solution_set : Set ℝ :=
  { x | x ∈ Set.Ioo (-4) (-14/3) ∪ Set.Ioi (6 + 3 * Real.sqrt 2) }

theorem inequality_solution :
  { x : ℝ | (2*x + 3) / (x + 4) > (5*x + 6) / (3*x + 14) } = solution_set :=
by sorry

end inequality_solution_l68_6855


namespace arithmetic_sequence_sum_l68_6865

/-- Given an arithmetic sequence {aₙ}, where Sₙ is the sum of the first n terms,
    prove that S₈ = 80 when a₃ = 20 - a₆ -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) →  -- sum formula
  a 3 = 20 - a 6 →
  S 8 = 80 := by
sorry

end arithmetic_sequence_sum_l68_6865


namespace trigonometric_identity_l68_6837

theorem trigonometric_identity : 
  (Real.sin (160 * π / 180) + Real.sin (40 * π / 180)) * 
  (Real.sin (140 * π / 180) + Real.sin (20 * π / 180)) + 
  (Real.sin (50 * π / 180) - Real.sin (70 * π / 180)) * 
  (Real.sin (130 * π / 180) - Real.sin (110 * π / 180)) = 1 := by
  sorry

end trigonometric_identity_l68_6837


namespace lauren_tuesday_earnings_l68_6842

/-- Represents Lauren's earnings from her social media channel on Tuesday -/
def LaurenEarnings : ℝ → ℝ → ℕ → ℕ → ℝ :=
  λ commercial_rate subscription_rate commercial_views subscriptions =>
    commercial_rate * (commercial_views : ℝ) + subscription_rate * (subscriptions : ℝ)

/-- Theorem stating Lauren's earnings on Tuesday -/
theorem lauren_tuesday_earnings :
  LaurenEarnings 0.5 1 100 27 = 77 := by
  sorry

end lauren_tuesday_earnings_l68_6842


namespace composition_equality_l68_6811

/-- Given two functions f and g, prove that if f(g(b)) = 4, then b = -1/2 -/
theorem composition_equality (f g : ℝ → ℝ) (b : ℝ) 
    (hf : ∀ x, f x = x / 3 + 2)
    (hg : ∀ x, g x = 5 - 2 * x)
    (h : f (g b) = 4) : 
  b = -1/2 := by
sorry

end composition_equality_l68_6811


namespace desired_depth_calculation_desired_depth_is_50_l68_6808

/-- Calculates the desired depth to be dug given the initial and changed conditions -/
theorem desired_depth_calculation (initial_men : ℕ) (initial_hours : ℕ) (initial_depth : ℕ) 
  (extra_men : ℕ) (new_hours : ℕ) : ℕ :=
  let total_men : ℕ := initial_men + extra_men
  let initial_man_hours : ℕ := initial_men * initial_hours
  let new_man_hours : ℕ := total_men * new_hours
  let desired_depth : ℕ := (new_man_hours * initial_depth) / initial_man_hours
  desired_depth

/-- Proves that the desired depth to be dug is 50 meters -/
theorem desired_depth_is_50 : 
  desired_depth_calculation 45 8 30 55 6 = 50 := by
  sorry

end desired_depth_calculation_desired_depth_is_50_l68_6808


namespace total_revenue_calculation_l68_6869

/-- Calculates the total revenue from selling various reading materials -/
theorem total_revenue_calculation (magazines newspapers books pamphlets : ℕ) 
  (magazine_price newspaper_price book_price pamphlet_price : ℚ) : 
  magazines = 425 → 
  newspapers = 275 → 
  books = 150 → 
  pamphlets = 75 → 
  magazine_price = 5/2 → 
  newspaper_price = 3/2 → 
  book_price = 5 → 
  pamphlet_price = 1/2 → 
  (magazines : ℚ) * magazine_price + 
  (newspapers : ℚ) * newspaper_price + 
  (books : ℚ) * book_price + 
  (pamphlets : ℚ) * pamphlet_price = 2262.5 := by
  sorry

end total_revenue_calculation_l68_6869


namespace probability_is_one_seventh_l68_6800

/-- Represents the total number of socks -/
def total_socks : ℕ := 10

/-- Represents the number of colors -/
def num_colors : ℕ := 5

/-- Represents the number of socks per color -/
def socks_per_color : ℕ := 2

/-- Represents the number of socks drawn -/
def socks_drawn : ℕ := 6

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the probability of drawing exactly two pairs of different colors -/
def probability_two_pairs : ℚ :=
  let total_outcomes := choose total_socks socks_drawn
  let favorable_outcomes := choose num_colors 2 * choose (num_colors - 2) 2
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_is_one_seventh :
  probability_two_pairs = 1 / 7 := by
  sorry

end probability_is_one_seventh_l68_6800


namespace nested_fraction_equality_l68_6847

theorem nested_fraction_equality : 
  (2 : ℚ) / (2 + 2 / (3 + 1 / 4)) = 13 / 17 := by sorry

end nested_fraction_equality_l68_6847


namespace school_supplies_problem_l68_6884

/-- Represents the school supplies problem --/
theorem school_supplies_problem 
  (num_students : ℕ) 
  (pens_per_student : ℕ) 
  (notebooks_per_student : ℕ) 
  (binders_per_student : ℕ) 
  (pen_cost : ℚ) 
  (notebook_cost : ℚ) 
  (binder_cost : ℚ) 
  (highlighter_cost : ℚ) 
  (teacher_discount : ℚ) 
  (total_spent : ℚ) 
  (h1 : num_students = 30)
  (h2 : pens_per_student = 5)
  (h3 : notebooks_per_student = 3)
  (h4 : binders_per_student = 1)
  (h5 : pen_cost = 1/2)
  (h6 : notebook_cost = 5/4)
  (h7 : binder_cost = 17/4)
  (h8 : highlighter_cost = 3/4)
  (h9 : teacher_discount = 100)
  (h10 : total_spent = 260) :
  (total_spent - (num_students * (pens_per_student * pen_cost + 
   notebooks_per_student * notebook_cost + 
   binders_per_student * binder_cost) - teacher_discount)) / 
  (num_students * highlighter_cost) = 2 := by
sorry


end school_supplies_problem_l68_6884


namespace combined_tax_rate_l68_6805

/-- The combined tax rate problem -/
theorem combined_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (julie_rate : ℝ) 
  (mindy_income : ℝ → ℝ) 
  (julie_income : ℝ → ℝ) 
  (h1 : mork_rate = 0.45)
  (h2 : mindy_rate = 0.25)
  (h3 : julie_rate = 0.35)
  (h4 : ∀ m, mindy_income m = 4 * m)
  (h5 : ∀ m, julie_income m = 2 * m)
  (h6 : ∀ m, julie_income m = (mindy_income m) / 2) :
  ∀ m : ℝ, m > 0 → 
    (mork_rate * m + mindy_rate * (mindy_income m) + julie_rate * (julie_income m)) / 
    (m + mindy_income m + julie_income m) = 2.15 / 7 := by
  sorry

end combined_tax_rate_l68_6805


namespace f_inequality_range_l68_6823

noncomputable def f (x : ℝ) : ℝ :=
  if x < -1/2 then 2*x + 1
  else if x < 3/2 then 2 - (3-2*x)
  else 2*x + 1 + (2*x-3)

theorem f_inequality_range (a : ℝ) :
  (∃ x, f x < 1) →
  (∀ x, f x ≤ |a|) →
  |a| ≥ 4 := by sorry

end f_inequality_range_l68_6823


namespace jimin_remaining_distance_l68_6846

/-- Calculates the remaining distance to travel given initial conditions. -/
def remaining_distance (speed : ℝ) (time : ℝ) (total_distance : ℝ) : ℝ :=
  total_distance - speed * time

/-- Proves that given the initial conditions, the remaining distance is 180 km. -/
theorem jimin_remaining_distance :
  remaining_distance 60 2 300 = 180 := by
  sorry

end jimin_remaining_distance_l68_6846


namespace water_breadth_in_cistern_l68_6892

/-- Calculates the breadth of water in a cistern given its dimensions and wet surface area -/
theorem water_breadth_in_cistern (length width wet_surface_area : ℝ) :
  length = 9 →
  width = 6 →
  wet_surface_area = 121.5 →
  ∃ (breadth : ℝ),
    breadth = 2.25 ∧
    wet_surface_area = length * width + 2 * length * breadth + 2 * width * breadth :=
by sorry

end water_breadth_in_cistern_l68_6892


namespace divisible_by_3_4_5_count_l68_6870

theorem divisible_by_3_4_5_count : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 50 ∧ (3 ∣ n ∨ 4 ∣ n ∨ 5 ∣ n)) ∧ 
  (∀ n, 1 ≤ n ∧ n ≤ 50 ∧ (3 ∣ n ∨ 4 ∣ n ∨ 5 ∣ n) → n ∈ S) ∧ 
  Finset.card S = 29 :=
sorry

end divisible_by_3_4_5_count_l68_6870


namespace root_existence_l68_6825

theorem root_existence : ∃ x : ℝ, x ∈ (Set.Ioo (-1) (-1/2)) ∧ 2^x + x = 0 := by
  sorry

end root_existence_l68_6825


namespace alice_forest_walk_l68_6835

def morning_walk : ℕ := 10
def days_per_week : ℕ := 5
def total_distance : ℕ := 110

theorem alice_forest_walk :
  let morning_total := morning_walk * days_per_week
  let forest_total := total_distance - morning_total
  forest_total / days_per_week = 12 := by
  sorry

end alice_forest_walk_l68_6835


namespace min_value_theorem_l68_6889

theorem min_value_theorem (a b : ℝ) (h1 : a > b) (h2 : a * b = 1) :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt 2 ∧
  ∀ x, x = (a^2 + b^2) / (a - b) → x ≥ min_val :=
sorry

end min_value_theorem_l68_6889


namespace earnings_exceed_goal_l68_6873

/-- Represents the berry-picking job scenario --/
structure BerryPicking where
  lingonberry_rate : ℝ
  cloudberry_rate : ℝ
  blueberry_rate : ℝ
  monday_lingonberry : ℝ
  monday_cloudberry : ℝ
  monday_blueberry : ℝ
  tuesday_lingonberry_factor : ℝ
  tuesday_cloudberry_factor : ℝ
  tuesday_blueberry : ℝ
  goal : ℝ

/-- Calculates the total earnings for Monday and Tuesday --/
def total_earnings (job : BerryPicking) : ℝ :=
  let monday_earnings := 
    job.lingonberry_rate * job.monday_lingonberry +
    job.cloudberry_rate * job.monday_cloudberry +
    job.blueberry_rate * job.monday_blueberry
  let tuesday_earnings := 
    job.lingonberry_rate * (job.tuesday_lingonberry_factor * job.monday_lingonberry) +
    job.cloudberry_rate * (job.tuesday_cloudberry_factor * job.monday_cloudberry) +
    job.blueberry_rate * job.tuesday_blueberry
  monday_earnings + tuesday_earnings

/-- Theorem: Steve's earnings exceed his goal after two days --/
theorem earnings_exceed_goal (job : BerryPicking) 
  (h1 : job.lingonberry_rate = 2)
  (h2 : job.cloudberry_rate = 3)
  (h3 : job.blueberry_rate = 5)
  (h4 : job.monday_lingonberry = 8)
  (h5 : job.monday_cloudberry = 10)
  (h6 : job.monday_blueberry = 0)
  (h7 : job.tuesday_lingonberry_factor = 3)
  (h8 : job.tuesday_cloudberry_factor = 2)
  (h9 : job.tuesday_blueberry = 5)
  (h10 : job.goal = 150) :
  total_earnings job > job.goal := by
  sorry

end earnings_exceed_goal_l68_6873


namespace leading_coefficient_of_p_l68_6897

/-- The polynomial in question -/
def p (x : ℝ) : ℝ := 5*(x^5 - 3*x^4 + 2*x^3) - 6*(x^5 + x^3 + 1) + 2*(3*x^5 - x^4 + x^2)

/-- The leading coefficient of a polynomial -/
def leadingCoefficient (p : ℝ → ℝ) : ℝ :=
  sorry -- Definition of leading coefficient

theorem leading_coefficient_of_p :
  leadingCoefficient p = 5 := by
  sorry

end leading_coefficient_of_p_l68_6897


namespace line_parallel_perpendicular_implies_planes_perpendicular_l68_6849

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- Theorem statement
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (c : Line) (α β : Plane) :
  parallel c α → perpendicular c β → perpendicularPlanes α β := by
  sorry

end line_parallel_perpendicular_implies_planes_perpendicular_l68_6849


namespace coordinates_wrt_origin_l68_6859

/-- In a Cartesian coordinate system, the coordinates of a point with respect to the origin are equal to the point's coordinates. -/
theorem coordinates_wrt_origin (x y : ℝ) : (x, y) = (x, y) := by sorry

end coordinates_wrt_origin_l68_6859


namespace heartsuit_not_commutative_l68_6834

-- Define the ♥ operation
def heartsuit (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem statement
theorem heartsuit_not_commutative : ¬ ∀ (x y : ℝ), heartsuit x y = heartsuit y x := by
  sorry

end heartsuit_not_commutative_l68_6834


namespace total_books_collected_l68_6872

def north_america : ℕ := 581
def south_america : ℕ := 435
def africa : ℕ := 524
def europe : ℕ := 688
def australia : ℕ := 319
def asia : ℕ := 526
def antarctica : ℕ := 276

theorem total_books_collected :
  north_america + south_america + africa + europe + australia + asia + antarctica = 3349 := by
  sorry

end total_books_collected_l68_6872


namespace maria_trip_distance_l68_6848

/-- Given a total trip distance of 400 miles, with stops at 1/2 of the total distance
    and 1/4 of the remaining distance after the first stop, the distance traveled
    after the second stop is 150 miles. -/
theorem maria_trip_distance : 
  let total_distance : ℝ := 400
  let first_stop_fraction : ℝ := 1/2
  let second_stop_fraction : ℝ := 1/4
  let distance_to_first_stop := total_distance * first_stop_fraction
  let remaining_after_first_stop := total_distance - distance_to_first_stop
  let distance_to_second_stop := remaining_after_first_stop * second_stop_fraction
  let distance_after_second_stop := remaining_after_first_stop - distance_to_second_stop
  distance_after_second_stop = 150 := by
sorry

end maria_trip_distance_l68_6848


namespace competition_results_l68_6802

/-- Represents the final scores of competitors -/
structure Scores where
  A : ℚ
  B : ℚ
  C : ℚ

/-- Represents the points awarded for each position -/
structure PointSystem where
  first : ℚ
  second : ℚ
  third : ℚ

/-- Represents the number of times each competitor finished in each position -/
structure CompetitorResults where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The main theorem statement -/
theorem competition_results 
  (scores : Scores)
  (points : PointSystem)
  (A_results B_results C_results : CompetitorResults)
  (h_scores : scores.A = 22 ∧ scores.B = 9 ∧ scores.C = 9)
  (h_B_won_100m : B_results.first ≥ 1)
  (h_no_ties : ∀ event : ℕ, 
    A_results.first + B_results.first + C_results.first = event ∧
    A_results.second + B_results.second + C_results.second = event ∧
    A_results.third + B_results.third + C_results.third = event)
  (h_score_calculation : 
    scores.A = A_results.first * points.first + A_results.second * points.second + A_results.third * points.third ∧
    scores.B = B_results.first * points.first + B_results.second * points.second + B_results.third * points.third ∧
    scores.C = C_results.first * points.first + C_results.second * points.second + C_results.third * points.third)
  : (A_results.first + A_results.second + A_results.third = 4) ∧ 
    (B_results.first + B_results.second + B_results.third = 4) ∧ 
    (C_results.first + C_results.second + C_results.third = 4) ∧
    A_results.second ≥ 1 := by
  sorry

end competition_results_l68_6802


namespace smallest_number_with_given_remainders_l68_6840

theorem smallest_number_with_given_remainders :
  ∃ (x : ℕ), x > 0 ∧
    x % 11 = 9 ∧
    x % 13 = 11 ∧
    x % 15 = 13 ∧
    (∀ y : ℕ, y > 0 ∧ y % 11 = 9 ∧ y % 13 = 11 ∧ y % 15 = 13 → x ≤ y) ∧
    x = 2143 :=
by sorry

end smallest_number_with_given_remainders_l68_6840


namespace center_is_nine_l68_6839

def Grid := Fin 3 → Fin 3 → Nat

def is_valid_arrangement (g : Grid) : Prop :=
  (∀ n : Nat, n ∈ Finset.range 9 → ∃ i j, g i j = n + 1) ∧
  (∀ i j, g i j ∈ Finset.range 9 → g i j ≤ 9) ∧
  (∀ n : Nat, n ∈ Finset.range 8 → 
    ∃ i j k l, g i j = n + 1 ∧ g k l = n + 2 ∧ 
    ((i = k ∧ (j = l + 1 ∨ j + 1 = l)) ∨ 
     (j = l ∧ (i = k + 1 ∨ i + 1 = k))))

def top_edge_sum (g : Grid) : Nat :=
  g 0 0 + g 0 1 + g 0 2

theorem center_is_nine (g : Grid) 
  (h1 : is_valid_arrangement g) 
  (h2 : top_edge_sum g = 15) : 
  g 1 1 = 9 := by
  sorry

end center_is_nine_l68_6839


namespace parabola_equation_l68_6890

/-- Represents a parabola with vertex at the origin and coordinate axes as axes of symmetry -/
structure Parabola where
  a : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = -2*a*x

/-- The parabola passes through the given point -/
def passes_through (p : Parabola) (x y : ℝ) : Prop :=
  p.equation x y

theorem parabola_equation : 
  ∃ (p : Parabola), passes_through p (-2) (-4) ∧ p.a = 4 := by
  sorry

end parabola_equation_l68_6890


namespace container_problem_l68_6826

theorem container_problem :
  ∃! (x y : ℕ), 130 * x + 160 * y = 3000 ∧ x = 12 ∧ y = 9 :=
by sorry

end container_problem_l68_6826


namespace interval_of_increase_l68_6841

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 5*x + 6) / Real.log (1/4)

def domain (x : ℝ) : Prop := x < 2 ∨ x > 3

theorem interval_of_increase :
  ∀ x y, domain x → domain y → x < y → x < 2 → f x > f y := by sorry

end interval_of_increase_l68_6841


namespace maddie_weekend_watch_l68_6815

def total_episodes : ℕ := 8
def episode_length : ℕ := 44
def monday_watch : ℕ := 138
def thursday_watch : ℕ := 21
def friday_episodes : ℕ := 2

def weekend_watch : ℕ := 105

theorem maddie_weekend_watch :
  let total_watch := total_episodes * episode_length
  let weekday_watch := monday_watch + thursday_watch + (friday_episodes * episode_length)
  total_watch - weekday_watch = weekend_watch := by sorry

end maddie_weekend_watch_l68_6815


namespace ellipse_intersection_theorem_l68_6832

noncomputable section

-- Define the ellipse Γ
def Γ (x y : ℝ) : Prop := x^2 / 12 + y^2 / 4 = 1

-- Define the line l
def l (x y m : ℝ) : Prop := y = x + m

-- Define the distance between two points
def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem ellipse_intersection_theorem :
  ∃ (xA yA xB yB m x₀ : ℝ),
    Γ xA yA ∧ Γ xB yB ∧  -- A and B are on the ellipse
    l xA yA m ∧ l xB yB m ∧  -- A and B are on the line l
    distance xA yA xB yB = 3 * Real.sqrt 2 ∧  -- |AB| = 3√2
    distance x₀ 2 xA yA = distance x₀ 2 xB yB ∧  -- |PA| = |PB|
    (x₀ = -3 ∨ x₀ = -1) :=
by sorry

end

end ellipse_intersection_theorem_l68_6832


namespace intersection_point_is_unique_l68_6806

-- Define the two line equations
def line1 (x y : ℝ) : Prop := 3 * y = -2 * x + 6
def line2 (x y : ℝ) : Prop := 4 * y = 7 * x - 8

-- Define the intersection point
def intersection_point : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem intersection_point_is_unique :
  (line1 intersection_point.1 intersection_point.2) ∧
  (line2 intersection_point.1 intersection_point.2) ∧
  (∀ x y : ℝ, line1 x y ∧ line2 x y → (x, y) = intersection_point) :=
by sorry

end intersection_point_is_unique_l68_6806


namespace probability_even_sum_l68_6833

def wheel1 : List ℕ := [1, 1, 2, 3, 3, 4]
def wheel2 : List ℕ := [2, 4, 5, 5, 6]

def is_even (n : ℕ) : Bool := n % 2 = 0

def count_even (l : List ℕ) : ℕ := (l.filter is_even).length

def total_outcomes : ℕ := wheel1.length * wheel2.length

def favorable_outcomes : ℕ := 
  (wheel1.filter is_even).length * (wheel2.filter is_even).length +
  (wheel1.filter (fun x => ¬(is_even x))).length * (wheel2.filter (fun x => ¬(is_even x))).length

theorem probability_even_sum : 
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 15 := by sorry

end probability_even_sum_l68_6833


namespace rectangle_area_change_l68_6871

theorem rectangle_area_change (L B x : ℝ) (h1 : L > 0) (h2 : B > 0) (h3 : x > 0) : 
  (L + x / 100 * L) * (B - x / 100 * B) = 99 / 100 * (L * B) → x = 10 := by
  sorry

end rectangle_area_change_l68_6871


namespace vitamin_shop_lcm_l68_6819

theorem vitamin_shop_lcm : ∃ n : ℕ, n > 0 ∧ n % 7 = 0 ∧ n % 17 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 7 = 0 ∧ m % 17 = 0) → n ≤ m :=
by sorry

end vitamin_shop_lcm_l68_6819


namespace luke_coin_count_l68_6836

/-- 
Given:
- Luke has 5 piles of quarters and 5 piles of dimes
- Each pile contains 3 coins
Prove that the total number of coins is 30
-/
theorem luke_coin_count (piles_quarters piles_dimes coins_per_pile : ℕ) 
  (h1 : piles_quarters = 5)
  (h2 : piles_dimes = 5)
  (h3 : coins_per_pile = 3) : 
  piles_quarters * coins_per_pile + piles_dimes * coins_per_pile = 30 := by
  sorry

#eval 5 * 3 + 5 * 3  -- Should output 30

end luke_coin_count_l68_6836


namespace like_terms_power_l68_6852

theorem like_terms_power (a b : ℕ) : 
  (∀ x y : ℝ, ∃ c : ℝ, x^(a+1) * y^2 = c * x^3 * y^b) → 
  a^b = 4 := by
sorry

end like_terms_power_l68_6852


namespace green_probability_is_half_l68_6857

/-- A cube with colored faces -/
structure ColoredCube where
  total_faces : ℕ
  green_faces : ℕ
  yellow_faces : ℕ
  red_faces : ℕ

/-- The probability of rolling a green face on a colored cube -/
def green_probability (cube : ColoredCube) : ℚ :=
  cube.green_faces / cube.total_faces

/-- Theorem: The probability of rolling a green face on a specific colored cube is 1/2 -/
theorem green_probability_is_half :
  let cube : ColoredCube := {
    total_faces := 6,
    green_faces := 3,
    yellow_faces := 2,
    red_faces := 1
  }
  green_probability cube = 1/2 := by
  sorry

end green_probability_is_half_l68_6857


namespace square_sum_plus_double_sum_squares_l68_6816

theorem square_sum_plus_double_sum_squares : (5 + 7)^2 + (5^2 + 7^2) * 2 = 292 := by
  sorry

end square_sum_plus_double_sum_squares_l68_6816


namespace min_value_f_l68_6876

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_period (x : ℝ) : f (x + 2) = 3 * f x

axiom f_def (x : ℝ) (h : x ∈ Set.Icc 0 2) : f x = x^2 - 2*x

-- Define the theorem
theorem min_value_f :
  ∃ (x : ℝ), x ∈ Set.Icc (-4) (-2) ∧
  f x = -1/9 ∧
  ∀ (y : ℝ), y ∈ Set.Icc (-4) (-2) → f y ≥ -1/9 :=
sorry

end min_value_f_l68_6876


namespace model_a_better_fit_l68_6813

def model_a (x : ℝ) : ℝ := x^2 + 1
def model_b (x : ℝ) : ℝ := 3*x - 1

def data_points : List (ℝ × ℝ) := [(1, 2), (2, 5), (3, 10.2)]

def error (model : ℝ → ℝ) (point : ℝ × ℝ) : ℝ :=
  (model point.1 - point.2)^2

def total_error (model : ℝ → ℝ) (points : List (ℝ × ℝ)) : ℝ :=
  points.foldl (λ acc p => acc + error model p) 0

theorem model_a_better_fit :
  total_error model_a data_points < total_error model_b data_points := by
  sorry

end model_a_better_fit_l68_6813


namespace five_dozen_apples_cost_l68_6867

/-- The cost of a given number of dozens of apples, given the cost of two dozens. -/
def apple_cost (two_dozen_cost : ℚ) (dozens : ℚ) : ℚ :=
  (dozens / 2) * two_dozen_cost

/-- Theorem: If two dozen apples cost $15.60, then five dozen apples cost $39.00 -/
theorem five_dozen_apples_cost (two_dozen_cost : ℚ) 
  (h : two_dozen_cost = 15.6) : 
  apple_cost two_dozen_cost 5 = 39 := by
  sorry

end five_dozen_apples_cost_l68_6867


namespace marcia_project_time_l68_6880

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The number of hours Marcia spent on her science project -/
def hours_spent : ℕ := 5

/-- The total number of minutes Marcia spent on her science project -/
def total_minutes : ℕ := hours_spent * minutes_per_hour

theorem marcia_project_time : total_minutes = 300 := by
  sorry

end marcia_project_time_l68_6880


namespace twin_primes_sum_divisible_by_12_l68_6875

theorem twin_primes_sum_divisible_by_12 (p : ℕ) (h1 : p > 3) (h2 : Prime p) (h3 : Prime (p + 2)) :
  12 ∣ (p + (p + 2)) :=
sorry

end twin_primes_sum_divisible_by_12_l68_6875


namespace problem_solution_l68_6858

theorem problem_solution (x y : ℝ) (h : |x - Real.sqrt 3 + 1| + Real.sqrt (y - 2) = 0) :
  (x = Real.sqrt 3 - 1 ∧ y = 2) ∧ x^2 + 2*x - 3*y = -4 := by
  sorry

end problem_solution_l68_6858


namespace smallest_valid_number_l68_6896

def is_valid (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ 
  4 * (n % 10 * 10 + n / 10) = 2 * n

theorem smallest_valid_number : 
  (∃ (n : ℕ), is_valid n) ∧ 
  (∀ (m : ℕ), is_valid m → m ≥ 52) ∧
  is_valid 52 := by
  sorry

end smallest_valid_number_l68_6896


namespace vector_sum_magnitude_l68_6878

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)
  (h2 : a = (Real.sqrt 3, 1))
  (h3 : Real.sqrt ((Prod.fst b)^2 + (Prod.snd b)^2) = 1) :
  Real.sqrt ((Prod.fst (a + 2 • b))^2 + (Prod.snd (a + 2 • b))^2) = 2 * Real.sqrt 3 :=
sorry

end vector_sum_magnitude_l68_6878


namespace two_number_problem_l68_6888

-- Define the types for logicians and their knowledge states
inductive Logician | A | B
inductive Knowledge | Known | Unknown

-- Define a function to represent the state of knowledge after each exchange
def knowledge_state (exchange : ℕ) (l : Logician) : Knowledge := sorry

-- Define the conditions for the sum and sum of squares
def sum_condition (u v : ℕ) : Prop := u + v = 17

def sum_of_squares_condition (u v : ℕ) : Prop := u^2 + v^2 = 145

-- Define the main theorem
theorem two_number_problem (u v : ℕ) :
  u > 0 ∧ v > 0 ∧
  sum_condition u v ∧
  sum_of_squares_condition u v ∧
  (∀ e, e < 6 → knowledge_state e Logician.A = Knowledge.Unknown) ∧
  (∀ e, e < 6 → knowledge_state e Logician.B = Knowledge.Unknown) ∧
  knowledge_state 6 Logician.B = Knowledge.Known
  → (u = 8 ∧ v = 9) ∨ (u = 9 ∧ v = 8) := by sorry

end two_number_problem_l68_6888


namespace carnation_percentage_is_67_point_5_l68_6821

/-- Represents a flower display with pink and red flowers, either roses or carnations -/
structure FlowerDisplay where
  total : ℝ
  pink_ratio : ℝ
  red_carnation_ratio : ℝ
  pink_rose_ratio : ℝ

/-- Calculates the percentage of carnations in the flower display -/
def carnation_percentage (display : FlowerDisplay) : ℝ :=
  let red_ratio := 1 - display.pink_ratio
  let pink_carnation_ratio := display.pink_ratio * (1 - display.pink_rose_ratio)
  let red_carnation_ratio := red_ratio * display.red_carnation_ratio
  (pink_carnation_ratio + red_carnation_ratio) * 100

/-- Theorem stating that under given conditions, 67.5% of flowers are carnations -/
theorem carnation_percentage_is_67_point_5
  (display : FlowerDisplay)
  (h_pink_ratio : display.pink_ratio = 7/10)
  (h_red_carnation_ratio : display.red_carnation_ratio = 1/2)
  (h_pink_rose_ratio : display.pink_rose_ratio = 1/4) :
  carnation_percentage display = 67.5 := by
  sorry

end carnation_percentage_is_67_point_5_l68_6821


namespace copper_needed_in_mixture_l68_6898

/-- Given a manufacturing mixture with specified percentages of materials,
    this theorem calculates the amount of copper needed when a certain amount of lead is used. -/
theorem copper_needed_in_mixture (total : ℝ) (cobalt_percent lead_percent copper_percent : ℝ) 
    (lead_amount : ℝ) (copper_amount : ℝ) : 
  cobalt_percent = 0.15 →
  lead_percent = 0.25 →
  copper_percent = 0.60 →
  cobalt_percent + lead_percent + copper_percent = 1 →
  lead_amount = 5 →
  total * lead_percent = lead_amount →
  copper_amount = total * copper_percent →
  copper_amount = 12 := by
sorry

end copper_needed_in_mixture_l68_6898


namespace problem_solution_l68_6877

/-- Given m ≥ 0 and f(x) = 2|x - 1| - |2x + m| with a maximum value of 3,
    prove that m = 1 and min(a² + b² + c²) = 1/6 where a - 2b + c = m -/
theorem problem_solution (m : ℝ) (h_m : m ≥ 0)
  (f : ℝ → ℝ) (h_f : ∀ x, f x = 2 * |x - 1| - |2*x + m|)
  (h_max : ∀ x, f x ≤ 3) (h_exists : ∃ x, f x = 3) :
  m = 1 ∧ (∃ a b c : ℝ, a - 2*b + c = m ∧
    a^2 + b^2 + c^2 = 1/6 ∧
    ∀ a' b' c' : ℝ, a' - 2*b' + c' = m → a'^2 + b'^2 + c'^2 ≥ 1/6) :=
by sorry

end problem_solution_l68_6877


namespace sequence_length_five_l68_6831

theorem sequence_length_five :
  ∃ (b₁ b₂ b₃ b₄ b₅ : ℕ), 
    b₁ < b₂ ∧ b₂ < b₃ ∧ b₃ < b₄ ∧ b₄ < b₅ ∧
    (2^433 + 1) / (2^49 + 1) = 2^b₁ + 2^b₂ + 2^b₃ + 2^b₄ + 2^b₅ := by
  sorry

end sequence_length_five_l68_6831


namespace assistant_productivity_increase_l68_6850

theorem assistant_productivity_increase 
  (original_bears : ℝ) 
  (original_hours : ℝ) 
  (bear_increase_rate : ℝ) 
  (hour_decrease_rate : ℝ) 
  (h₁ : bear_increase_rate = 0.8) 
  (h₂ : hour_decrease_rate = 0.1) 
  : (((1 + bear_increase_rate) * original_bears) / ((1 - hour_decrease_rate) * original_hours)) / 
    (original_bears / original_hours) - 1 = 1 := by
  sorry

end assistant_productivity_increase_l68_6850


namespace stream_speed_l68_6804

/-- Given a boat traveling downstream and upstream, prove the speed of the stream. -/
theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ) 
  (h1 : downstream_distance = 60) 
  (h2 : upstream_distance = 30) 
  (h3 : time = 3) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * time ∧
    upstream_distance = (boat_speed - stream_speed) * time ∧
    stream_speed = 5 := by
  sorry

#check stream_speed

end stream_speed_l68_6804


namespace system_solution_ratio_l68_6822

theorem system_solution_ratio (x y a b : ℝ) (h1 : 4*x - 2*y = a) (h2 : 6*y - 12*x = b) (h3 : b ≠ 0) :
  a / b = -1/3 := by
  sorry

end system_solution_ratio_l68_6822


namespace circular_seating_arrangement_l68_6851

/-- Given a circular arrangement of n people, this function calculates the clockwise distance between two positions -/
def clockwise_distance (n : ℕ) (a b : ℕ) : ℕ :=
  (b - a + n) % n

/-- Given a circular arrangement of n people, this function calculates the counterclockwise distance between two positions -/
def counterclockwise_distance (n : ℕ) (a b : ℕ) : ℕ :=
  (a - b + n) % n

theorem circular_seating_arrangement (n : ℕ) (h1 : n > 31) :
  clockwise_distance n 31 7 = counterclockwise_distance n 31 14 → n = 41 := by
  sorry

#eval clockwise_distance 41 31 7
#eval counterclockwise_distance 41 31 14

end circular_seating_arrangement_l68_6851


namespace train_length_l68_6809

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 36 → ∃ (length_m : ℝ), 
  (abs (length_m - 600.12) < 0.01) ∧ (length_m = speed_kmh * (5/18) * time_s) := by
  sorry

#check train_length

end train_length_l68_6809


namespace jill_cookie_sales_l68_6807

def cookie_sales (goal : ℕ) (first second third fourth fifth : ℕ) : Prop :=
  let total_sold := first + second + third + fourth + fifth
  goal - total_sold = 75

theorem jill_cookie_sales :
  cookie_sales 150 5 20 10 30 10 :=
by
  sorry

end jill_cookie_sales_l68_6807


namespace soccer_team_math_enrollment_l68_6803

theorem soccer_team_math_enrollment (total_players : ℕ) (physics_players : ℕ) (both_subjects : ℕ) :
  total_players = 25 →
  physics_players = 15 →
  both_subjects = 6 →
  ∃ (math_players : ℕ), math_players = 16 ∧ 
    total_players = physics_players + math_players - both_subjects :=
by
  sorry

end soccer_team_math_enrollment_l68_6803


namespace revenue_change_l68_6879

/-- Given a price increase of 80% and a sales decrease of 35%, prove that revenue increases by 17% -/
theorem revenue_change (P Q : ℝ) (h_P : P > 0) (h_Q : Q > 0) : 
  let R := P * Q
  let P_new := P * (1 + 0.80)
  let Q_new := Q * (1 - 0.35)
  let R_new := P_new * Q_new
  (R_new - R) / R = 0.17 := by
sorry

end revenue_change_l68_6879


namespace same_combination_probability_l68_6853

/-- Represents the number of candies of each color in the jar -/
structure JarContents where
  red : Nat
  blue : Nat
  green : Nat

/-- Calculates the probability of two people picking the same color combination -/
def probability_same_combination (jar : JarContents) : ℚ :=
  sorry

/-- The main theorem stating the probability for the given jar contents -/
theorem same_combination_probability :
  let jar : JarContents := { red := 12, blue := 12, green := 6 }
  probability_same_combination jar = 2783 / 847525 := by
  sorry

end same_combination_probability_l68_6853


namespace hexagon_side_length_l68_6843

/-- Given a triangle ABC, prove that a hexagon with sides parallel to the triangle's sides
    and equal length d satisfies the equation: d = (abc) / (ab + bc + ca) -/
theorem hexagon_side_length (a b c d : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  d = (a * b * c) / (a * b + b * c + c * a) ↔
  (1 / d = 1 / a + 1 / b + 1 / c) :=
sorry

end hexagon_side_length_l68_6843


namespace three_digit_numbers_decreasing_by_factor_of_six_l68_6886

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ (a b c : ℕ), 
    n = 100 * a + 10 * b + c ∧
    a ≠ 0 ∧
    10 * b + c = (100 * a + 10 * b + c) / 6

theorem three_digit_numbers_decreasing_by_factor_of_six : 
  {n : ℕ | is_valid_number n} = {120, 240, 360, 480} := by sorry

end three_digit_numbers_decreasing_by_factor_of_six_l68_6886


namespace number_operations_l68_6891

theorem number_operations (x : ℚ) : x = 192 → 6 * (((x/8) + 8) - 30) = 12 := by
  sorry

end number_operations_l68_6891


namespace rosa_flower_count_l68_6863

/-- The number of flowers Rosa has after receiving flowers from Andre -/
def total_flowers (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Rosa's total flowers is the sum of her initial flowers and received flowers -/
theorem rosa_flower_count (initial : ℕ) (received : ℕ) :
  total_flowers initial received = initial + received :=
by sorry

end rosa_flower_count_l68_6863


namespace purple_sequins_count_l68_6829

/-- The number of purple sequins in each row on Jane's costume. -/
def purple_sequins_per_row : ℕ :=
  let total_sequins : ℕ := 162
  let blue_rows : ℕ := 6
  let blue_per_row : ℕ := 8
  let purple_rows : ℕ := 5
  let green_rows : ℕ := 9
  let green_per_row : ℕ := 6
  let blue_sequins : ℕ := blue_rows * blue_per_row
  let green_sequins : ℕ := green_rows * green_per_row
  let purple_sequins : ℕ := total_sequins - (blue_sequins + green_sequins)
  purple_sequins / purple_rows

theorem purple_sequins_count : purple_sequins_per_row = 12 := by
  sorry

end purple_sequins_count_l68_6829


namespace complex_magnitude_problem_l68_6893

theorem complex_magnitude_problem (a b : ℝ) (h : a^2 - 4 + b * Complex.I - Complex.I = 0) :
  Complex.abs (a + b * Complex.I) = Real.sqrt 5 := by
  sorry

end complex_magnitude_problem_l68_6893


namespace max_consecutive_digit_sums_l68_6854

/-- Given a natural number, returns the sum of its digits. -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Returns true if the given list of natural numbers contains n consecutive numbers. -/
def isConsecutive (l : List ℕ) (n : ℕ) : Prop := sorry

/-- Theorem: 18 is the maximum value of n for which there exists a sequence of n consecutive 
    natural numbers whose digit sums form another sequence of n consecutive numbers. -/
theorem max_consecutive_digit_sums : 
  ∀ n : ℕ, n > 18 → 
  ¬∃ (start : ℕ), 
    let numbers := List.range n |>.map (λ i => start + i)
    let digitSums := numbers.map sumOfDigits
    isConsecutive numbers n ∧ isConsecutive digitSums n :=
by sorry

end max_consecutive_digit_sums_l68_6854


namespace fraction_to_decimal_l68_6860

theorem fraction_to_decimal : (3 : ℚ) / 50 = 0.06 := by
  sorry

end fraction_to_decimal_l68_6860


namespace divisible_by_eight_l68_6818

theorem divisible_by_eight (n : ℕ) : ∃ k : ℤ, 5^n + 2 * 3^(n-1) + 1 = 8*k := by
  sorry

end divisible_by_eight_l68_6818


namespace train_bridge_crossing_time_l68_6887

/-- The time taken for a train to completely cross a bridge -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (train_speed : ℝ) 
  (h1 : train_length = 400) 
  (h2 : bridge_length = 300) 
  (h3 : train_speed = 55.99999999999999) : 
  (train_length + bridge_length) / train_speed = (400 + 300) / 55.99999999999999 := by
  sorry

end train_bridge_crossing_time_l68_6887


namespace rectangle_area_l68_6861

theorem rectangle_area : 
  ∀ (square_side : ℝ) (circle_radius : ℝ) (rectangle_length : ℝ) (rectangle_breadth : ℝ),
    square_side^2 = 625 →
    circle_radius = square_side →
    rectangle_length = (2/5) * circle_radius →
    rectangle_breadth = 10 →
    rectangle_length * rectangle_breadth = 100 := by
  sorry

end rectangle_area_l68_6861


namespace election_votes_l68_6885

theorem election_votes (total_votes : ℕ) : 
  (∃ (winner_votes loser_votes : ℕ),
    winner_votes + loser_votes = total_votes ∧
    winner_votes = (70 * total_votes) / 100 ∧
    loser_votes = (30 * total_votes) / 100 ∧
    winner_votes - loser_votes = 180) →
  total_votes = 450 :=
by sorry

end election_votes_l68_6885


namespace no_equal_arithmetic_operations_l68_6830

theorem no_equal_arithmetic_operations (v t : ℝ) (hv : v > 0) (ht : t > 0) : 
  ¬(v + t = v * t ∧ v + t = v / t) :=
by sorry

end no_equal_arithmetic_operations_l68_6830


namespace betty_order_cost_l68_6864

/-- The total cost of Betty's order -/
def total_cost (slippers_quantity : ℕ) (slippers_price : ℚ) 
               (lipstick_quantity : ℕ) (lipstick_price : ℚ) 
               (hair_color_quantity : ℕ) (hair_color_price : ℚ) : ℚ :=
  slippers_quantity * slippers_price + 
  lipstick_quantity * lipstick_price + 
  hair_color_quantity * hair_color_price

/-- Theorem stating that Betty's total order cost is $44 -/
theorem betty_order_cost : 
  total_cost 6 (5/2) 4 (5/4) 8 3 = 44 := by
  sorry

end betty_order_cost_l68_6864
