import Mathlib

namespace x_power_twenty_equals_one_l3005_300510

theorem x_power_twenty_equals_one (x : ℝ) (h : x + 1/x = 2) : x^20 = 1 := by
  sorry

end x_power_twenty_equals_one_l3005_300510


namespace inequality_solution_l3005_300586

theorem inequality_solution (x : ℝ) : 
  (x + 1) / 2 > 1 - (2 * x - 1) / 3 ↔ x > 5 / 7 := by
  sorry

end inequality_solution_l3005_300586


namespace video_game_earnings_l3005_300592

def total_games : ℕ := 10
def non_working_games : ℕ := 2
def price_per_game : ℕ := 4

theorem video_game_earnings :
  (total_games - non_working_games) * price_per_game = 32 := by
  sorry

end video_game_earnings_l3005_300592


namespace quadratic_factorization_l3005_300591

theorem quadratic_factorization : 
  ∃ (c d : ℤ), (∀ y : ℝ, 4 * y^2 + 4 * y - 32 = (4 * y + c) * (y + d)) ∧ c - d = 12 := by
  sorry

end quadratic_factorization_l3005_300591


namespace discounted_price_is_correct_l3005_300500

/-- Calculate the final price after applying two successive discounts -/
def final_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  initial_price * (1 - discount1) * (1 - discount2)

/-- Theorem stating that the final price after discounts is approximately 59.85 -/
theorem discounted_price_is_correct :
  let initial_price : ℝ := 70
  let discount1 : ℝ := 0.1  -- 10%
  let discount2 : ℝ := 0.04999999999999997  -- 4.999999999999997%
  abs (final_price initial_price discount1 discount2 - 59.85) < 0.01 := by
  sorry

end discounted_price_is_correct_l3005_300500


namespace libor_number_theorem_l3005_300509

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def all_digits_odd (n : ℕ) : Prop :=
  ∀ d, d ∈ (n.digits 10) → d % 2 = 1

def no_odd_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ (n.digits 10) → d % 2 = 0

theorem libor_number_theorem :
  ∀ n : ℕ, is_three_digit n ∧ all_digits_odd n ∧ is_three_digit (n + 421) ∧ no_odd_digits (n + 421) →
    n = 179 ∨ n = 199 ∨ n = 379 ∨ n = 399 :=
sorry

end libor_number_theorem_l3005_300509


namespace division_simplification_l3005_300518

theorem division_simplification (x : ℝ) (hx : x ≠ 0) :
  (1 + 1/x) / ((x^2 + x)/x) = 1/x := by sorry

end division_simplification_l3005_300518


namespace cos_300_degrees_l3005_300564

theorem cos_300_degrees (θ : Real) : 
  θ = 300 * Real.pi / 180 → Real.cos θ = 1/2 := by
  sorry

end cos_300_degrees_l3005_300564


namespace perfect_square_trinomial_l3005_300595

/-- A trinomial x^2 + 2ax + 9 is a perfect square if and only if a = ±3 -/
theorem perfect_square_trinomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 + 2*a*x + 9 = (x + b)^2) ↔ (a = 3 ∨ a = -3) := by
  sorry

end perfect_square_trinomial_l3005_300595


namespace bills_max_papers_l3005_300568

/-- Represents the number of items Bill can buy -/
structure BillsPurchase where
  pens : ℕ
  pencils : ℕ
  papers : ℕ

/-- The cost of Bill's purchase -/
def cost (b : BillsPurchase) : ℕ := 3 * b.pens + 5 * b.pencils + 9 * b.papers

/-- A purchase is valid if it meets the given conditions -/
def isValid (b : BillsPurchase) : Prop :=
  b.pens ≥ 2 ∧ b.pencils ≥ 1 ∧ cost b = 72

/-- The maximum number of papers Bill can buy -/
def maxPapers : ℕ := 6

theorem bills_max_papers :
  ∀ b : BillsPurchase, isValid b → b.papers ≤ maxPapers ∧
  ∃ b' : BillsPurchase, isValid b' ∧ b'.papers = maxPapers :=
sorry

end bills_max_papers_l3005_300568


namespace cuboid_volume_example_l3005_300527

/-- The volume of a cuboid with given base area and height -/
def cuboid_volume (base_area : ℝ) (height : ℝ) : ℝ :=
  base_area * height

/-- Theorem: The volume of a cuboid with base area 14 cm² and height 13 cm is 182 cm³ -/
theorem cuboid_volume_example : cuboid_volume 14 13 = 182 := by
  sorry

end cuboid_volume_example_l3005_300527


namespace ferris_wheel_small_seats_l3005_300532

/-- Represents a Ferris wheel with small and large seats -/
structure FerrisWheel where
  small_seats : ℕ
  large_seats : ℕ
  small_seat_capacity : ℕ
  people_on_small_seats : ℕ

/-- The number of small seats on the Ferris wheel is 2 -/
theorem ferris_wheel_small_seats (fw : FerrisWheel) 
  (h1 : fw.large_seats = 23)
  (h2 : fw.small_seat_capacity = 14)
  (h3 : fw.people_on_small_seats = 28) :
  fw.small_seats = 2 := by
  sorry

#check ferris_wheel_small_seats

end ferris_wheel_small_seats_l3005_300532


namespace composition_equality_l3005_300596

theorem composition_equality (a : ℝ) (h1 : a > 1) : 
  let f (x : ℝ) := x^2 + 2
  let g (x : ℝ) := x^2 + 2
  f (g a) = 12 → a = Real.sqrt (Real.sqrt 10 - 2) := by
sorry

end composition_equality_l3005_300596


namespace evaluate_expression_l3005_300502

theorem evaluate_expression : -(16 / 4 * 11 - 50 + 2^3 * 5) = -34 := by
  sorry

end evaluate_expression_l3005_300502


namespace tan_135_degrees_l3005_300563

theorem tan_135_degrees : 
  let angle : Real := 135 * Real.pi / 180
  let point : Fin 2 → Real := ![-(Real.sqrt 2) / 2, (Real.sqrt 2) / 2]
  Real.tan angle = -1 := by
  sorry

end tan_135_degrees_l3005_300563


namespace expansion_terms_count_l3005_300570

/-- The number of dissimilar terms in the expansion of (a + b + c + d)^7 -/
def dissimilar_terms : ℕ := Nat.choose 10 3

/-- Theorem stating that the number of dissimilar terms in (a + b + c + d)^7 is equal to (10 choose 3) -/
theorem expansion_terms_count : dissimilar_terms = 120 := by sorry

end expansion_terms_count_l3005_300570


namespace arithmetic_progression_equiv_square_product_l3005_300539

theorem arithmetic_progression_equiv_square_product 
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (∃ d : ℝ, Real.log y - Real.log x = d ∧ Real.log z - Real.log y = d) ↔ 
  y^2 = x*z := by
sorry

end arithmetic_progression_equiv_square_product_l3005_300539


namespace g_3_equals_109_l3005_300519

def g (x : ℝ) : ℝ := 7 * x^3 - 8 * x^2 - 5 * x + 7

theorem g_3_equals_109 : g 3 = 109 := by
  sorry

end g_3_equals_109_l3005_300519


namespace point_translation_l3005_300599

/-- Given a point M(-2, 3) in the Cartesian coordinate system,
    prove that after translating it 3 units downwards and then 1 unit to the right,
    the resulting point has coordinates (-1, 0). -/
theorem point_translation (M : ℝ × ℝ) :
  M = (-2, 3) →
  let M' := (M.1, M.2 - 3)  -- Translate 3 units downwards
  let M'' := (M'.1 + 1, M'.2)  -- Translate 1 unit to the right
  M'' = (-1, 0) := by
sorry

end point_translation_l3005_300599


namespace volunteer_selection_l3005_300543

theorem volunteer_selection (n : ℕ) (h : n = 5) : 
  (n.choose 1) * ((n - 1).choose 1 * (n - 2).choose 1) = 60 := by
  sorry

end volunteer_selection_l3005_300543


namespace correlation_coefficient_equals_height_variation_total_variation_is_one_l3005_300579

/-- The correlation coefficient between height and weight -/
def correlation_coefficient : ℝ := 0.76

/-- The proportion of weight variation explained by height -/
def height_explained_variation : ℝ := 0.76

/-- The proportion of weight variation explained by random errors -/
def random_error_variation : ℝ := 0.24

/-- Theorem stating that the correlation coefficient is equal to the proportion of variation explained by height -/
theorem correlation_coefficient_equals_height_variation :
  correlation_coefficient = height_explained_variation :=
by sorry

/-- Theorem stating that the sum of variations explained by height and random errors is 1 -/
theorem total_variation_is_one :
  height_explained_variation + random_error_variation = 1 :=
by sorry

end correlation_coefficient_equals_height_variation_total_variation_is_one_l3005_300579


namespace average_tv_watching_l3005_300533

def tv_hours : List ℝ := [10, 8, 12]

theorem average_tv_watching :
  (tv_hours.sum / tv_hours.length : ℝ) = 10 := by
  sorry

end average_tv_watching_l3005_300533


namespace ellipse_to_hyperbola_l3005_300517

theorem ellipse_to_hyperbola (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : b = Real.sqrt 3 * c) 
  (h4 : a + c = 3 * Real.sqrt 3) 
  (h5 : a^2 = b^2 + c^2) :
  ∃ (x y : ℝ), y^2 / 12 - x^2 / 9 = 1 := by sorry

end ellipse_to_hyperbola_l3005_300517


namespace octal_perfect_square_last_digit_l3005_300538

/-- A perfect square in octal form (abc)₈ where a ≠ 0 always has c = 1 -/
theorem octal_perfect_square_last_digit (a b c : Nat) (h1 : a ≠ 0) 
  (h2 : ∃ (n : Nat), n^2 = a * 8^2 + b * 8 + c) : c = 1 := by
  sorry

end octal_perfect_square_last_digit_l3005_300538


namespace pencil_eraser_cost_l3005_300541

theorem pencil_eraser_cost :
  ∃ (p e : ℕ), 
    p > 0 ∧ 
    e > 0 ∧ 
    7 * p + 5 * e = 130 ∧ 
    p > e ∧ 
    p + e = 22 := by
  sorry

end pencil_eraser_cost_l3005_300541


namespace rachels_mystery_book_shelves_l3005_300559

theorem rachels_mystery_book_shelves 
  (books_per_shelf : ℕ) 
  (picture_book_shelves : ℕ) 
  (total_books : ℕ) 
  (h1 : books_per_shelf = 9)
  (h2 : picture_book_shelves = 2)
  (h3 : total_books = 72) :
  (total_books - picture_book_shelves * books_per_shelf) / books_per_shelf = 6 := by
  sorry

end rachels_mystery_book_shelves_l3005_300559


namespace prism_volume_l3005_300536

/-- The volume of a right rectangular prism with given face areas and sum of dimensions -/
theorem prism_volume (a b c : ℝ) 
  (h1 : a * b = 18) 
  (h2 : b * c = 20) 
  (h3 : c * a = 12) 
  (h4 : a + b + c = 11) : 
  a * b * c = 12 * Real.sqrt 15 := by
  sorry

end prism_volume_l3005_300536


namespace keith_added_scissors_l3005_300594

/-- The number of scissors Keith added to the drawer -/
def scissors_added (initial final : ℕ) : ℕ := final - initial

/-- Proof that Keith added 22 scissors to the drawer -/
theorem keith_added_scissors : scissors_added 54 76 = 22 := by
  sorry

end keith_added_scissors_l3005_300594


namespace max_self_intersections_l3005_300574

/-- A polygonal chain on a graph paper -/
structure PolygonalChain where
  segments : ℕ
  closed : Bool
  on_graph_paper : Bool
  no_segments_on_same_line : Bool

/-- The number of self-intersection points of a polygonal chain -/
def self_intersection_points (chain : PolygonalChain) : ℕ := sorry

/-- Theorem: The maximum number of self-intersection points for a closed 14-segment polygonal chain 
    on a graph paper, where no two segments lie on the same line, is 17 -/
theorem max_self_intersections (chain : PolygonalChain) :
  chain.segments = 14 ∧ 
  chain.closed ∧ 
  chain.on_graph_paper ∧ 
  chain.no_segments_on_same_line →
  self_intersection_points chain ≤ 17 ∧ 
  ∃ (chain' : PolygonalChain), 
    chain'.segments = 14 ∧ 
    chain'.closed ∧ 
    chain'.on_graph_paper ∧ 
    chain'.no_segments_on_same_line ∧
    self_intersection_points chain' = 17 :=
sorry

end max_self_intersections_l3005_300574


namespace repair_time_is_30_minutes_l3005_300548

/-- The time it takes to replace the buckle on one shoe (in minutes) -/
def buckle_time : ℕ := 5

/-- The time it takes to even out the heel on one shoe (in minutes) -/
def heel_time : ℕ := 10

/-- The number of shoes Melissa is repairing -/
def num_shoes : ℕ := 2

/-- The total time Melissa spends repairing her shoes -/
def total_repair_time : ℕ := (buckle_time + heel_time) * num_shoes

theorem repair_time_is_30_minutes : total_repair_time = 30 := by
  sorry

end repair_time_is_30_minutes_l3005_300548


namespace elements_beginning_with_3_l3005_300546

/-- The set of powers of 7 from 0 to 2011 -/
def T : Set ℕ := {n : ℕ | ∃ k : ℕ, 0 ≤ k ∧ k ≤ 2011 ∧ n = 7^k}

/-- The number of digits in 7^2011 -/
def digits_of_7_2011 : ℕ := 1602

/-- Function to check if a natural number begins with the digit 3 -/
def begins_with_3 (n : ℕ) : Prop := sorry

/-- The count of elements in T that begin with 3 -/
def count_begins_with_3 (S : Set ℕ) : ℕ := sorry

theorem elements_beginning_with_3 :
  count_begins_with_3 T = 45 :=
sorry

end elements_beginning_with_3_l3005_300546


namespace cos_alpha_value_l3005_300528

theorem cos_alpha_value (α : Real) 
  (h1 : π/4 < α) 
  (h2 : α < 3*π/4) 
  (h3 : Real.sin (α - π/4) = 4/5) : 
  Real.cos α = -Real.sqrt 2 / 10 := by
  sorry

end cos_alpha_value_l3005_300528


namespace first_concert_attendance_calculation_l3005_300588

/-- The number of people attending the second concert -/
def second_concert_attendance : ℕ := 66018

/-- The difference in attendance between the second and first concerts -/
def attendance_difference : ℕ := 119

/-- The number of people attending the first concert -/
def first_concert_attendance : ℕ := second_concert_attendance - attendance_difference

theorem first_concert_attendance_calculation :
  first_concert_attendance = 65899 :=
by sorry

end first_concert_attendance_calculation_l3005_300588


namespace hyperbola_focal_distance_l3005_300531

/-- A hyperbola with equation x²/m - y²/4 = 1 and focal distance 6 has m = 5 -/
theorem hyperbola_focal_distance (m : ℝ) : 
  (∃ (x y : ℝ), x^2/m - y^2/4 = 1) →  -- Hyperbola equation
  (∃ (c : ℝ), c = 3) →                -- Focal distance is 6, so c = 3
  m = 5 := by sorry

end hyperbola_focal_distance_l3005_300531


namespace probability_same_color_l3005_300514

/-- The number of marbles of each color in the box -/
def marbles_per_color : ℕ := 3

/-- The total number of colors -/
def num_colors : ℕ := 3

/-- The total number of marbles in the box -/
def total_marbles : ℕ := marbles_per_color * num_colors

/-- The number of marbles drawn -/
def drawn_marbles : ℕ := 3

/-- The probability of drawing 3 marbles of the same color -/
theorem probability_same_color :
  (num_colors * (Nat.choose marbles_per_color drawn_marbles)) /
  (Nat.choose total_marbles drawn_marbles) = 1 / 28 :=
sorry

end probability_same_color_l3005_300514


namespace negation_of_implication_l3005_300504

theorem negation_of_implication (a b c : ℝ) :
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c = 3 ∧ a^2 + b^2 + c^2 < 3) :=
by sorry

end negation_of_implication_l3005_300504


namespace shooting_game_probability_l3005_300551

-- Define the probability of hitting the target
variable (p : ℝ)

-- Define the number of shooting attempts
def η : ℕ → ℝ
| 1 => p
| 2 => (1 - p) * p
| 3 => (1 - p)^2
| _ => 0

-- Define the expected value of η
def E_η : ℝ := p + 2 * (1 - p) * p + 3 * (1 - p)^2

-- Theorem statement
theorem shooting_game_probability (h1 : 0 < p) (h2 : p < 1) (h3 : E_η > 7/4) :
  p ∈ Set.Ioo 0 (1/2) :=
sorry

end shooting_game_probability_l3005_300551


namespace no_geometric_sequence_trig_l3005_300501

open Real

theorem no_geometric_sequence_trig (θ : ℝ) : 
  0 < θ ∧ θ < 2 * π ∧ ¬ ∃ k : ℤ, θ = k * (π / 2) →
  ¬ (cos θ * tan θ = sin θ ^ 3 ∨ sin θ * cos θ = cos θ ^ 2 * tan θ) :=
by sorry

end no_geometric_sequence_trig_l3005_300501


namespace comparison_theorem_l3005_300553

theorem comparison_theorem (n : ℕ) (h : n ≥ 2) :
  (2^(2^2) * n < 3^(3^(3^3)) * n - 1) ∧
  (3^(3^(3^3)) * n > 4^(4^(4^4)) * n - 1) := by
  sorry

end comparison_theorem_l3005_300553


namespace six_digit_number_concatenation_divisibility_l3005_300544

theorem six_digit_number_concatenation_divisibility : 
  let a : ℕ := 166667
  let b : ℕ := 333334
  -- a and b are six-digit numbers
  (100000 ≤ a ∧ a < 1000000) ∧
  (100000 ≤ b ∧ b < 1000000) ∧
  -- The concatenated number is divisible by the product
  (1000000 * a + b) % (a * b) = 0 := by
sorry

end six_digit_number_concatenation_divisibility_l3005_300544


namespace lcm_factor_proof_l3005_300589

theorem lcm_factor_proof (A B : ℕ) (X : ℕ) : 
  A > 0 → B > 0 →
  Nat.gcd A B = 59 →
  Nat.lcm A B = 59 * X * 16 →
  A = 944 →
  X = 1 := by
sorry

end lcm_factor_proof_l3005_300589


namespace max_imaginary_part_of_roots_l3005_300569

theorem max_imaginary_part_of_roots (z : ℂ) (φ : ℝ) :
  z^6 - z^4 + z^2 - 1 = 0 →
  -π/2 ≤ φ ∧ φ ≤ π/2 →
  z.im = Real.sin φ →
  z.im ≤ Real.sin (π/4) :=
by sorry

end max_imaginary_part_of_roots_l3005_300569


namespace annual_fixed_costs_satisfy_profit_equation_l3005_300521

/-- Represents the annual fixed costs for Model X -/
def annual_fixed_costs : ℝ := 50200000

/-- Represents the desired annual profit -/
def desired_profit : ℝ := 30500000

/-- Represents the selling price per unit -/
def selling_price : ℝ := 9035

/-- Represents the variable cost per unit -/
def variable_cost : ℝ := 5000

/-- Represents the number of units sold -/
def units_sold : ℝ := 20000

/-- The profit equation -/
def profit_equation (fixed_costs : ℝ) : ℝ :=
  selling_price * units_sold - variable_cost * units_sold - fixed_costs

/-- Theorem stating that the annual fixed costs satisfy the profit equation -/
theorem annual_fixed_costs_satisfy_profit_equation :
  profit_equation annual_fixed_costs = desired_profit := by
  sorry

end annual_fixed_costs_satisfy_profit_equation_l3005_300521


namespace simplify_expression_l3005_300580

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^3 + b^2) - 2 * b^3 = 9 * b^4 + b^3 := by
  sorry

end simplify_expression_l3005_300580


namespace parallel_vectors_sum_magnitude_l3005_300585

/-- Given two parallel vectors p and q, prove that the magnitude of their sum is √13 -/
theorem parallel_vectors_sum_magnitude (p q : ℝ × ℝ) (x : ℝ) : 
  p = (2, -3) → 
  q = (x, 6) → 
  (2 * 6 = -3 * x) →  -- parallelism condition
  ‖p + q‖ = Real.sqrt 13 := by
sorry

end parallel_vectors_sum_magnitude_l3005_300585


namespace walters_coins_theorem_l3005_300555

/-- The value of a penny in cents -/
def penny : ℕ := 1

/-- The value of a nickel in cents -/
def nickel : ℕ := 5

/-- The value of a dime in cents -/
def dime : ℕ := 10

/-- The value of a quarter in cents -/
def quarter : ℕ := 25

/-- The number of cents in a dollar -/
def cents_in_dollar : ℕ := 100

/-- The percentage of a dollar represented by Walter's coins -/
def walters_coins_percentage : ℚ := (penny + nickel + dime + quarter : ℚ) / cents_in_dollar * 100

theorem walters_coins_theorem : walters_coins_percentage = 41 := by
  sorry

end walters_coins_theorem_l3005_300555


namespace ordering_of_exp_and_log_l3005_300577

theorem ordering_of_exp_and_log : 
  (Real.exp 0.1 - 1 : ℝ) > (0.1 : ℝ) ∧ (0.1 : ℝ) > Real.log 1.1 := by
  sorry

end ordering_of_exp_and_log_l3005_300577


namespace first_2500_even_integers_digits_l3005_300523

/-- The total number of digits used to write the first n positive even integers -/
def totalDigits (n : ℕ) : ℕ :=
  sorry

/-- The 2500th positive even integer -/
def evenInteger2500 : ℕ := 5000

theorem first_2500_even_integers_digits :
  totalDigits 2500 = 9449 :=
sorry

end first_2500_even_integers_digits_l3005_300523


namespace four_digit_square_decrease_theorem_l3005_300573

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def all_digits_decreasable (n k : ℕ) : Prop :=
  ∀ d, d ∈ [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10] → d ≥ k

def decrease_all_digits (n k : ℕ) : ℕ :=
  (n / 1000 - k) * 1000 + ((n / 100) % 10 - k) * 100 + ((n / 10) % 10 - k) * 10 + (n % 10 - k)

theorem four_digit_square_decrease_theorem :
  ∀ n : ℕ, is_four_digit n → is_perfect_square n →
  (∃ k : ℕ, k > 0 ∧ all_digits_decreasable n k ∧
   is_four_digit (decrease_all_digits n k) ∧ is_perfect_square (decrease_all_digits n k)) →
  n = 3136 ∨ n = 4489 := by sorry

end four_digit_square_decrease_theorem_l3005_300573


namespace clara_sticker_ratio_l3005_300571

/-- Given Clara's sticker distribution, prove the ratio of stickers given to best friends
    to stickers left after giving to the boy is 1:2 -/
theorem clara_sticker_ratio :
  ∀ (initial stickers_to_boy stickers_left : ℕ),
  initial = 100 →
  stickers_to_boy = 10 →
  stickers_left = 45 →
  (initial - stickers_to_boy - stickers_left) * 2 = initial - stickers_to_boy :=
by sorry

end clara_sticker_ratio_l3005_300571


namespace right_triangle_k_value_l3005_300524

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the vectors
def vector_AB (k : ℝ) : ℝ × ℝ := (k, 1)
def vector_AC : ℝ × ℝ := (2, 3)

-- Define the right angle condition
def is_right_angle (t : Triangle) : Prop :=
  let BC := (t.C.1 - t.B.1, t.C.2 - t.B.2)
  (t.C.1 - t.A.1) * BC.1 + (t.C.2 - t.A.2) * BC.2 = 0

-- Theorem statement
theorem right_triangle_k_value (t : Triangle) (k : ℝ) :
  is_right_angle t →
  t.B - t.A = vector_AB k →
  t.C - t.A = vector_AC →
  k = 5 := by
  sorry

end right_triangle_k_value_l3005_300524


namespace count_nine_digit_integers_l3005_300572

/-- The number of different 9-digit positive integers -/
def nine_digit_integers : ℕ := 9 * (10 ^ 8)

theorem count_nine_digit_integers : nine_digit_integers = 900000000 := by
  sorry

end count_nine_digit_integers_l3005_300572


namespace sphere_surface_area_of_circumscribed_prism_l3005_300549

theorem sphere_surface_area_of_circumscribed_prism (h : ℝ) (v : ℝ) (r : ℝ) :
  h = 4 →
  v = 16 →
  v = h * r^2 →
  let d := Real.sqrt (h^2 + 2 * r^2)
  (4 / 3) * π * (d / 2)^3 = (4 / 3) * π * r^2 * h →
  4 * π * (d / 2)^2 = 24 * π :=
by sorry

end sphere_surface_area_of_circumscribed_prism_l3005_300549


namespace family_member_bites_eq_two_l3005_300503

/-- The number of mosquito bites each family member (excluding Cyrus) has, given the conditions in the problem. -/
def family_member_bites : ℕ :=
  let cyrus_arm_leg_bites : ℕ := 14
  let cyrus_body_bites : ℕ := 10
  let cyrus_total_bites : ℕ := cyrus_arm_leg_bites + cyrus_body_bites
  let family_size : ℕ := 6
  let family_total_bites : ℕ := cyrus_total_bites / 2
  family_total_bites / family_size

theorem family_member_bites_eq_two : family_member_bites = 2 := by
  sorry

end family_member_bites_eq_two_l3005_300503


namespace profit_difference_l3005_300525

-- Define the types of statues
inductive StatueType
| Giraffe
| Elephant
| Rhinoceros

-- Define the properties of each statue type
def jade_required (s : StatueType) : ℕ :=
  match s with
  | StatueType.Giraffe => 120
  | StatueType.Elephant => 240
  | StatueType.Rhinoceros => 180

def original_price (s : StatueType) : ℕ :=
  match s with
  | StatueType.Giraffe => 150
  | StatueType.Elephant => 350
  | StatueType.Rhinoceros => 250

-- Define the bulk discount
def bulk_discount : ℚ := 0.9

-- Define the total jade available
def total_jade : ℕ := 1920

-- Calculate the number of statues that can be made
def num_statues (s : StatueType) : ℕ :=
  total_jade / jade_required s

-- Calculate the revenue for a statue type
def revenue (s : StatueType) : ℚ :=
  if num_statues s > 3 then
    (num_statues s : ℚ) * (original_price s : ℚ) * bulk_discount
  else
    (num_statues s : ℚ) * (original_price s : ℚ)

-- Theorem to prove
theorem profit_difference : 
  revenue StatueType.Elephant - revenue StatueType.Rhinoceros = 270 := by
  sorry

end profit_difference_l3005_300525


namespace michael_digging_time_l3005_300513

/-- The time it takes Michael to dig his hole given the conditions -/
theorem michael_digging_time 
  (father_rate : ℝ) 
  (father_time : ℝ) 
  (michael_rate : ℝ) 
  (michael_depth_diff : ℝ) :
  father_rate = 4 →
  father_time = 400 →
  michael_rate = father_rate →
  michael_depth_diff = 400 →
  (2 * (father_rate * father_time) - michael_depth_diff) / michael_rate = 700 :=
by sorry

end michael_digging_time_l3005_300513


namespace tara_wrong_questions_l3005_300516

theorem tara_wrong_questions
  (total_questions : ℕ)
  (t u v w : ℕ)
  (h1 : t + u = v + w)
  (h2 : t + w = u + v + 6)
  (h3 : v = 3)
  (h4 : total_questions = 40) :
  t = 9 := by
sorry

end tara_wrong_questions_l3005_300516


namespace max_value_of_expression_l3005_300557

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y)^2 / (x^2 + y^2) ≤ 2 ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (a + b)^2 / (a^2 + b^2) = 2 :=
by sorry

end max_value_of_expression_l3005_300557


namespace car_stopping_distance_l3005_300576

/-- Represents the distance traveled by a car in a given second -/
def distance_per_second (n : ℕ) : ℕ :=
  max (40 - 10 * n) 0

/-- Calculates the total distance traveled by the car -/
def total_distance : ℕ :=
  (List.range 5).map distance_per_second |>.sum

/-- Theorem: The total distance traveled by the car is 100 feet -/
theorem car_stopping_distance : total_distance = 100 := by
  sorry

#eval total_distance

end car_stopping_distance_l3005_300576


namespace seats_per_bus_l3005_300581

/-- Given a school trip scenario with students and buses, calculate the number of seats per bus. -/
theorem seats_per_bus (students : ℕ) (buses : ℕ) (h1 : students = 111) (h2 : buses = 37) :
  students / buses = 3 := by
  sorry


end seats_per_bus_l3005_300581


namespace not_divides_for_all_m_l3005_300583

theorem not_divides_for_all_m : ∀ m : ℕ, ¬((1000^m - 1) ∣ (1978^m - 1)) := by
  sorry

end not_divides_for_all_m_l3005_300583


namespace product_of_cubic_fractions_l3005_300520

theorem product_of_cubic_fractions :
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 2) * (f 3) * (f 4) * (f 5) * (f 6) = 43 / 63 := by
sorry

end product_of_cubic_fractions_l3005_300520


namespace sphere_surface_area_containing_unit_cube_l3005_300505

/-- The surface area of a sphere that contains all eight vertices of a unit cube -/
theorem sphere_surface_area_containing_unit_cube : ℝ := by
  -- Define a cube with edge length 1
  let cube_edge_length : ℝ := 1

  -- Define the sphere that contains all vertices of the cube
  let sphere_radius : ℝ := (Real.sqrt 3) / 2

  -- Define the surface area of the sphere
  let sphere_surface_area : ℝ := 4 * Real.pi * sphere_radius^2

  -- Prove that the surface area equals 3π
  have : sphere_surface_area = 3 * Real.pi := by sorry

  -- Return the result
  exact 3 * Real.pi


end sphere_surface_area_containing_unit_cube_l3005_300505


namespace original_number_l3005_300511

theorem original_number (N : ℕ) : (∀ k : ℕ, N - 7 ≠ 12 * k) ∧ (∃ k : ℕ, N - 7 = 12 * k) → N = 7 := by
  sorry

end original_number_l3005_300511


namespace base_equation_solution_l3005_300506

/-- Converts a base-10 number to base-a representation --/
def toBaseA (n : ℕ) (a : ℕ) : List ℕ := sorry

/-- Converts a base-a number to base-10 representation --/
def fromBaseA (digits : List ℕ) (a : ℕ) : ℕ := sorry

/-- Adds two numbers in base-a --/
def addBaseA (n1 : List ℕ) (n2 : List ℕ) (a : ℕ) : List ℕ := sorry

theorem base_equation_solution :
  ∃! a : ℕ, 
    a > 11 ∧ 
    addBaseA (toBaseA 396 a) (toBaseA 574 a) a = toBaseA (96 * 11) a := by
  sorry

end base_equation_solution_l3005_300506


namespace age_ratio_in_3_years_l3005_300507

def franks_current_age : ℕ := 12
def johns_current_age : ℕ := franks_current_age + 15

def franks_age_in_3_years : ℕ := franks_current_age + 3
def johns_age_in_3_years : ℕ := johns_current_age + 3

theorem age_ratio_in_3_years :
  ∃ (k : ℕ), k > 0 ∧ johns_age_in_3_years = k * franks_age_in_3_years ∧
  johns_age_in_3_years / franks_age_in_3_years = 2 :=
sorry

end age_ratio_in_3_years_l3005_300507


namespace intersection_implies_a_values_l3005_300582

def A (a : ℝ) : Set ℝ := {-1, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {9, 5-a, 4-a}

theorem intersection_implies_a_values (a : ℝ) :
  A a ∩ B a = {9} → a = 3 ∨ a = -3 := by
  sorry

end intersection_implies_a_values_l3005_300582


namespace floor_sqrt_equality_l3005_300515

theorem floor_sqrt_equality (n : ℕ+) :
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (4 * n + 1)⌋ ∧
  ⌊Real.sqrt (4 * n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ ∧
  ⌊Real.sqrt (4 * n + 2)⌋ = ⌊Real.sqrt (4 * n + 3)⌋ := by
  sorry

end floor_sqrt_equality_l3005_300515


namespace gcd_square_le_sum_l3005_300512

theorem gcd_square_le_sum (a b : ℕ) (h1 : (a + 1) % b = 0) (h2 : (b + 1) % a = 0) : 
  (Nat.gcd a b)^2 ≤ a + b := by
  sorry

end gcd_square_le_sum_l3005_300512


namespace sophia_stamp_collection_value_l3005_300562

/-- Given a collection of stamps with equal value, calculate the total value. -/
def stamp_collection_value (total_stamps : ℕ) (sample_stamps : ℕ) (sample_value : ℕ) : ℕ :=
  total_stamps * (sample_value / sample_stamps)

/-- Theorem: Sophia's stamp collection is worth 120 dollars. -/
theorem sophia_stamp_collection_value :
  stamp_collection_value 24 8 40 = 120 := by
  sorry

#eval stamp_collection_value 24 8 40

end sophia_stamp_collection_value_l3005_300562


namespace kenya_peanuts_count_l3005_300529

/-- The number of peanuts Jose has -/
def jose_peanuts : ℕ := 85

/-- The additional number of peanuts Kenya has compared to Jose -/
def kenya_extra_peanuts : ℕ := 48

/-- The number of peanuts Kenya has -/
def kenya_peanuts : ℕ := jose_peanuts + kenya_extra_peanuts

theorem kenya_peanuts_count : kenya_peanuts = 133 := by
  sorry

end kenya_peanuts_count_l3005_300529


namespace geometric_sequence_sum_l3005_300542

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum (n : ℕ) :
  geometric_sum (1/3) (1/3) n = 26/81 → n = 4 := by
  sorry

end geometric_sequence_sum_l3005_300542


namespace inequality_one_inequality_two_l3005_300537

-- Part 1
theorem inequality_one (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a := by
  sorry

-- Part 2
theorem inequality_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  a*b + b*c + c*a ≤ 1/3 := by
  sorry

end inequality_one_inequality_two_l3005_300537


namespace hardcover_probability_l3005_300584

theorem hardcover_probability (total_books : Nat) (hardcover_books : Nat) (selected_books : Nat) :
  total_books = 15 →
  hardcover_books = 5 →
  selected_books = 3 →
  (Nat.choose hardcover_books selected_books * Nat.choose (total_books - hardcover_books) (selected_books - hardcover_books) +
   Nat.choose hardcover_books (selected_books - 1) * Nat.choose (total_books - hardcover_books) 1 +
   Nat.choose hardcover_books selected_books) / Nat.choose total_books selected_books = 67 / 91 := by
  sorry

end hardcover_probability_l3005_300584


namespace abs_z_equals_five_l3005_300550

theorem abs_z_equals_five (z : ℂ) (h : z - 3 = (3 + I) / I) : Complex.abs z = 5 := by
  sorry

end abs_z_equals_five_l3005_300550


namespace sequence_is_arithmetic_l3005_300534

/-- The sum of the first n terms of the sequence -/
def S (a n : ℕ) : ℕ := a * n^2 + n

/-- The n-th term of the sequence -/
def a_n (a n : ℕ) : ℤ := S a n - S a (n-1)

/-- A sequence is arithmetic if the difference between consecutive terms is constant -/
def is_arithmetic_sequence (f : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, f (n+1) - f n = d

theorem sequence_is_arithmetic (a : ℕ) (h : a > 0) :
  is_arithmetic_sequence (a_n a) := by
  sorry

end sequence_is_arithmetic_l3005_300534


namespace paint_intensity_problem_l3005_300547

theorem paint_intensity_problem (original_intensity new_intensity replacement_fraction : ℚ)
  (h1 : original_intensity = 15 / 100)
  (h2 : new_intensity = 30 / 100)
  (h3 : replacement_fraction = 3 / 2)
  : ∃ added_intensity : ℚ,
    added_intensity = 40 / 100 ∧
    (original_intensity * (1 / (1 + replacement_fraction)) + 
     added_intensity * (replacement_fraction / (1 + replacement_fraction)) = new_intensity) :=
by sorry

end paint_intensity_problem_l3005_300547


namespace mike_seashells_l3005_300552

/-- The total number of seashells Mike found -/
def total_seashells (initial : ℝ) (later : ℝ) : ℝ := initial + later

/-- Theorem stating that Mike found 10.75 seashells in total -/
theorem mike_seashells :
  let initial_seashells : ℝ := 6.5
  let later_seashells : ℝ := 4.25
  total_seashells initial_seashells later_seashells = 10.75 := by
  sorry

end mike_seashells_l3005_300552


namespace fly_path_distance_l3005_300526

theorem fly_path_distance (r : ℝ) (last_leg : ℝ) (h1 : r = 60) (h2 : last_leg = 85) :
  let diameter := 2 * r
  let second_leg := Real.sqrt (diameter^2 - last_leg^2)
  diameter + last_leg + second_leg = 205 + Real.sqrt 7175 := by
  sorry

end fly_path_distance_l3005_300526


namespace trapezoid_area_l3005_300593

/-- Given an outer equilateral triangle with area 64 and an inner equilateral triangle
    with area 4, where the space between them is divided into three congruent trapezoids,
    prove that the area of one trapezoid is 20. -/
theorem trapezoid_area (outer_area inner_area : ℝ) (h1 : outer_area = 64) (h2 : inner_area = 4) :
  (outer_area - inner_area) / 3 = 20 := by
  sorry

end trapezoid_area_l3005_300593


namespace chemistry_class_gender_difference_l3005_300535

theorem chemistry_class_gender_difference :
  ∀ (boys girls : ℕ),
  (3 : ℕ) * boys = (4 : ℕ) * girls →
  boys + girls = 42 →
  girls - boys = 6 :=
by sorry

end chemistry_class_gender_difference_l3005_300535


namespace apple_pyramid_theorem_l3005_300587

/-- Calculates the number of apples in a layer of the pyramid --/
def apples_in_layer (base_width : ℕ) (base_length : ℕ) (layer : ℕ) : ℕ :=
  (base_width - layer + 1) * (base_length - layer + 1)

/-- Calculates the total number of apples in the pyramid --/
def total_apples (base_width : ℕ) (base_length : ℕ) : ℕ :=
  let max_layers := min base_width base_length
  (List.range max_layers).foldl (fun acc layer => acc + apples_in_layer base_width base_length layer) 0

/-- The theorem stating that a pyramid with a 6x9 base contains 154 apples --/
theorem apple_pyramid_theorem :
  total_apples 6 9 = 154 := by
  sorry

end apple_pyramid_theorem_l3005_300587


namespace bucket_capacity_proof_l3005_300567

theorem bucket_capacity_proof (x : ℝ) : 
  (12 * x = 132 * 5) → x = 55 := by
  sorry

end bucket_capacity_proof_l3005_300567


namespace min_balls_guarantee_l3005_300545

def red_balls : ℕ := 35
def blue_balls : ℕ := 25
def green_balls : ℕ := 22
def yellow_balls : ℕ := 18
def white_balls : ℕ := 14
def black_balls : ℕ := 12

def total_balls : ℕ := red_balls + blue_balls + green_balls + yellow_balls + white_balls + black_balls

def min_balls_for_guarantee : ℕ := 95

theorem min_balls_guarantee :
  ∀ (drawn : ℕ), drawn ≥ min_balls_for_guarantee →
    ∃ (color : ℕ), color ≥ 18 ∧
      (color ≤ red_balls ∨ color ≤ blue_balls ∨ color ≤ green_balls ∨
       color ≤ yellow_balls ∨ color ≤ white_balls ∨ color ≤ black_balls) :=
by sorry

end min_balls_guarantee_l3005_300545


namespace jasons_military_career_l3005_300558

theorem jasons_military_career (join_age retire_age : ℕ) 
  (chief_to_master_chief_factor : ℚ) (additional_years : ℕ) :
  join_age = 18 →
  retire_age = 46 →
  chief_to_master_chief_factor = 1.25 →
  additional_years = 10 →
  ∃ (years_to_chief : ℕ),
    years_to_chief + (chief_to_master_chief_factor * years_to_chief) + additional_years = retire_age - join_age ∧
    years_to_chief = 8 := by
  sorry

end jasons_military_career_l3005_300558


namespace monochromatic_cycle_exists_l3005_300578

/-- A complete bipartite graph K_{n,n} -/
structure CompleteBipartiteGraph (n : ℕ) where
  left : Fin n
  right : Fin n

/-- A 2-coloring of edges -/
def Coloring (n : ℕ) := CompleteBipartiteGraph n → Bool

/-- A 4-cycle in the graph -/
structure Cycle4 (n : ℕ) where
  v1 : Fin n
  v2 : Fin n
  v3 : Fin n
  v4 : Fin n

/-- Check if a 4-cycle is monochromatic under a given coloring -/
def isMonochromatic (n : ℕ) (c : Coloring n) (cycle : Cycle4 n) : Prop :=
  let color1 := c ⟨cycle.v1, cycle.v2⟩
  let color2 := c ⟨cycle.v2, cycle.v3⟩
  let color3 := c ⟨cycle.v3, cycle.v4⟩
  let color4 := c ⟨cycle.v4, cycle.v1⟩
  color1 = color2 ∧ color2 = color3 ∧ color3 = color4

/-- The main theorem: Any 2-coloring of K_{5,5} contains a monochromatic 4-cycle -/
theorem monochromatic_cycle_exists :
  ∀ (c : Coloring 5), ∃ (cycle : Cycle4 5), isMonochromatic 5 c cycle :=
sorry

end monochromatic_cycle_exists_l3005_300578


namespace k_squared_upper_bound_l3005_300522

theorem k_squared_upper_bound (k n : ℕ) (h1 : 121 < k^2) (h2 : k^2 < n) 
  (h3 : ∀ m : ℕ, 121 < m^2 → m^2 < n → m ≤ k + 5) : n ≤ 324 :=
sorry

end k_squared_upper_bound_l3005_300522


namespace equation_graph_is_two_parallel_lines_l3005_300598

-- Define the equation
def equation (x y : ℝ) : Prop := x^3 * (x + y + 2) = y^3 * (x + y + 2)

-- Define what it means for two lines to be parallel
def parallel (l₁ l₂ : ℝ → ℝ) : Prop := 
  ∃ (k : ℝ), ∀ x, l₂ x = l₁ x + k

-- Theorem statement
theorem equation_graph_is_two_parallel_lines :
  ∃ (l₁ l₂ : ℝ → ℝ), 
    (∀ x y, equation x y ↔ (y = l₁ x ∨ y = l₂ x)) ∧
    parallel l₁ l₂ :=
sorry

end equation_graph_is_two_parallel_lines_l3005_300598


namespace smaller_bucket_capacity_proof_l3005_300597

/-- The capacity of the smaller bucket in liters -/
def smaller_bucket_capacity : ℝ := 3

/-- The capacity of the medium bucket in liters -/
def medium_bucket_capacity : ℝ := 5

/-- The capacity of the larger bucket in liters -/
def larger_bucket_capacity : ℝ := 6

/-- The amount of water that can be added to the larger bucket after pouring from the medium bucket -/
def remaining_capacity : ℝ := 4

theorem smaller_bucket_capacity_proof :
  smaller_bucket_capacity = medium_bucket_capacity - (larger_bucket_capacity - remaining_capacity) :=
by sorry

end smaller_bucket_capacity_proof_l3005_300597


namespace symmetric_points_sum_l3005_300565

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other. -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The theorem states that if points A(m-1, -3) and B(2, n) are symmetric with respect to the origin,
    then m + n = 2. -/
theorem symmetric_points_sum (m n : ℝ) :
  symmetric_wrt_origin (m - 1) (-3) 2 n → m + n = 2 := by
  sorry

end symmetric_points_sum_l3005_300565


namespace easter_egg_hunt_l3005_300540

theorem easter_egg_hunt (total_eggs : ℕ) 
  (hannah_ratio : ℕ) (harry_extra : ℕ) : 
  total_eggs = 63 ∧ hannah_ratio = 2 ∧ harry_extra = 3 →
  ∃ (helen_eggs hannah_eggs harry_eggs : ℕ),
    helen_eggs = 12 ∧
    hannah_eggs = 24 ∧
    harry_eggs = 27 ∧
    hannah_eggs = hannah_ratio * helen_eggs ∧
    harry_eggs = hannah_eggs + harry_extra ∧
    helen_eggs + hannah_eggs + harry_eggs = total_eggs :=
by sorry

end easter_egg_hunt_l3005_300540


namespace valid_numerical_pyramid_exists_l3005_300561

/-- Represents a row in the numerical pyramid --/
structure PyramidRow where
  digits : List ℕ
  result : ℕ

/-- Represents the entire numerical pyramid --/
structure NumericalPyramid where
  row1 : PyramidRow
  row2 : PyramidRow
  row3 : PyramidRow
  row4 : PyramidRow
  row5 : PyramidRow
  row6 : PyramidRow
  row7 : PyramidRow

/-- Function to check if a pyramid satisfies all conditions --/
def is_valid_pyramid (p : NumericalPyramid) : Prop :=
  p.row1.digits = [1, 2] ∧ p.row1.result = 3 ∧
  p.row2.digits = [1, 2, 3] ∧ p.row2.result = 4 ∧
  p.row3.digits = [1, 2, 3, 4] ∧ p.row3.result = 5 ∧
  p.row4.digits = [1, 2, 3, 4, 5] ∧ p.row4.result = 6 ∧
  p.row5.digits = [1, 2, 3, 4, 5, 6] ∧ p.row5.result = 7 ∧
  p.row6.digits = [1, 2, 3, 4, 5, 6, 7] ∧ p.row6.result = 8 ∧
  p.row7.digits = [1, 2, 3, 4, 5, 6, 7, 8] ∧ p.row7.result = 9

/-- Theorem stating that a valid numerical pyramid exists --/
theorem valid_numerical_pyramid_exists : ∃ (p : NumericalPyramid), is_valid_pyramid p := by
  sorry

end valid_numerical_pyramid_exists_l3005_300561


namespace pool_water_volume_l3005_300554

/-- Calculates the remaining water volume in a pool after evaporation --/
def remaining_water_volume (initial_volume : ℕ) (evaporation_rate : ℕ) (days : ℕ) : ℕ :=
  initial_volume - evaporation_rate * days

/-- Theorem: The remaining water volume after 45 days is 355 gallons --/
theorem pool_water_volume : 
  remaining_water_volume 400 1 45 = 355 := by
  sorry

end pool_water_volume_l3005_300554


namespace negation_of_proposition_l3005_300508

theorem negation_of_proposition (P : ℝ → Prop) : 
  (∀ x : ℝ, x^2 - 2*x + 2 > 0) ↔ ¬(∃ x : ℝ, x^2 - 2*x + 2 ≤ 0) :=
by sorry

end negation_of_proposition_l3005_300508


namespace final_depth_calculation_l3005_300575

/-- Calculates the final depth aimed to dig given initial and new working conditions -/
theorem final_depth_calculation 
  (initial_men : ℕ) 
  (initial_hours : ℕ) 
  (initial_depth : ℕ) 
  (extra_men : ℕ) 
  (new_hours : ℕ) : 
  initial_men = 75 → 
  initial_hours = 8 → 
  initial_depth = 50 → 
  extra_men = 65 → 
  new_hours = 6 → 
  (initial_men + extra_men) * new_hours * initial_depth = initial_men * initial_hours * 70 := by
  sorry

#check final_depth_calculation

end final_depth_calculation_l3005_300575


namespace cosine_relationship_triangle_area_l3005_300590

/-- Represents a triangle with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem stating the relationship between cosines in a triangle -/
theorem cosine_relationship (t : Triangle) 
  (h : t.a * Real.cos t.C = (2 * t.b - t.c) * Real.cos t.A) : 
  Real.cos t.A = 1 / 2 := by sorry

/-- Theorem for calculating the area of a specific triangle -/
theorem triangle_area (t : Triangle) 
  (h1 : t.a = 6) 
  (h2 : t.b + t.c = 8) 
  (h3 : Real.cos t.A = 1 / 2) : 
  (1 / 2) * t.a * t.b * Real.sin t.C = 7 * Real.sqrt 3 / 3 := by sorry

end cosine_relationship_triangle_area_l3005_300590


namespace simplify_expression_l3005_300556

theorem simplify_expression (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ 0) (h3 : a ≠ -1) :
  (a^2 - 2*a + 1) / (a^2 - 1) / (a - 2*a / (a + 1)) = 1 / a :=
by sorry

end simplify_expression_l3005_300556


namespace common_measure_proof_l3005_300566

def segment1 : ℚ := 1/5
def segment2 : ℚ := 1/3
def commonMeasure : ℚ := 1/15

theorem common_measure_proof :
  (∃ (n m : ℕ), n * commonMeasure = segment1 ∧ m * commonMeasure = segment2) ∧
  (∀ (x : ℚ), x > 0 → (∃ (n m : ℕ), n * x = segment1 ∧ m * x = segment2) → x ≤ commonMeasure) :=
by sorry

end common_measure_proof_l3005_300566


namespace red_peaches_count_l3005_300530

theorem red_peaches_count (total : ℕ) (yellow : ℕ) (green : ℕ) (red : ℕ) 
  (h1 : total = 30)
  (h2 : yellow = 15)
  (h3 : green = 8)
  (h4 : total = red + yellow + green) : 
  red = 7 := by
  sorry

end red_peaches_count_l3005_300530


namespace prime_roots_sum_fraction_l3005_300560

theorem prime_roots_sum_fraction (p q m : ℕ) : 
  Prime p → Prime q → 
  p^2 - 99*p + m = 0 → 
  q^2 - 99*q + m = 0 → 
  (p : ℚ) / q + (q : ℚ) / p = 9413 / 194 := by
  sorry

end prime_roots_sum_fraction_l3005_300560
