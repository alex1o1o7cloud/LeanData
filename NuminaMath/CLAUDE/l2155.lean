import Mathlib

namespace sum_of_inverse_G_power_three_l2155_215504

def G : ℕ → ℚ
  | 0 => 0
  | 1 => 8/3
  | (n+2) => 3 * G (n+1) - (1/2) * G n

theorem sum_of_inverse_G_power_three : ∑' n, 1 / G (3^n) = 1 := by sorry

end sum_of_inverse_G_power_three_l2155_215504


namespace jame_card_tearing_l2155_215594

/-- The number of cards Jame can tear at a time -/
def cards_per_tear : ℕ := 30

/-- The number of cards in a new deck -/
def cards_per_deck : ℕ := 55

/-- The number of times Jame tears cards per week -/
def tears_per_week : ℕ := 3

/-- The number of decks Jame buys -/
def decks_bought : ℕ := 18

/-- The number of weeks Jame can tear cards -/
def weeks_of_tearing : ℕ := 11

theorem jame_card_tearing :
  (cards_per_tear * tears_per_week) * weeks_of_tearing ≤ cards_per_deck * decks_bought ∧
  (cards_per_tear * tears_per_week) * (weeks_of_tearing + 1) > cards_per_deck * decks_bought :=
by sorry

end jame_card_tearing_l2155_215594


namespace sara_pumpkins_l2155_215562

/-- The number of pumpkins eaten by rabbits -/
def pumpkins_eaten : ℕ := 23

/-- The number of pumpkins Sara has left -/
def pumpkins_left : ℕ := 20

/-- The original number of pumpkins Sara grew -/
def original_pumpkins : ℕ := pumpkins_eaten + pumpkins_left

theorem sara_pumpkins : original_pumpkins = 43 := by
  sorry

end sara_pumpkins_l2155_215562


namespace octal_131_equals_binary_1011001_l2155_215585

-- Define octal_to_decimal function
def octal_to_decimal (octal : ℕ) : ℕ :=
  let ones := octal % 10
  let eights := (octal / 10) % 10
  let sixty_fours := octal / 100
  ones + 8 * eights + 64 * sixty_fours

-- Define decimal_to_binary function
def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

-- Theorem statement
theorem octal_131_equals_binary_1011001 :
  decimal_to_binary (octal_to_decimal 131) = [1, 0, 1, 1, 0, 0, 1] :=
sorry

end octal_131_equals_binary_1011001_l2155_215585


namespace not_right_triangle_sides_l2155_215503

theorem not_right_triangle_sides (a b c : ℝ) (h1 : a = Real.sqrt 3) (h2 : b = Real.sqrt 4) (h3 : c = Real.sqrt 5) :
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :=
by sorry

end not_right_triangle_sides_l2155_215503


namespace reciprocal_of_negative_2022_l2155_215512

theorem reciprocal_of_negative_2022 : ((-2022)⁻¹ : ℚ) = -1 / 2022 := by
  sorry

end reciprocal_of_negative_2022_l2155_215512


namespace mixture_composition_l2155_215507

-- Define the initial mixture
def initial_mixture : ℝ := 90

-- Define the initial milk to water ratio
def milk_water_ratio : ℚ := 2 / 1

-- Define the amount of water evaporated
def water_evaporated : ℝ := 10

-- Define the relation between liquid L and milk
def liquid_L_milk_ratio : ℚ := 1 / 3

-- Define the relation between milk and water after additions
def final_milk_water_ratio : ℚ := 2 / 1

-- Theorem to prove
theorem mixture_composition :
  let initial_milk := initial_mixture * (milk_water_ratio / (1 + milk_water_ratio))
  let initial_water := initial_mixture * (1 / (1 + milk_water_ratio))
  let remaining_water := initial_water - water_evaporated
  let liquid_L := initial_milk * liquid_L_milk_ratio
  let final_milk := initial_milk
  let final_water := remaining_water
  (liquid_L = 20) ∧ (final_milk / final_water = 3 / 1) := by
  sorry

end mixture_composition_l2155_215507


namespace leading_zeros_in_decimal_representation_l2155_215539

theorem leading_zeros_in_decimal_representation (n : ℕ) (m : ℕ) :
  (∃ k : ℕ, (1 : ℚ) / (2^7 * 5^3) = (k : ℚ) / 10^n ∧ 
   k ≠ 0 ∧ k < 10^m) → n - m = 5 := by
  sorry

end leading_zeros_in_decimal_representation_l2155_215539


namespace grunters_win_probability_l2155_215559

/-- The number of games played -/
def n : ℕ := 6

/-- The number of games to be won -/
def k : ℕ := 5

/-- The probability of winning a single game -/
def p : ℚ := 4/5

/-- The probability of winning exactly k out of n games -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem grunters_win_probability :
  binomial_probability n k p = 6144/15625 := by
  sorry

end grunters_win_probability_l2155_215559


namespace composition_equation_solution_l2155_215537

/-- Given functions f and g, prove that if f(g(a)) = 4, then a = 3/4 -/
theorem composition_equation_solution (f g : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = (2*x - 1) / 3 + 2)
  (hg : ∀ x, g x = 5 - 2*x)
  (h : f (g a) = 4) : 
  a = 3/4 := by
  sorry

end composition_equation_solution_l2155_215537


namespace klinked_from_connectivity_and_edges_l2155_215550

/-- A graph is k-linked if for any k pairs of vertices (s₁, t₁), ..., (sₖ, tₖ),
    there exist k vertex-disjoint paths P₁, ..., Pₖ such that Pᵢ connects sᵢ to tᵢ. -/
def IsKLinked (G : SimpleGraph α) (k : ℕ) : Prop := sorry

/-- A graph is k-connected if it remains connected after removing any k-1 vertices. -/
def IsKConnected (G : SimpleGraph α) (k : ℕ) : Prop := sorry

/-- The number of edges in a graph. -/
def NumEdges (G : SimpleGraph α) : ℕ := sorry

theorem klinked_from_connectivity_and_edges
  {α : Type*} (G : SimpleGraph α) (k : ℕ) :
  IsKConnected G (2 * k) →
  NumEdges G ≥ 8 * k →
  IsKLinked G k :=
sorry

end klinked_from_connectivity_and_edges_l2155_215550


namespace homework_situations_l2155_215564

/-- The number of teachers who have assigned homework -/
def num_teachers : ℕ := 3

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of possible homework situations for all students -/
def total_situations : ℕ := num_teachers ^ num_students

theorem homework_situations :
  total_situations = 3^4 := by
  sorry

end homework_situations_l2155_215564


namespace pizza_slices_l2155_215597

theorem pizza_slices (total_pizzas : ℕ) (total_slices : ℕ) (slices_per_pizza : ℕ) 
  (h1 : total_pizzas = 7)
  (h2 : total_slices = 14)
  (h3 : total_slices = total_pizzas * slices_per_pizza) :
  slices_per_pizza = 2 := by
  sorry

end pizza_slices_l2155_215597


namespace marcus_pebble_count_l2155_215596

/-- Given an initial number of pebbles, calculate the number of pebbles
    after skipping half and receiving more. -/
def final_pebble_count (initial : ℕ) (received : ℕ) : ℕ :=
  initial / 2 + received

/-- Theorem stating that given 18 initial pebbles and 30 received pebbles,
    the final count is 39. -/
theorem marcus_pebble_count :
  final_pebble_count 18 30 = 39 := by
  sorry

end marcus_pebble_count_l2155_215596


namespace cube_root_of_negative_27_l2155_215548

theorem cube_root_of_negative_27 : ∃ x : ℝ, x^3 = -27 ∧ x = -3 := by sorry

end cube_root_of_negative_27_l2155_215548


namespace shopping_spree_remaining_amount_l2155_215543

def initial_amount : ℝ := 78

def kite_price_euro : ℝ := 6
def euro_to_usd : ℝ := 1.2

def frisbee_price_pound : ℝ := 7
def pound_to_usd : ℝ := 1.4

def roller_skates_price : ℝ := 15
def roller_skates_discount : ℝ := 0.125

def lego_set_price : ℝ := 25
def lego_set_discount : ℝ := 0.15

def puzzle_price : ℝ := 12
def puzzle_tax : ℝ := 0.075

def remaining_amount : ℝ := initial_amount - 
  (kite_price_euro * euro_to_usd +
   frisbee_price_pound * pound_to_usd +
   roller_skates_price * (1 - roller_skates_discount) +
   lego_set_price * (1 - lego_set_discount) +
   puzzle_price * (1 + puzzle_tax))

theorem shopping_spree_remaining_amount : 
  remaining_amount = 13.725 := by sorry

end shopping_spree_remaining_amount_l2155_215543


namespace machine_does_not_require_repair_no_repair_needed_l2155_215567

/-- Represents the nominal portion weight in grams -/
def nominal_weight : ℝ := 390

/-- Represents the greatest deviation from the mean among preserved measurements in grams -/
def max_deviation : ℝ := 39

/-- Represents the threshold for requiring repair in grams -/
def repair_threshold : ℝ := 39

/-- Condition: The greatest deviation does not exceed 10% of the nominal weight -/
axiom max_deviation_condition : max_deviation ≤ 0.1 * nominal_weight

/-- Condition: All deviations are no more than the maximum deviation -/
axiom all_deviations_bounded (deviation : ℝ) : deviation ≤ max_deviation

/-- Condition: The standard deviation does not exceed the greatest deviation -/
axiom standard_deviation_bounded (σ : ℝ) : σ ≤ max_deviation

/-- Theorem: The standard deviation is no more than the repair threshold -/
theorem machine_does_not_require_repair (σ : ℝ) : 
  σ ≤ repair_threshold :=
sorry

/-- Corollary: The machine does not require repair -/
theorem no_repair_needed : 
  ∃ (σ : ℝ), σ ≤ repair_threshold :=
sorry

end machine_does_not_require_repair_no_repair_needed_l2155_215567


namespace absolute_value_and_exponents_l2155_215561

theorem absolute_value_and_exponents : |-3| + 2^2 - (Real.sqrt 3 - 1)^0 = 6 := by
  sorry

end absolute_value_and_exponents_l2155_215561


namespace part_one_part_two_l2155_215540

/-- Given expressions for A and B -/
def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := -a^2 + a * b - 1

/-- Theorem for part 1 -/
theorem part_one (a b : ℝ) :
  4 * A a b - (3 * A a b - 2 * B a b) = 5 * a * b - 2 * a - 3 := by sorry

/-- Theorem for part 2 -/
theorem part_two (b : ℝ) :
  (∀ a : ℝ, A a b + 2 * B a b = A 0 b + 2 * B 0 b) → b = 2/5 := by sorry

end part_one_part_two_l2155_215540


namespace correct_time_exists_l2155_215584

/-- Represents the position of a watch hand on the face of the watch -/
def HandPosition := ℝ

/-- Represents the angle of rotation for the watch dial -/
def DialRotation := ℝ

/-- Represents a point in time within a 24-hour period -/
def TimePoint := ℝ

/-- A watch with fixed hour and minute hands -/
structure Watch where
  hourHand : HandPosition
  minuteHand : HandPosition

/-- Calculates the correct angle between hour and minute hands for a given time -/
noncomputable def correctAngle (t : TimePoint) : ℝ :=
  sorry

/-- Calculates the actual angle between hour and minute hands for a given watch and dial rotation -/
noncomputable def actualAngle (w : Watch) (r : DialRotation) : ℝ :=
  sorry

/-- States that for any watch with fixed hands, there exists a dial rotation
    such that the watch shows the correct time at least once in a 24-hour period -/
theorem correct_time_exists (w : Watch) :
  ∃ r : DialRotation, ∃ t : TimePoint, actualAngle w r = correctAngle t :=
sorry

end correct_time_exists_l2155_215584


namespace sum_edges_vertices_faces_l2155_215534

/-- A rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The number of edges in a rectangular prism -/
def num_edges (p : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism -/
def num_vertices (p : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (p : RectangularPrism) : ℕ := 6

/-- The sum of edges, vertices, and faces in a rectangular prism is 26 -/
theorem sum_edges_vertices_faces (p : RectangularPrism) :
  num_edges p + num_vertices p + num_faces p = 26 := by
  sorry

#check sum_edges_vertices_faces

end sum_edges_vertices_faces_l2155_215534


namespace triangle_sequence_properties_l2155_215500

/-- Isosceles triangle with perimeter 2s -/
structure IsoscelesTriangle (s : ℝ) :=
  (base : ℝ)
  (leg : ℝ)
  (perimeter_eq : base + 2 * leg = 2 * s)
  (isosceles : leg ≥ base / 2)

/-- Sequence of isosceles triangles -/
def triangle_sequence (s : ℝ) : ℕ → IsoscelesTriangle s
| 0 => ⟨2, 49, sorry, sorry⟩
| (n + 1) => ⟨(triangle_sequence s n).leg, sorry, sorry, sorry⟩

/-- Angle between the legs of triangle i -/
def angle (s : ℝ) (i : ℕ) : ℝ := sorry

theorem triangle_sequence_properties (s : ℝ) :
  (∀ j : ℕ, angle s (2 * j) < angle s (2 * (j + 1))) ∧
  (∀ j : ℕ, angle s (2 * j + 1) > angle s (2 * (j + 1) + 1)) ∧
  (abs (angle s 11 - Real.pi / 3) < Real.pi / 180) :=
sorry

end triangle_sequence_properties_l2155_215500


namespace photographer_application_choices_l2155_215510

theorem photographer_application_choices :
  let n : ℕ := 5  -- Total number of pre-selected photos
  let k₁ : ℕ := 3 -- First option for number of photos to include
  let k₂ : ℕ := 4 -- Second option for number of photos to include
  (Nat.choose n k₁) + (Nat.choose n k₂) = 15 := by
  sorry

end photographer_application_choices_l2155_215510


namespace log_equation_implies_y_value_l2155_215519

-- Define a positive real number type for the base of logarithms
def PositiveReal := {x : ℝ | x > 0}

-- Define the logarithm function
noncomputable def log (base : PositiveReal) (x : PositiveReal) : ℝ := Real.log x / Real.log base.val

-- The main theorem
theorem log_equation_implies_y_value 
  (a b c x : PositiveReal) 
  (p q r y : ℝ) 
  (base : PositiveReal)
  (h1 : log base a / p = log base b / q)
  (h2 : log base b / q = log base c / r)
  (h3 : log base c / r = log base x)
  (h4 : x.val ≠ 1)
  (h5 : b.val^2 / (a.val * c.val) = x.val^y) :
  y = 2*q - p - r := by
  sorry

end log_equation_implies_y_value_l2155_215519


namespace nadia_hannah_walk_l2155_215527

/-- The total distance walked by Nadia and Hannah -/
def total_distance (nadia_distance : ℝ) (hannah_distance : ℝ) : ℝ :=
  nadia_distance + hannah_distance

/-- Theorem: Given Nadia walked 18 km and twice as far as Hannah, their total distance is 27 km -/
theorem nadia_hannah_walk :
  let nadia_distance : ℝ := 18
  let hannah_distance : ℝ := nadia_distance / 2
  total_distance nadia_distance hannah_distance = 27 := by
sorry

end nadia_hannah_walk_l2155_215527


namespace fitness_equipment_problem_l2155_215599

/-- Unit price of type A fitness equipment -/
def unit_price_A : ℝ := 360

/-- Unit price of type B fitness equipment -/
def unit_price_B : ℝ := 540

/-- Total number of fitness equipment to be purchased -/
def total_equipment : ℕ := 50

/-- Maximum total cost allowed -/
def max_total_cost : ℝ := 21000

/-- Theorem stating the conditions and conclusions of the fitness equipment problem -/
theorem fitness_equipment_problem :
  (unit_price_B = 1.5 * unit_price_A) ∧
  (7200 / unit_price_A - 5400 / unit_price_B = 10) ∧
  (∀ x : ℕ, x ≤ total_equipment →
    unit_price_A * x + unit_price_B * (total_equipment - x) ≤ max_total_cost →
    x ≥ 34) :=
sorry

end fitness_equipment_problem_l2155_215599


namespace coefficient_x3_equals_negative_30_l2155_215565

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in the expansion of (1-2x)(1-x)^5
def coefficient_x3 : ℤ :=
  -1 * (-1) * binomial 5 3 + (-2) * binomial 5 2

-- Theorem statement
theorem coefficient_x3_equals_negative_30 : coefficient_x3 = -30 := by sorry

end coefficient_x3_equals_negative_30_l2155_215565


namespace negative_fraction_comparison_l2155_215555

theorem negative_fraction_comparison : -5/6 < -4/5 := by
  sorry

end negative_fraction_comparison_l2155_215555


namespace train_distance_difference_l2155_215521

theorem train_distance_difference (v1 v2 total_distance : ℝ) 
  (h1 : v1 = 20)
  (h2 : v2 = 25)
  (h3 : total_distance = 675) :
  let t := total_distance / (v1 + v2)
  let d1 := v1 * t
  let d2 := v2 * t
  |d2 - d1| = 75 := by sorry

end train_distance_difference_l2155_215521


namespace radish_basket_difference_l2155_215558

theorem radish_basket_difference (total : ℕ) (first_basket : ℕ) : 
  total = 88 → first_basket = 37 → total - first_basket - first_basket = 14 :=
by
  sorry

end radish_basket_difference_l2155_215558


namespace speed_ratio_walking_l2155_215563

/-- Theorem: Ratio of speeds when two people walk towards each other and in the same direction -/
theorem speed_ratio_walking (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : b > a) : ∃ (v₁ v₂ : ℝ),
  v₁ > 0 ∧ v₂ > 0 ∧ 
  (∃ (S : ℝ), S > 0 ∧ S = a * (v₁ + v₂) ∧ S = b * (v₁ - v₂)) ∧
  v₂ / v₁ = (a + b) / (b - a) :=
by sorry

end speed_ratio_walking_l2155_215563


namespace cannot_determine_heavier_l2155_215517

variable (M P O : ℝ)

def mandarin_lighter_than_pear := M < P
def orange_heavier_than_mandarin := O > M

theorem cannot_determine_heavier (h1 : mandarin_lighter_than_pear M P) 
  (h2 : orange_heavier_than_mandarin O M) : 
  ¬(∀ x y : ℝ, (x < y) ∨ (y < x)) :=
sorry

end cannot_determine_heavier_l2155_215517


namespace complex_multiplication_l2155_215535

theorem complex_multiplication (i : ℂ) : i * i = -1 → (2 + 3*i) * (3 - 2*i) = 12 + 5*i := by
  sorry

end complex_multiplication_l2155_215535


namespace inequality_implies_lower_bound_l2155_215518

theorem inequality_implies_lower_bound (a : ℝ) :
  (∀ x ∈ Set.Icc (0 : ℝ) (1/2), 4^x + x - a ≤ 3/2) → a ≥ 1 := by
  sorry

end inequality_implies_lower_bound_l2155_215518


namespace division_problem_l2155_215515

/-- Given the conditions of the division problem, prove the values of the divisors -/
theorem division_problem (D₁ D₂ : ℕ) : 
  1526 = 34 * D₁ + 18 → 
  34 * D₂ + 52 = 421 → 
  D₁ = 44 ∧ D₂ = 11 := by
  sorry

#check division_problem

end division_problem_l2155_215515


namespace max_value_of_a_l2155_215574

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem max_value_of_a :
  (∀ x : ℝ, determinant (x - 1) (a - 2) (a + 1) x ≥ 1) →
  a ≤ 3/2 ∧ ∃ a₀ : ℝ, a₀ ≤ 3/2 ∧ ∀ x : ℝ, determinant (x - 1) (a₀ - 2) (a₀ + 1) x ≥ 1 :=
sorry

end max_value_of_a_l2155_215574


namespace rabbit_population_estimate_l2155_215587

/-- Calculates the approximate number of rabbits in a forest using the capture-recapture method. -/
def estimate_rabbit_population (initial_tagged : ℕ) (recaptured : ℕ) (tagged_in_recapture : ℕ) : ℕ :=
  (initial_tagged * recaptured) / tagged_in_recapture

/-- The approximate number of rabbits in the forest is 50. -/
theorem rabbit_population_estimate :
  let initial_tagged : ℕ := 10
  let recaptured : ℕ := 10
  let tagged_in_recapture : ℕ := 2
  estimate_rabbit_population initial_tagged recaptured tagged_in_recapture = 50 := by
  sorry

#eval estimate_rabbit_population 10 10 2

end rabbit_population_estimate_l2155_215587


namespace teresa_jogging_time_l2155_215568

-- Define the constants
def distance : ℝ := 45  -- kilometers
def speed : ℝ := 7      -- kilometers per hour
def break_time : ℝ := 0.5  -- hours (30 minutes)

-- Define the theorem
theorem teresa_jogging_time :
  let jogging_time := distance / speed
  let total_time := jogging_time + break_time
  total_time = 6.93 :=
by
  sorry


end teresa_jogging_time_l2155_215568


namespace root_sum_cubes_l2155_215576

-- Define the equation
def equation (x : ℝ) : Prop := (x - (8 : ℝ)^(1/3)) * (x - (27 : ℝ)^(1/3)) * (x - (64 : ℝ)^(1/3)) = 1

-- Define the roots
def roots (u v w : ℝ) : Prop := equation u ∧ equation v ∧ equation w ∧ u ≠ v ∧ u ≠ w ∧ v ≠ w

-- Theorem statement
theorem root_sum_cubes (u v w : ℝ) : roots u v w → u^3 + v^3 + w^3 = 102 := by
  sorry

end root_sum_cubes_l2155_215576


namespace remainder_17_power_100_mod_7_l2155_215528

theorem remainder_17_power_100_mod_7 : 17^100 % 7 = 4 := by
  sorry

end remainder_17_power_100_mod_7_l2155_215528


namespace remainder_of_sum_l2155_215516

theorem remainder_of_sum (d : ℕ) (h1 : 242 % d = 8) (h2 : 698 % d = 9) (h3 : d = 13) :
  (242 + 698) % d = 4 := by
  sorry

end remainder_of_sum_l2155_215516


namespace perfect_square_condition_l2155_215577

theorem perfect_square_condition (Z K : ℤ) : 
  (50 < Z ∧ Z < 5000) →
  K > 1 →
  Z = K * K^2 →
  (∃ n : ℤ, Z = n^2) ↔ (K = 4 ∨ K = 9 ∨ K = 16) :=
by sorry

end perfect_square_condition_l2155_215577


namespace rectangle_max_area_l2155_215589

theorem rectangle_max_area (x y : ℝ) (h : x > 0 ∧ y > 0) :
  (2 * x + 2 * y = 60) → x * y ≤ 225 := by
  sorry

end rectangle_max_area_l2155_215589


namespace tomato_suggestion_count_l2155_215501

theorem tomato_suggestion_count (total students_potatoes students_bacon : ℕ) 
  (h1 : total = 826)
  (h2 : students_potatoes = 324)
  (h3 : students_bacon = 374) :
  total - (students_potatoes + students_bacon) = 128 := by
sorry

end tomato_suggestion_count_l2155_215501


namespace congruence_intercepts_sum_l2155_215593

theorem congruence_intercepts_sum (x₀ y₀ : ℕ) : 
  (0 ≤ x₀ ∧ x₀ < 40) → 
  (0 ≤ y₀ ∧ y₀ < 40) → 
  (5 * x₀ ≡ -2 [ZMOD 40]) → 
  (3 * y₀ ≡ 2 [ZMOD 40]) → 
  x₀ + y₀ = 8 := by
  sorry

end congruence_intercepts_sum_l2155_215593


namespace range_for_two_roots_roots_for_negative_integer_k_l2155_215511

/-- The quadratic equation with parameter k -/
def quadratic (k : ℝ) (x : ℝ) : ℝ :=
  x^2 + (2*k + 1)*x + k^2 - 1

/-- The discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ :=
  (2*k + 1)^2 - 4*(k^2 - 1)

/-- Theorem stating the range of k for which the equation has two distinct real roots -/
theorem range_for_two_roots :
  ∀ k : ℝ, (∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0) ↔ k > -5/4 :=
sorry

/-- Theorem stating the roots when k is a negative integer satisfying the range condition -/
theorem roots_for_negative_integer_k :
  ∀ k : ℤ, k < 0 → k > -5/4 → quadratic (↑k) 0 = 0 ∧ quadratic (↑k) 1 = 0 :=
sorry

end range_for_two_roots_roots_for_negative_integer_k_l2155_215511


namespace unique_true_proposition_l2155_215506

theorem unique_true_proposition :
  (¬ ∀ x : ℝ, x^2 + 3 < 0) ∧
  (¬ ∀ x : ℕ, x^2 ≥ 1) ∧
  (∃ x : ℤ, x^5 < 1) ∧
  (¬ ∃ x : ℚ, x^2 = 3) := by
  sorry

end unique_true_proposition_l2155_215506


namespace mother_twice_bob_age_year_l2155_215560

def bob_age_2010 : ℕ := 10
def mother_age_2010 : ℕ := 5 * bob_age_2010

def year_mother_twice_bob_age : ℕ :=
  2010 + (mother_age_2010 - 2 * bob_age_2010)

theorem mother_twice_bob_age_year :
  year_mother_twice_bob_age = 2040 := by
  sorry

end mother_twice_bob_age_year_l2155_215560


namespace complementary_angles_difference_theorem_l2155_215595

def complementary_angles_difference (a b : ℝ) : Prop :=
  a + b = 90 ∧ a / b = 5 / 3 → |a - b| = 22.5

theorem complementary_angles_difference_theorem :
  ∀ a b : ℝ, complementary_angles_difference a b :=
by sorry

end complementary_angles_difference_theorem_l2155_215595


namespace shopping_time_calculation_l2155_215549

/-- Calculates the time spent shopping, performing tasks, and traveling between sections --/
theorem shopping_time_calculation (total_trip_time waiting_times break_time browsing_times walking_time_per_trip num_sections : ℕ) :
  total_trip_time = 165 ∧
  waiting_times = 5 + 10 + 8 + 15 + 20 ∧
  break_time = 10 ∧
  browsing_times = 12 + 7 + 10 ∧
  walking_time_per_trip = 2 ∧ -- Rounded up from 1.5
  num_sections = 8 →
  total_trip_time - (waiting_times + break_time + browsing_times + walking_time_per_trip * (num_sections - 1)) = 86 :=
by sorry

end shopping_time_calculation_l2155_215549


namespace perimeter_of_square_arrangement_l2155_215547

theorem perimeter_of_square_arrangement (total_area : ℝ) (num_squares : ℕ) 
  (arrangement_width : ℕ) (arrangement_height : ℕ) :
  total_area = 216 →
  num_squares = 6 →
  arrangement_width = 3 →
  arrangement_height = 2 →
  let square_area := total_area / num_squares
  let side_length := Real.sqrt square_area
  let perimeter := 2 * (arrangement_width + arrangement_height) * side_length
  perimeter = 60 :=
by
  sorry

end perimeter_of_square_arrangement_l2155_215547


namespace max_distance_line_equation_l2155_215590

/-- The line of maximum distance from the origin passing through (2, 3) -/
def max_distance_line (x y : ℝ) : Prop :=
  2 * x + 3 * y - 13 = 0

/-- The point through which the line passes -/
def point : ℝ × ℝ := (2, 3)

/-- Theorem stating that the line of maximum distance from the origin
    passing through (2, 3) has the equation 2x + 3y - 13 = 0 -/
theorem max_distance_line_equation :
  ∀ x y : ℝ, (x, y) ∈ ({p : ℝ × ℝ | p.1 * point.2 + p.2 * point.1 = point.1 * point.2} : Set (ℝ × ℝ)) →
  max_distance_line x y :=
sorry

end max_distance_line_equation_l2155_215590


namespace power_of_81_equals_9_l2155_215583

theorem power_of_81_equals_9 : (81 : ℝ) ^ (0.25 : ℝ) * (81 : ℝ) ^ (0.20 : ℝ) = 9 := by
  sorry

end power_of_81_equals_9_l2155_215583


namespace mapping_result_l2155_215572

-- Define the set A (and B) as pairs of real numbers
def A : Type := ℝ × ℝ

-- Define the mapping f
def f (p : A) : A :=
  let (x, y) := p
  (x - y, x + y)

-- Theorem statement
theorem mapping_result : f (-1, 2) = (-3, 1) := by
  sorry

end mapping_result_l2155_215572


namespace rectangle_side_ratio_l2155_215544

/-- Represents a rectangle with side lengths x and y -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- Represents the configuration of rectangles and squares -/
structure CrossConfiguration where
  inner_square_side : ℝ
  outer_square_side : ℝ
  rectangle : Rectangle

/-- The cross configuration satisfies the given conditions -/
def valid_configuration (c : CrossConfiguration) : Prop :=
  c.outer_square_side = 3 * c.inner_square_side ∧
  c.rectangle.y = c.inner_square_side ∧
  c.rectangle.x + c.inner_square_side = c.outer_square_side

/-- The theorem stating the ratio of rectangle sides -/
theorem rectangle_side_ratio (c : CrossConfiguration) 
  (h : valid_configuration c) : 
  c.rectangle.x / c.rectangle.y = 2 := by
  sorry

end rectangle_side_ratio_l2155_215544


namespace gdp_scientific_notation_correct_l2155_215588

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The GDP value of Anning City in the first quarter of 2023 -/
def gdp_value : ℕ := 17580000000

/-- The scientific notation representation of the GDP value -/
def gdp_scientific : ScientificNotation :=
  { coefficient := 1.758
    exponent := 10
    is_valid := by sorry }

/-- Theorem stating that the GDP value is correctly represented in scientific notation -/
theorem gdp_scientific_notation_correct :
  (gdp_scientific.coefficient * (10 : ℝ) ^ gdp_scientific.exponent) = gdp_value := by sorry

end gdp_scientific_notation_correct_l2155_215588


namespace matrix_product_is_zero_l2155_215529

-- Define the matrices
def matrix1 (d e f : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, d, -e],
    ![-d, 0, f],
    ![e, -f, 0]]

def matrix2 (a b c : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![a^3, a^2*b, a^2*c],
    ![a*b^2, b^3, b^2*c],
    ![a*c^2, b*c^2, c^3]]

-- Theorem statement
theorem matrix_product_is_zero (a b c d e f : ℝ) :
  matrix1 d e f * matrix2 a b c = 0 := by
  sorry

end matrix_product_is_zero_l2155_215529


namespace probability_sum_less_than_ten_l2155_215508

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The set of possible outcomes when rolling two dice -/
def outcomes : Finset (ℕ × ℕ) := Finset.product (Finset.range sides) (Finset.range sides)

/-- The favorable outcomes (sum less than 10) -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  outcomes.filter (fun p => p.1 + p.2 < 10)

/-- The probability of the sum being less than 10 when rolling two fair six-sided dice -/
def probability : ℚ :=
  (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ)

theorem probability_sum_less_than_ten : probability = 5 / 6 := by
  sorry

end probability_sum_less_than_ten_l2155_215508


namespace intersection_slope_l2155_215557

/-- Given two lines that intersect at a point, find the slope of one line -/
theorem intersection_slope (k : ℝ) : 
  (∃ (x y : ℝ), y = 2*x + 3 ∧ y = k*x + 4 ∧ x = 1 ∧ y = 5) → k = 1 := by
  sorry

end intersection_slope_l2155_215557


namespace apple_sales_loss_percentage_l2155_215573

/-- Represents the shopkeeper's apple sales scenario -/
structure AppleSales where
  total_apples : ℝ
  sale_percentages : Fin 4 → ℝ
  profit_percentages : Fin 4 → ℝ
  unsold_percentage : ℝ
  storage_cost : ℝ
  packaging_cost : ℝ
  transportation_cost : ℝ

/-- Calculates the effective loss percentage for the given apple sales scenario -/
def effective_loss_percentage (sales : AppleSales) : ℝ :=
  sorry

/-- The given apple sales scenario -/
def given_scenario : AppleSales :=
  { total_apples := 150,
    sale_percentages := ![0.30, 0.25, 0.15, 0.10],
    profit_percentages := ![0.20, 0.30, 0.40, 0.35],
    unsold_percentage := 0.20,
    storage_cost := 15,
    packaging_cost := 10,
    transportation_cost := 25 }

/-- Theorem stating that the effective loss percentage for the given scenario is approximately 32.83% -/
theorem apple_sales_loss_percentage :
  abs (effective_loss_percentage given_scenario - 32.83) < 0.01 :=
sorry

end apple_sales_loss_percentage_l2155_215573


namespace tan_product_30_degrees_l2155_215570

theorem tan_product_30_degrees :
  let A : Real := 30 * π / 180
  let B : Real := 30 * π / 180
  (1 + Real.tan A) * (1 + Real.tan B) = (4 + 2 * Real.sqrt 3) / 3 := by
  sorry

end tan_product_30_degrees_l2155_215570


namespace right_triangle_hypotenuse_l2155_215524

/-- Given a right triangle with legs x and y, if rotating about one leg produces a cone of volume 1000π cm³
    and rotating about the other leg produces a cone of volume 2250π cm³, 
    then the hypotenuse is approximately 39.08 cm. -/
theorem right_triangle_hypotenuse (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) :
  (1/3 * π * y^2 * x = 1000 * π) →
  (1/3 * π * x^2 * y = 2250 * π) →
  abs (Real.sqrt (x^2 + y^2) - 39.08) < 0.01 := by
  sorry

end right_triangle_hypotenuse_l2155_215524


namespace square_of_negative_product_l2155_215514

theorem square_of_negative_product (a b : ℝ) : (-3 * a * b^2)^2 = 9 * a^2 * b^4 := by
  sorry

end square_of_negative_product_l2155_215514


namespace smallest_perfect_square_divisible_by_3_and_5_l2155_215553

/-- A number is a perfect square if it's equal to some integer squared. -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- A number is divisible by both 3 and 5 if it's divisible by 15. -/
def divisible_by_3_and_5 (n : ℕ) : Prop := n % 15 = 0

theorem smallest_perfect_square_divisible_by_3_and_5 :
  (∀ n : ℕ, n > 0 → is_perfect_square n → divisible_by_3_and_5 n → n ≥ 225) ∧
  (is_perfect_square 225 ∧ divisible_by_3_and_5 225) :=
sorry

end smallest_perfect_square_divisible_by_3_and_5_l2155_215553


namespace honey_work_days_l2155_215582

/-- Proves that Honey worked for 20 days given her daily earnings and total spent and saved amounts. -/
theorem honey_work_days (daily_earnings : ℕ) (total_spent : ℕ) (total_saved : ℕ) :
  daily_earnings = 80 →
  total_spent = 1360 →
  total_saved = 240 →
  (total_spent + total_saved) / daily_earnings = 20 :=
by sorry

end honey_work_days_l2155_215582


namespace total_money_value_l2155_215556

def gold_value : ℕ := 75
def silver_value : ℕ := 40
def bronze_value : ℕ := 20
def titanium_value : ℕ := 10

def gold_count : ℕ := 6
def silver_count : ℕ := 8
def bronze_count : ℕ := 10
def titanium_count : ℕ := 4

def cash : ℕ := 45

theorem total_money_value :
  gold_value * gold_count +
  silver_value * silver_count +
  bronze_value * bronze_count +
  titanium_value * titanium_count +
  cash = 1055 := by sorry

end total_money_value_l2155_215556


namespace unique_three_digit_number_l2155_215536

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def units_digit (n : ℕ) : ℕ := n % 10

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem unique_three_digit_number :
  ∃! n : ℕ, is_three_digit n ∧
            units_digit n = 4 ∧
            hundreds_digit n = 5 ∧
            tens_digit n % 2 = 0 ∧
            n % 8 = 0 ∧
            n = 544 :=
by sorry

end unique_three_digit_number_l2155_215536


namespace algebraic_expression_value_l2155_215554

theorem algebraic_expression_value (a b : ℝ) : 
  (a * 2^3 + b * 2 - 7 = -19) → 
  (a * (-2)^3 + b * (-2) - 7 = 5) := by
sorry

end algebraic_expression_value_l2155_215554


namespace product_abcd_l2155_215586

theorem product_abcd (a b c d : ℚ) : 
  (2 * a + 3 * b + 5 * c + 7 * d = 42) →
  (4 * (d + c) = b) →
  (2 * b + 2 * c = a) →
  (c - 2 = d) →
  (a * b * c * d = -26880 / 729) := by
sorry

end product_abcd_l2155_215586


namespace inverse_proportion_problem_l2155_215566

-- Define the inverse proportionality relationship
def inverse_proportional (y x : ℝ) := ∃ k : ℝ, y = k / (x + 2)

-- Define the theorem
theorem inverse_proportion_problem (y x : ℝ) 
  (h1 : inverse_proportional y x) 
  (h2 : y = 3 ∧ x = -1) :
  (∀ x, y = 3 / (x + 2)) ∧ 
  (x = 0 → y = 3/2) := by
  sorry

end inverse_proportion_problem_l2155_215566


namespace percentage_relation_l2155_215533

theorem percentage_relation (x y z : ℝ) :
  y = 0.3 * z →
  x = 0.36 * z →
  x = y * 1.2 :=
by
  sorry

end percentage_relation_l2155_215533


namespace knitting_rate_theorem_l2155_215578

/-- The number of days it takes A to knit a pair of socks -/
def A_days : ℝ := 3

/-- The number of days it takes A and B together to knit two pairs of socks -/
def AB_days : ℝ := 4

/-- The number of days it takes B to knit a pair of socks -/
def B_days : ℝ := 6

/-- Theorem stating that given A's knitting rate and the combined rate of A and B,
    B's individual knitting rate can be determined -/
theorem knitting_rate_theorem :
  (1 / A_days + 1 / B_days) * AB_days = 2 :=
sorry

end knitting_rate_theorem_l2155_215578


namespace min_value_of_M_l2155_215598

theorem min_value_of_M (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 →
    Real.sqrt (1 + 2 * x^2) + 2 * Real.sqrt ((5/12)^2 + y^2) ≥ 5 * Real.sqrt 34 / 12) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧
    Real.sqrt (1 + 2 * x^2) + 2 * Real.sqrt ((5/12)^2 + y^2) = 5 * Real.sqrt 34 / 12) :=
by sorry

end min_value_of_M_l2155_215598


namespace unique_base_conversion_l2155_215523

def base_conversion (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

theorem unique_base_conversion : ∃! x : Nat,
  x < 1000 ∧
  x ≥ 100 ∧
  let digits := [x / 100, (x / 10) % 10, x % 10]
  base_conversion digits 20 = 2 * base_conversion digits 13 :=
by
  sorry

end unique_base_conversion_l2155_215523


namespace exactly_two_in_favor_l2155_215513

def probability_in_favor : ℝ := 0.6

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem exactly_two_in_favor :
  binomial_probability 4 2 probability_in_favor = 0.3456 := by
  sorry

end exactly_two_in_favor_l2155_215513


namespace min_value_theorem_l2155_215505

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (x + 2*y)) + (y / x) ≥ Real.sqrt 2 - 1/2 := by
  sorry

end min_value_theorem_l2155_215505


namespace triangle_height_ratio_l2155_215542

theorem triangle_height_ratio (a b c h₁ h₂ h₃ : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧
  (a : ℝ) / 3 = (b : ℝ) / 4 ∧ (b : ℝ) / 4 = (c : ℝ) / 5 ∧
  a * h₁ = b * h₂ ∧ b * h₂ = c * h₃ →
  (h₁ : ℝ) / 20 = (h₂ : ℝ) / 15 ∧ (h₂ : ℝ) / 15 = (h₃ : ℝ) / 12 :=
by sorry

end triangle_height_ratio_l2155_215542


namespace garland_arrangements_correct_l2155_215530

/-- The number of ways to arrange 6 blue, 7 red, and 9 white light bulbs in a garland,
    such that no two white light bulbs are consecutive -/
def garland_arrangements : ℕ :=
  Nat.choose 13 6 * Nat.choose 14 9

/-- Theorem stating that the number of garland arrangements is correct -/
theorem garland_arrangements_correct :
  garland_arrangements = 3435432 := by sorry

end garland_arrangements_correct_l2155_215530


namespace unique_real_root_l2155_215581

theorem unique_real_root : 
  (∃ x : ℝ, x^2 + 3 = 0) = false ∧ 
  (∃ x : ℝ, x^3 + 3 = 0) = true ∧ 
  (∃ x : ℝ, |1 / (x^2 - 3)| = 0) = false ∧ 
  (∃ x : ℝ, |x| + 3 = 0) = false :=
by sorry

end unique_real_root_l2155_215581


namespace max_value_of_expression_l2155_215541

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  ∃ (max : ℝ), max = 2 ∧ ∀ (x y z : ℝ), 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 2 → x + y^3 + z^4 ≤ max :=
sorry

end max_value_of_expression_l2155_215541


namespace translate_quadratic_l2155_215520

/-- Represents a quadratic function of the form y = ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Applies a horizontal and vertical translation to a quadratic function -/
def translate (f : QuadraticFunction) (h : ℝ) (v : ℝ) : QuadraticFunction :=
  { a := f.a
  , b := -2 * f.a * h + f.b
  , c := f.a * h^2 - f.b * h + f.c + v }

theorem translate_quadratic :
  let f : QuadraticFunction := { a := 2, b := 0, c := 0 }
  let g : QuadraticFunction := translate f (-1) 3
  g = { a := 2, b := 4, c := 5 } :=
by sorry

end translate_quadratic_l2155_215520


namespace factorization_identity_l2155_215571

theorem factorization_identity (a : ℝ) : 3 * a^2 + 6 * a + 3 = 3 * (a + 1)^2 := by
  sorry

end factorization_identity_l2155_215571


namespace P_in_second_quadrant_l2155_215509

/-- Definition of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def secondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The given point P -/
def P : Point :=
  { x := -2, y := 3 }

/-- Theorem: Point P is in the second quadrant -/
theorem P_in_second_quadrant : secondQuadrant P := by
  sorry

end P_in_second_quadrant_l2155_215509


namespace largest_number_l2155_215526

theorem largest_number (a b c d : ℝ) 
  (h : a + 5 = b^2 - 1 ∧ a + 5 = c^2 + 3 ∧ a + 5 = d - 4) : 
  d > a ∧ d > b ∧ d > c := by
  sorry

end largest_number_l2155_215526


namespace triangle_abc_is_right_triangle_l2155_215552

/-- Given a triangle ABC where:
    - The sides opposite to angles A, B, C are a, b, c respectively
    - A = π/3
    - a = √3
    - b = 1
    Prove that C = π/2, i.e., the triangle is a right triangle -/
theorem triangle_abc_is_right_triangle (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 3 →
  b = 1 →
  A = π / 3 →
  a / Real.sin A = b / Real.sin B →
  A + B + C = π →
  C = π / 2 := by
  sorry

end triangle_abc_is_right_triangle_l2155_215552


namespace pauls_vertical_distance_l2155_215525

/-- The total vertical distance traveled by Paul in a week -/
def total_vertical_distance (floor : ℕ) (trips_per_day : ℕ) (days : ℕ) (story_height : ℕ) : ℕ :=
  floor * story_height * trips_per_day * 2 * days

/-- Theorem stating the total vertical distance Paul travels in a week -/
theorem pauls_vertical_distance :
  total_vertical_distance 5 3 7 10 = 2100 := by
  sorry

end pauls_vertical_distance_l2155_215525


namespace root_product_sum_l2155_215579

theorem root_product_sum (x₁ x₂ x₃ : ℝ) : 
  x₁ < x₂ ∧ x₂ < x₃ ∧ 
  (Real.sqrt 1008) * x₁^3 - 2016 * x₁^2 + 5 * x₁ + 2 = 0 ∧
  (Real.sqrt 1008) * x₂^3 - 2016 * x₂^2 + 5 * x₂ + 2 = 0 ∧
  (Real.sqrt 1008) * x₃^3 - 2016 * x₃^2 + 5 * x₃ + 2 = 0 →
  x₂ * (x₁ + x₃) = 1010 / 1008 := by
sorry

end root_product_sum_l2155_215579


namespace opposite_sign_expression_value_l2155_215575

theorem opposite_sign_expression_value (a b : ℝ) :
  (|a + 2| = 0 ∧ (b - 5/2)^2 = 0) →
  (2*a + 3*b) * (2*b - 3*a) = 26 :=
by sorry

end opposite_sign_expression_value_l2155_215575


namespace hikers_count_l2155_215546

theorem hikers_count (total : ℕ) (difference : ℕ) (hikers bike_riders : ℕ) 
  (h1 : total = hikers + bike_riders)
  (h2 : hikers = bike_riders + difference)
  (h3 : total = 676)
  (h4 : difference = 178) :
  hikers = 427 := by
  sorry

end hikers_count_l2155_215546


namespace ceiling_floor_calculation_l2155_215532

theorem ceiling_floor_calculation : ⌈(15 / 8) * (-35 / 4)⌉ - ⌊(15 / 8) * ⌊(-35 / 4) + (1 / 4)⌋⌋ = 1 := by
  sorry

end ceiling_floor_calculation_l2155_215532


namespace even_function_implies_a_value_l2155_215545

def f (x a : ℝ) : ℝ := (x + 1) * (2 * x + 3 * a)

theorem even_function_implies_a_value :
  (∀ x, f x a = f (-x) a) → a = -2/3 := by
  sorry

end even_function_implies_a_value_l2155_215545


namespace estimate_student_population_l2155_215522

theorem estimate_student_population (n : ℕ) 
  (h1 : n > 0) 
  (h2 : 80 ≤ n) 
  (h3 : 100 ≤ n) : 
  (80 : ℝ) / n * 100 = 20 → n = 400 := by
  sorry

end estimate_student_population_l2155_215522


namespace outstanding_student_allocation_schemes_l2155_215569

theorem outstanding_student_allocation_schemes :
  let total_slots : ℕ := 7
  let num_schools : ℕ := 5
  let min_slots_for_two_schools : ℕ := 2
  let remaining_slots : ℕ := total_slots - 2 * min_slots_for_two_schools
  Nat.choose (remaining_slots + num_schools - 1) (num_schools - 1) = Nat.choose total_slots (total_slots - remaining_slots) := by
  sorry

end outstanding_student_allocation_schemes_l2155_215569


namespace min_value_sum_sqrt_ratios_equality_condition_l2155_215580

theorem min_value_sum_sqrt_ratios (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.sqrt (a^2 / b^2) + Real.sqrt (b^2 / c^2) + Real.sqrt (c^2 / a^2) ≥ 3 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.sqrt (a^2 / b^2) + Real.sqrt (b^2 / c^2) + Real.sqrt (c^2 / a^2) = 3 ↔ a = b ∧ b = c :=
by sorry

end min_value_sum_sqrt_ratios_equality_condition_l2155_215580


namespace least_y_solution_l2155_215538

-- Define the function we're trying to solve
def f (y : ℝ) := y + y^2

-- State the theorem
theorem least_y_solution :
  ∃ y : ℝ, y > 2 ∧ f y = 360 ∧ ∃ ε > 0, |y - 18.79| < ε :=
sorry

end least_y_solution_l2155_215538


namespace exactly_one_positive_integer_satisfies_condition_l2155_215531

theorem exactly_one_positive_integer_satisfies_condition : 
  ∃! (n : ℕ), n > 0 ∧ 30 - 6 * n > 18 :=
by sorry

end exactly_one_positive_integer_satisfies_condition_l2155_215531


namespace integral_x_squared_plus_sin_x_l2155_215592

theorem integral_x_squared_plus_sin_x : ∫ x in (-1)..1, (x^2 + Real.sin x) = 2/3 := by
  sorry

end integral_x_squared_plus_sin_x_l2155_215592


namespace negation_of_implication_l2155_215551

theorem negation_of_implication (A B : Set α) (a b : α) :
  ¬(a ∉ A → b ∈ B) ↔ (a ∉ A ∧ b ∉ B) := by
  sorry

end negation_of_implication_l2155_215551


namespace rectangular_garden_width_l2155_215502

theorem rectangular_garden_width (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 507 →
  width = 13 := by
sorry

end rectangular_garden_width_l2155_215502


namespace parabola_and_tangent_line_l2155_215591

/-- Parabola with vertex at origin and focus on positive y-axis -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  h_focus_pos : focus.2 > 0
  h_focus_eq : focus = (0, p)

/-- Line through focus intersecting parabola -/
structure IntersectingLine (para : Parabola) where
  a : ℝ × ℝ
  b : ℝ × ℝ
  h_on_parabola_a : a.1^2 = 2 * para.p * a.2
  h_on_parabola_b : b.1^2 = 2 * para.p * b.2
  h_through_focus : ∃ t : ℝ, (1 - t) • a + t • b = para.focus

/-- Line with y-intercept 6 intersecting parabola -/
structure TangentLine (para : Parabola) where
  m : ℝ
  p : ℝ × ℝ
  q : ℝ × ℝ
  r : ℝ × ℝ
  h_on_parabola_p : p.1^2 = 2 * para.p * p.2
  h_on_parabola_q : q.1^2 = 2 * para.p * q.2
  h_on_line_p : p.2 = m * p.1 + 6
  h_on_line_q : q.2 = m * q.1 + 6
  h_r_on_directrix : r.2 = -para.p
  h_qfr_collinear : ∃ t : ℝ, (1 - t) • q + t • r = para.focus
  h_pr_tangent : (p.2 - r.2) / (p.1 - r.1) = p.1 / (2 * para.p)

theorem parabola_and_tangent_line (para : Parabola) 
  (line : IntersectingLine para) 
  (tline : TangentLine para) :
  (∀ (t : ℝ), (1 - t) • line.a + t • line.b - (0, 3) = (0, 1)) →
  (‖line.a - line.b‖ = 8) →
  (para.p = 2 ∧ (tline.m = 1/2 ∨ tline.m = -1/2)) :=
sorry

end parabola_and_tangent_line_l2155_215591
