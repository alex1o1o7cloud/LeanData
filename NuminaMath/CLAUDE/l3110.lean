import Mathlib

namespace factory_assignment_l3110_311020

-- Define the workers and machines
inductive Worker : Type
  | Dan : Worker
  | Emma : Worker
  | Fiona : Worker

inductive Machine : Type
  | A : Machine
  | B : Machine
  | C : Machine

-- Define the assignment of workers to machines
def Assignment := Worker → Machine

-- Define the conditions
def condition1 (a : Assignment) : Prop := a Worker.Emma ≠ Machine.A
def condition2 (a : Assignment) : Prop := a Worker.Dan = Machine.C
def condition3 (a : Assignment) : Prop := a Worker.Fiona = Machine.B

-- Define the correct assignment
def correct_assignment : Assignment :=
  fun w => match w with
    | Worker.Dan => Machine.C
    | Worker.Emma => Machine.A
    | Worker.Fiona => Machine.B

-- Theorem statement
theorem factory_assignment :
  ∀ (a : Assignment),
    (a Worker.Dan ≠ a Worker.Emma ∧ a Worker.Dan ≠ a Worker.Fiona ∧ a Worker.Emma ≠ a Worker.Fiona) →
    ((condition1 a ∧ ¬condition2 a ∧ ¬condition3 a) ∨
     (¬condition1 a ∧ condition2 a ∧ ¬condition3 a) ∨
     (¬condition1 a ∧ ¬condition2 a ∧ condition3 a)) →
    a = correct_assignment :=
  sorry

end factory_assignment_l3110_311020


namespace orange_cost_l3110_311024

/-- Given Alexander's shopping scenario, prove the cost of each orange. -/
theorem orange_cost (apple_price : ℝ) (apple_count : ℕ) (orange_count : ℕ) (total_spent : ℝ) :
  apple_price = 1 →
  apple_count = 5 →
  orange_count = 2 →
  total_spent = 9 →
  (total_spent - apple_price * apple_count) / orange_count = 2 := by
sorry

end orange_cost_l3110_311024


namespace factorial_10_mod_13_l3110_311032

-- Define factorial function
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

-- Statement to prove
theorem factorial_10_mod_13 : factorial 10 % 13 = 6 := by
  sorry

end factorial_10_mod_13_l3110_311032


namespace min_value_arithmetic_sequence_l3110_311047

/-- Given an arithmetic sequence {a_n} with common difference d ≠ 0,
    where a₁ = 1 and a₁, a₃, a₁₃ form a geometric sequence,
    prove that the minimum value of (2S_n + 8) / (a_n + 3) for n ∈ ℕ* is 5/2,
    where S_n is the sum of the first n terms of {a_n}. -/
theorem min_value_arithmetic_sequence (d : ℝ) (h_d : d ≠ 0) :
  let a : ℕ → ℝ := λ n => 1 + (n - 1) * d
  let S : ℕ → ℝ := λ n => (n * (2 + (n - 1) * d)) / 2
  (a 1 = 1) ∧ 
  (a 1 * a 13 = (a 3)^2) →
  (∃ (n : ℕ), n > 0 ∧ (2 * S n + 8) / (a n + 3) = 5/2) ∧
  (∀ (n : ℕ), n > 0 → (2 * S n + 8) / (a n + 3) ≥ 5/2) :=
by sorry

end min_value_arithmetic_sequence_l3110_311047


namespace sufficient_not_necessary_l3110_311048

def vector_a (x : ℝ) : ℝ × ℝ := (6, x)

theorem sufficient_not_necessary :
  ∃ (x : ℝ), (x ≠ 8 ∧ ‖vector_a x‖ = 10) ∧
  ∀ (x : ℝ), (x = 8 → ‖vector_a x‖ = 10) :=
sorry

end sufficient_not_necessary_l3110_311048


namespace first_term_value_l3110_311003

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℚ
  -- Common difference of the sequence
  d : ℚ
  -- Sum of first 40 terms is 180
  sum_first_40 : (40 : ℚ) / 2 * (2 * a + 39 * d) = 180
  -- Sum of next 40 terms (41st to 80th) is 2200
  sum_next_40 : (40 : ℚ) / 2 * (2 * (a + 40 * d) + 39 * d) = 2200
  -- 20th term is 75
  term_20 : a + 19 * d = 75

/-- The first term of the arithmetic sequence with given properties is 51.0125 -/
theorem first_term_value (seq : ArithmeticSequence) : seq.a = 51.0125 := by
  sorry

end first_term_value_l3110_311003


namespace square_product_extension_l3110_311028

theorem square_product_extension (a b : ℕ) 
  (h1 : ∃ x : ℕ, a * b = x ^ 2)
  (h2 : ∃ y : ℕ, (a + 1) * (b + 1) = y ^ 2) :
  ∃ n : ℕ, n > 1 ∧ ∃ z : ℕ, (a + n) * (b + n) = z ^ 2 := by
  sorry

end square_product_extension_l3110_311028


namespace total_hockey_games_l3110_311051

theorem total_hockey_games (attended : ℕ) (missed : ℕ) 
  (h1 : attended = 13) (h2 : missed = 18) : 
  attended + missed = 31 := by
  sorry

end total_hockey_games_l3110_311051


namespace sum_of_x_and_y_l3110_311041

theorem sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 18*x - 10*y + 22) :
  x + y = 4 + 2 * Real.sqrt 42 := by
  sorry

end sum_of_x_and_y_l3110_311041


namespace max_chocolate_bars_correct_l3110_311013

/-- The maximum number of chocolate bars Henrique could buy -/
def max_chocolate_bars : ℕ :=
  7

/-- The cost of each chocolate bar in dollars -/
def cost_per_bar : ℚ :=
  135/100

/-- The amount Henrique paid in dollars -/
def amount_paid : ℚ :=
  10

/-- Theorem stating that max_chocolate_bars is the maximum number of bars Henrique could buy -/
theorem max_chocolate_bars_correct :
  (max_chocolate_bars : ℚ) * cost_per_bar < amount_paid ∧
  ((max_chocolate_bars + 1 : ℚ) * cost_per_bar > amount_paid ∨
   amount_paid - (max_chocolate_bars : ℚ) * cost_per_bar ≥ 1) :=
by sorry

end max_chocolate_bars_correct_l3110_311013


namespace wire_cutting_problem_l3110_311093

theorem wire_cutting_problem (total_length : ℝ) (ratio : ℝ) :
  total_length = 120 →
  ratio = 7 / 13 →
  ∃ (shorter_piece longer_piece : ℝ),
    shorter_piece + longer_piece = total_length ∧
    longer_piece = ratio * shorter_piece ∧
    shorter_piece = 78 := by
  sorry

end wire_cutting_problem_l3110_311093


namespace henry_trips_problem_l3110_311044

def henry_trips (carry_capacity : ℕ) (table1_trays : ℕ) (table2_trays : ℕ) : ℕ :=
  (table1_trays + table2_trays + carry_capacity - 1) / carry_capacity

theorem henry_trips_problem : henry_trips 9 29 52 = 9 := by
  sorry

end henry_trips_problem_l3110_311044


namespace hens_not_laying_eggs_l3110_311056

theorem hens_not_laying_eggs 
  (total_chickens : ℕ)
  (roosters : ℕ)
  (eggs_per_hen : ℕ)
  (total_eggs : ℕ)
  (h1 : total_chickens = 440)
  (h2 : roosters = 39)
  (h3 : eggs_per_hen = 3)
  (h4 : total_eggs = 1158) :
  total_chickens - roosters - (total_eggs / eggs_per_hen) = 15 :=
by
  sorry

end hens_not_laying_eggs_l3110_311056


namespace fourth_week_sugar_l3110_311035

def sugar_reduction (initial_amount : ℚ) (weeks : ℕ) : ℚ :=
  initial_amount / (2 ^ weeks)

theorem fourth_week_sugar : sugar_reduction 24 3 = 3 := by
  sorry

end fourth_week_sugar_l3110_311035


namespace triangle_side_length_l3110_311070

theorem triangle_side_length 
  (A B C : Real) -- Angles of the triangle
  (a b c : Real) -- Side lengths of the triangle
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) -- Side lengths are positive
  (h2 : B = Real.pi / 3) -- B = 60°
  (h3 : (1/2) * a * c * Real.sin B = Real.sqrt 3) -- Area of the triangle is √3
  (h4 : a^2 + c^2 = 3*a*c) -- Given equation
  : b = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_length_l3110_311070


namespace work_completion_time_l3110_311069

theorem work_completion_time (a b c : ℝ) : 
  (a > 0) →
  (b > 0) →
  (c > 0) →
  (a = 1/6) →
  (a + b + c = 1/3) →
  (c = 1/8 * (a + b)) →
  (b = 1/28) :=
sorry

end work_completion_time_l3110_311069


namespace ada_original_seat_l3110_311016

-- Define the seat numbers
inductive Seat
| one
| two
| three
| four
| five

-- Define the friends
inductive Friend
| Ada
| Bea
| Ceci
| Dee
| Edie

-- Define the seating arrangement as a function from Friend to Seat
def Seating := Friend → Seat

-- Define the movement function
def move (s : Seat) (n : Int) : Seat :=
  match s, n with
  | Seat.one, 1 => Seat.two
  | Seat.one, 2 => Seat.three
  | Seat.two, -1 => Seat.one
  | Seat.two, 1 => Seat.three
  | Seat.two, 2 => Seat.four
  | Seat.three, -1 => Seat.two
  | Seat.three, 1 => Seat.four
  | Seat.three, 2 => Seat.five
  | Seat.four, -1 => Seat.three
  | Seat.four, 1 => Seat.five
  | Seat.five, -1 => Seat.four
  | _, _ => s  -- Default case: no movement

-- Define the theorem
theorem ada_original_seat (initial_seating final_seating : Seating) :
  (∀ f : Friend, f ≠ Friend.Ada → 
    (f = Friend.Bea → move (initial_seating f) 2 = final_seating f) ∧
    (f = Friend.Ceci → move (initial_seating f) (-1) = final_seating f) ∧
    ((f = Friend.Dee ∨ f = Friend.Edie) → 
      (initial_seating Friend.Dee = final_seating Friend.Edie ∧
       initial_seating Friend.Edie = final_seating Friend.Dee))) →
  (final_seating Friend.Ada = Seat.one ∨ final_seating Friend.Ada = Seat.five) →
  initial_seating Friend.Ada = Seat.two :=
sorry

end ada_original_seat_l3110_311016


namespace parabola_equations_l3110_311040

/-- Parabola with x-axis symmetry -/
def parabola_x_axis (m : ℝ) (x y : ℝ) : Prop :=
  y^2 = m * x

/-- Parabola with y-axis symmetry -/
def parabola_y_axis (p : ℝ) (x y : ℝ) : Prop :=
  x^2 = 4 * p * y

theorem parabola_equations :
  (∃ m : ℝ, m ≠ 0 ∧ parabola_x_axis m 6 (-3)) ∧
  (∃ p : ℝ, p > 0 ∧ parabola_y_axis p x y ↔ x^2 = 12 * y ∨ x^2 = -12 * y) :=
by sorry

end parabola_equations_l3110_311040


namespace non_student_ticket_cost_l3110_311045

theorem non_student_ticket_cost
  (total_tickets : ℕ)
  (student_ticket_cost : ℚ)
  (total_amount : ℚ)
  (student_tickets : ℕ)
  (h1 : total_tickets = 193)
  (h2 : student_ticket_cost = 1/2)
  (h3 : total_amount = 412/2)
  (h4 : student_tickets = 83) :
  (total_amount - student_ticket_cost * student_tickets) / (total_tickets - student_tickets) = 3/2 := by
sorry

end non_student_ticket_cost_l3110_311045


namespace no_equal_partition_for_2002_l3110_311038

theorem no_equal_partition_for_2002 :
  ¬ ∃ (S : Finset ℕ),
    S ⊆ Finset.range 2003 ∧
    S.sum id = ((Finset.range 2003).sum id) / 2 :=
by sorry

end no_equal_partition_for_2002_l3110_311038


namespace complex_modulus_l3110_311087

theorem complex_modulus (z : ℂ) (h : z * (1 + Complex.I) = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_l3110_311087


namespace largest_inscribed_circle_diameter_squared_l3110_311099

/-- An equiangular hexagon with specified side lengths -/
structure EquiangularHexagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  equiangular : True  -- We're not proving this property, just stating it

/-- The diameter of the largest inscribed circle in an equiangular hexagon -/
def largest_inscribed_circle_diameter (h : EquiangularHexagon) : ℝ :=
  sorry  -- Definition not provided, as it's part of what needs to be proved

/-- Theorem: The square of the diameter of the largest inscribed circle in the given hexagon is 147 -/
theorem largest_inscribed_circle_diameter_squared (h : EquiangularHexagon)
  (h_AB : h.AB = 6)
  (h_BC : h.BC = 8)
  (h_CD : h.CD = 10)
  (h_DE : h.DE = 12) :
  (largest_inscribed_circle_diameter h)^2 = 147 :=
by sorry

end largest_inscribed_circle_diameter_squared_l3110_311099


namespace general_formula_minimize_s_l3110_311036

-- Define the sequence and its sum
def s (n : ℕ) : ℤ := 2 * n^2 - 30 * n

-- Define the general term of the sequence
def a (n : ℕ) : ℤ := 4 * n - 32

-- Theorem 1: The general formula for a_n is 4n - 32
theorem general_formula : ∀ n : ℕ, a n = s n - s (n - 1) := by sorry

-- Theorem 2: s_n is minimized when n = 7 or n = 8
theorem minimize_s : ∃ n : ℕ, (n = 7 ∨ n = 8) ∧ ∀ m : ℕ, s n ≤ s m := by sorry

end general_formula_minimize_s_l3110_311036


namespace milford_future_age_l3110_311075

/-- Proves that Milford's age in 3 years will be 21, given the conditions about Eustace's age. -/
theorem milford_future_age :
  ∀ (eustace_age milford_age : ℕ),
  eustace_age = 2 * milford_age →
  eustace_age + 3 = 39 →
  milford_age + 3 = 21 := by
sorry

end milford_future_age_l3110_311075


namespace integral_x_power_five_minus_one_to_one_equals_zero_l3110_311017

theorem integral_x_power_five_minus_one_to_one_equals_zero :
  ∫ x in (-1)..1, x^5 = 0 := by sorry

end integral_x_power_five_minus_one_to_one_equals_zero_l3110_311017


namespace base_10_to_9_conversion_l3110_311057

-- Define a custom type for base-9 digits
inductive Base9Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | H

def base9ToNat : List Base9Digit → Nat
  | [] => 0
  | d::ds => 
    match d with
    | Base9Digit.D0 => 0 + 9 * base9ToNat ds
    | Base9Digit.D1 => 1 + 9 * base9ToNat ds
    | Base9Digit.D2 => 2 + 9 * base9ToNat ds
    | Base9Digit.D3 => 3 + 9 * base9ToNat ds
    | Base9Digit.D4 => 4 + 9 * base9ToNat ds
    | Base9Digit.D5 => 5 + 9 * base9ToNat ds
    | Base9Digit.D6 => 6 + 9 * base9ToNat ds
    | Base9Digit.D7 => 7 + 9 * base9ToNat ds
    | Base9Digit.D8 => 8 + 9 * base9ToNat ds
    | Base9Digit.H => 8 + 9 * base9ToNat ds

theorem base_10_to_9_conversion :
  base9ToNat [Base9Digit.D3, Base9Digit.D1, Base9Digit.D4] = 256 := by
  sorry

end base_10_to_9_conversion_l3110_311057


namespace hexagon_tile_difference_l3110_311064

/-- Given a hexagonal figure with initial red and yellow tiles, prove the difference
    between yellow and red tiles after adding a border of yellow tiles. -/
theorem hexagon_tile_difference (initial_red : ℕ) (initial_yellow : ℕ) 
    (sides : ℕ) (tiles_per_side : ℕ) :
  initial_red = 15 →
  initial_yellow = 9 →
  sides = 6 →
  tiles_per_side = 4 →
  let new_yellow := initial_yellow + sides * tiles_per_side
  new_yellow - initial_red = 18 := by
sorry

end hexagon_tile_difference_l3110_311064


namespace sum_of_expressions_l3110_311021

def replace_asterisks (n : ℕ) : ℕ := 2^(n-1)

theorem sum_of_expressions : 
  (replace_asterisks 6) = 32 :=
sorry

end sum_of_expressions_l3110_311021


namespace circle_radius_difference_l3110_311026

-- Define the circles and points
def larger_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 13^2}
def smaller_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9^2}
def P : ℝ × ℝ := (5, 12)
def S (k : ℝ) : ℝ × ℝ := (0, k)

-- State the theorem
theorem circle_radius_difference (k : ℝ) : 
  P ∈ larger_circle ∧ 
  S k ∈ smaller_circle ∧
  (13 : ℝ) - 9 = 4 →
  k = 9 := by sorry

end circle_radius_difference_l3110_311026


namespace comparison_sqrt_l3110_311023

theorem comparison_sqrt : 3 * Real.sqrt 2 > Real.sqrt 13 := by
  sorry

end comparison_sqrt_l3110_311023


namespace parabola_translation_l3110_311033

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The standard parabola y = x^2 -/
def standard_parabola : Parabola := ⟨1, 0, 0⟩

/-- The final parabola y = (x+4)^2 - 5 -/
def final_parabola : Parabola := ⟨1, 8, 11⟩

/-- Translate a parabola horizontally -/
def translate_horizontal (p : Parabola) (d : ℝ) : Parabola :=
  ⟨p.a, p.b - 2 * p.a * d, p.a * d^2 + p.b * d + p.c⟩

/-- Translate a parabola vertically -/
def translate_vertical (p : Parabola) (d : ℝ) : Parabola :=
  ⟨p.a, p.b, p.c + d⟩

/-- Theorem: The final parabola can be obtained by translating the standard parabola
    4 units to the left and then 5 units downward -/
theorem parabola_translation :
  translate_vertical (translate_horizontal standard_parabola (-4)) (-5) = final_parabola := by
  sorry

end parabola_translation_l3110_311033


namespace fh_length_squared_value_l3110_311001

/-- Represents a parallelogram EFGH with specific properties -/
structure Parallelogram where
  /-- Area of the parallelogram -/
  area : ℝ
  /-- Length of JK, where J and K are projections of E and G onto FH -/
  jk_length : ℝ
  /-- Length of LM, where L and M are projections of F and H onto EG -/
  lm_length : ℝ
  /-- Assertion that EG is √2 times shorter than FH -/
  eg_fh_ratio : ℝ

/-- The square of the length of FH in the parallelogram -/
def fh_length_squared (p : Parallelogram) : ℝ := sorry

/-- Theorem stating the square of FH's length given specific conditions -/
theorem fh_length_squared_value (p : Parallelogram) 
  (h_area : p.area = 20)
  (h_jk : p.jk_length = 7)
  (h_lm : p.lm_length = 9)
  (h_ratio : p.eg_fh_ratio = Real.sqrt 2) :
  fh_length_squared p = 27.625 := by sorry

end fh_length_squared_value_l3110_311001


namespace brenda_spay_count_l3110_311050

/-- The number of cats Brenda needs to spay -/
def num_cats : ℕ := 7

/-- The number of dogs Brenda needs to spay -/
def num_dogs : ℕ := 2 * num_cats

/-- The total number of animals Brenda needs to spay -/
def total_animals : ℕ := num_cats + num_dogs

theorem brenda_spay_count : total_animals = 21 := by
  sorry

end brenda_spay_count_l3110_311050


namespace book_selling_price_l3110_311077

theorem book_selling_price (cost_price : ℚ) 
  (h1 : cost_price * (1 + 1/10) = 550) 
  (h2 : ∃ original_price : ℚ, original_price = cost_price * (1 - 1/10)) : 
  ∃ original_price : ℚ, original_price = 450 := by
sorry

end book_selling_price_l3110_311077


namespace vector_sum_magnitude_l3110_311085

/-- The angle between two vectors in radians -/
def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem vector_sum_magnitude (a b : ℝ × ℝ) 
  (h1 : angle_between a b = π / 3)  -- 60 degrees in radians
  (h2 : a = (1, Real.sqrt 3))
  (h3 : Real.sqrt (b.1^2 + b.2^2) = 1) :  -- |b| = 1
  Real.sqrt (((a.1 + 2*b.1)^2) + ((a.2 + 2*b.2)^2)) = 2 * Real.sqrt 3 := by
  sorry

end vector_sum_magnitude_l3110_311085


namespace definite_integral_problem_l3110_311059

theorem definite_integral_problem (f : ℝ → ℝ) :
  (∫ x in (π/4)..(π/2), (x * Real.cos x + Real.sin x) / (x * Real.sin x)^2) = (4 * Real.sqrt 2 - 2) / π := by
  sorry

end definite_integral_problem_l3110_311059


namespace least_positive_t_for_geometric_progression_l3110_311071

open Real

theorem least_positive_t_for_geometric_progression : 
  ∃ (t : ℝ), t > 0 ∧ 
  (∀ (α : ℝ), 0 < α → α < π/2 → 
    (∃ (r : ℝ), r > 0 ∧
      arcsin (sin α) * r = arcsin (sin (2*α)) ∧
      arcsin (sin (2*α)) * r = arcsin (sin (5*α)) ∧
      arcsin (sin (5*α)) * r = arcsin (sin (t*α)))) ∧
  (∀ (t' : ℝ), 0 < t' → t' < t →
    ¬(∀ (α : ℝ), 0 < α → α < π/2 → 
      (∃ (r : ℝ), r > 0 ∧
        arcsin (sin α) * r = arcsin (sin (2*α)) ∧
        arcsin (sin (2*α)) * r = arcsin (sin (5*α)) ∧
        arcsin (sin (5*α)) * r = arcsin (sin (t'*α))))) ∧
  t = 8 :=
by sorry

end least_positive_t_for_geometric_progression_l3110_311071


namespace passengers_remaining_approx_40_l3110_311092

/-- Calculates the number of passengers remaining after three stops -/
def passengers_after_stops (initial : ℕ) : ℚ :=
  let after_first := initial - (initial / 3)
  let after_second := after_first - (after_first / 4)
  let after_third := after_second - (after_second / 5)
  after_third

/-- Theorem: Given 100 initial passengers and three stops with specified fractions of passengers getting off, 
    the number of remaining passengers is approximately 40 -/
theorem passengers_remaining_approx_40 :
  ∃ ε > 0, ε < 1 ∧ |passengers_after_stops 100 - 40| < ε :=
sorry

end passengers_remaining_approx_40_l3110_311092


namespace optimal_ticket_price_l3110_311004

/-- Represents the net income function for the cinema --/
def net_income (x : ℕ) : ℝ :=
  if x ≤ 10 then 100 * x - 575
  else -3 * x^2 + 130 * x - 575

/-- The domain of valid ticket prices --/
def valid_price (x : ℕ) : Prop :=
  6 ≤ x ∧ x ≤ 38

theorem optimal_ticket_price :
  ∀ x : ℕ, valid_price x → net_income x ≤ net_income 22 :=
sorry

end optimal_ticket_price_l3110_311004


namespace semicircle_area_problem_l3110_311008

/-- The area of the shaded region in the semicircle problem -/
theorem semicircle_area_problem (A B C D E F G : ℝ) : 
  A < B ∧ B < C ∧ C < D ∧ D < E ∧ E < F ∧ F < G →
  B - A = 3 →
  C - B = 3 →
  D - C = 3 →
  E - D = 3 →
  F - E = 3 →
  G - F = 6 →
  let semicircle_area (d : ℝ) := π * d^2 / 8
  let total_small_area := semicircle_area (B - A) + semicircle_area (C - B) + 
                          semicircle_area (D - C) + semicircle_area (E - D) + 
                          semicircle_area (F - E) + semicircle_area (G - F)
  let large_semicircle_area := semicircle_area (G - A)
  large_semicircle_area - total_small_area = 225 * π / 8 := by
  sorry

end semicircle_area_problem_l3110_311008


namespace class_gender_ratio_l3110_311088

theorem class_gender_ratio :
  ∀ (girls boys : ℕ),
  girls + boys = 28 →
  girls = boys + 4 →
  (girls : ℚ) / (boys : ℚ) = 4 / 3 := by
  sorry

end class_gender_ratio_l3110_311088


namespace cos_ninety_degrees_equals_zero_l3110_311062

theorem cos_ninety_degrees_equals_zero : Real.cos (π / 2) = 0 := by
  sorry

end cos_ninety_degrees_equals_zero_l3110_311062


namespace f_never_prime_l3110_311058

def f (n : ℕ+) : ℕ := n^4 + 100 * n^2 + 169

theorem f_never_prime : ∀ n : ℕ+, ¬ Nat.Prime (f n) := by
  sorry

end f_never_prime_l3110_311058


namespace mike_washed_nine_cars_l3110_311009

/-- Time in minutes to wash one car -/
def wash_time : ℕ := 10

/-- Time in minutes to change oil on one car -/
def oil_change_time : ℕ := 15

/-- Time in minutes to change one set of tires -/
def tire_change_time : ℕ := 30

/-- Number of cars Mike changed oil on -/
def oil_changes : ℕ := 6

/-- Number of sets of tires Mike changed -/
def tire_changes : ℕ := 2

/-- Total time Mike worked in minutes -/
def total_work_time : ℕ := 4 * 60

/-- Function to calculate the number of cars Mike washed -/
def cars_washed : ℕ :=
  (total_work_time - (oil_changes * oil_change_time + tire_changes * tire_change_time)) / wash_time

/-- Theorem stating that Mike washed 9 cars -/
theorem mike_washed_nine_cars : cars_washed = 9 := by
  sorry

end mike_washed_nine_cars_l3110_311009


namespace light_travel_distance_l3110_311098

/-- The distance light travels in one year in kilometers -/
def light_year : ℝ := 9460000000000

/-- The number of years we're considering -/
def years : ℕ := 120

/-- Theorem stating the distance light travels in 120 years -/
theorem light_travel_distance :
  light_year * years = 1.1352e15 := by
  sorry

end light_travel_distance_l3110_311098


namespace distance_between_points_on_parabola_l3110_311042

/-- The distance between two points on a parabola -/
theorem distance_between_points_on_parabola
  (a b c x₁ x₂ : ℝ) :
  let y₁ := a * x₁^2 + b * x₁ + c
  let y₂ := a * x₂^2 + b * x₂ + c
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = |x₂ - x₁| * Real.sqrt (1 + (a * (x₂ + x₁) + b)^2) :=
by sorry

end distance_between_points_on_parabola_l3110_311042


namespace angle_sum_around_point_l3110_311074

theorem angle_sum_around_point (y : ℝ) : 
  6 * y + 7 * y + 3 * y + 2 * y = 360 → y = 20 := by
  sorry

end angle_sum_around_point_l3110_311074


namespace recipe_total_is_24_l3110_311096

/-- The total cups of ingredients required for Mary's cake recipe -/
def total_ingredients (sugar flour cocoa : ℕ) : ℕ :=
  sugar + flour + cocoa

/-- Theorem stating that the total ingredients for the recipe is 24 cups -/
theorem recipe_total_is_24 :
  total_ingredients 11 8 5 = 24 := by
  sorry

end recipe_total_is_24_l3110_311096


namespace angle_conversion_l3110_311086

def angle : Real := 54.12

theorem angle_conversion (ε : Real) (h : ε > 0) :
  ∃ (d : ℕ) (m : ℕ) (s : ℕ),
    d = 54 ∧ m = 7 ∧ s = 12 ∧ 
    abs (angle - (d : Real) - (m : Real) / 60 - (s : Real) / 3600) < ε :=
by sorry

end angle_conversion_l3110_311086


namespace product_sum_bounds_l3110_311046

def X : Finset ℕ := Finset.range 11

theorem product_sum_bounds (A B : Finset ℕ) (hA : A ⊆ X) (hB : B ⊆ X) 
  (hAB : A ∪ B = X) (hAnonempty : A.Nonempty) (hBnonempty : B.Nonempty) :
  12636 ≤ (A.prod id) + (B.prod id) ∧ (A.prod id) + (B.prod id) ≤ 2 * Nat.factorial 11 := by
  sorry

end product_sum_bounds_l3110_311046


namespace min_distance_complex_points_l3110_311083

theorem min_distance_complex_points (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min : ℝ), min = 3 ∧ ∀ w : ℂ, Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 2 - 2*I) ≥ min :=
sorry

end min_distance_complex_points_l3110_311083


namespace abs_two_implies_plus_minus_two_l3110_311012

theorem abs_two_implies_plus_minus_two (a : ℝ) : |a| = 2 → a = 2 ∨ a = -2 := by
  sorry

end abs_two_implies_plus_minus_two_l3110_311012


namespace quadratic_roots_range_l3110_311073

/-- Given a quadratic equation x^2 - x + 1 - m = 0 with two real roots α and β 
    satisfying |α| + |β| ≤ 5, the range of m is [3/4, 7]. -/
theorem quadratic_roots_range (m : ℝ) (α β : ℝ) : 
  (∀ x, x^2 - x + 1 - m = 0 ↔ x = α ∨ x = β) →
  (|α| + |β| ≤ 5) →
  (3/4 ≤ m ∧ m ≤ 7) :=
sorry

end quadratic_roots_range_l3110_311073


namespace aaron_sheep_count_l3110_311053

theorem aaron_sheep_count (beth_sheep : ℕ) (aaron_sheep : ℕ) : 
  aaron_sheep = 7 * beth_sheep →
  aaron_sheep + beth_sheep = 608 →
  aaron_sheep = 532 := by
sorry

end aaron_sheep_count_l3110_311053


namespace cube_root_of_y_fourth_root_of_y_to_six_l3110_311067

theorem cube_root_of_y_fourth_root_of_y_to_six (y : ℝ) :
  (y * (y^6)^(1/4))^(1/3) = 5 → y = 5^(6/5) := by
  sorry

end cube_root_of_y_fourth_root_of_y_to_six_l3110_311067


namespace max_trig_ratio_max_trig_ratio_equals_one_l3110_311052

theorem max_trig_ratio (x : ℝ) : 
  (Real.sin x)^4 + (Real.cos x)^4 + 2 ≤ (Real.sin x)^2 + (Real.cos x)^2 + 2 := by
  sorry

theorem max_trig_ratio_equals_one :
  ∃ x : ℝ, (Real.sin x)^4 + (Real.cos x)^4 + 2 = (Real.sin x)^2 + (Real.cos x)^2 + 2 := by
  sorry

end max_trig_ratio_max_trig_ratio_equals_one_l3110_311052


namespace quartet_songs_theorem_l3110_311068

theorem quartet_songs_theorem (a b c d e : ℕ) 
  (h1 : (a + b + c + d + e) % 4 = 0)
  (h2 : e = 8)
  (h3 : a = 5)
  (h4 : b > 5 ∧ b < 8)
  (h5 : c > 5 ∧ c < 8)
  (h6 : d > 5 ∧ d < 8) :
  (a + b + c + d + e) / 4 = 8 := by
sorry

end quartet_songs_theorem_l3110_311068


namespace classroom_tables_l3110_311030

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7 + (n % 10)

/-- The number of students in base 7 notation -/
def studentsBase7 : Nat := 321

/-- The number of students per table -/
def studentsPerTable : Nat := 3

/-- Theorem: The number of tables in the classroom is 54 -/
theorem classroom_tables :
  (base7ToBase10 studentsBase7) / studentsPerTable = 54 := by
  sorry


end classroom_tables_l3110_311030


namespace largest_office_number_l3110_311027

def house_number : List Nat := [9, 0, 2, 3, 4]

def office_number_sum : Nat := house_number.sum

def is_valid_office_number (n : List Nat) : Prop :=
  n.length = 10 ∧ n.sum = office_number_sum

def lexicographically_greater (a b : List Nat) : Prop :=
  ∃ i, (∀ j < i, a.get! j = b.get! j) ∧ a.get! i > b.get! i

theorem largest_office_number :
  ∃ (max : List Nat),
    is_valid_office_number max ∧
    (∀ n, is_valid_office_number n → lexicographically_greater max n ∨ max = n) ∧
    max = [9, 0, 5, 4, 0, 0, 0, 0, 0, 4] :=
sorry

end largest_office_number_l3110_311027


namespace program_arrangement_count_l3110_311078

def num_singing_programs : ℕ := 4
def num_skit_programs : ℕ := 2
def num_singing_between_skits : ℕ := 3

def arrange_programs : ℕ := sorry

theorem program_arrangement_count :
  arrange_programs = 96 := by sorry

end program_arrangement_count_l3110_311078


namespace paradise_park_ferris_wheel_seats_l3110_311061

/-- The number of seats on a Ferris wheel -/
def ferris_wheel_seats (total_people : ℕ) (people_per_seat : ℕ) : ℕ :=
  total_people / people_per_seat

/-- Theorem: The Ferris wheel in paradise park has 4 seats -/
theorem paradise_park_ferris_wheel_seats :
  ferris_wheel_seats 20 5 = 4 := by
  sorry

end paradise_park_ferris_wheel_seats_l3110_311061


namespace event_probability_range_l3110_311081

/-- The probability of event A occurring in a single trial -/
def p : ℝ := sorry

/-- The number of independent trials -/
def n : ℕ := 4

/-- The probability of event A occurring exactly k times in n trials -/
def prob_k (k : ℕ) : ℝ := sorry

theorem event_probability_range :
  (0 ≤ p ∧ p ≤ 1) →  -- Probability is between 0 and 1
  (prob_k 1 ≤ prob_k 2) →  -- Probability of occurring once ≤ probability of occurring twice
  (2/5 ≤ p ∧ p ≤ 1) :=  -- The range of probability p is [2/5, 1]
sorry

end event_probability_range_l3110_311081


namespace group_b_sample_size_l3110_311034

/-- Calculates the number of cities to be sampled from a group in stratified sampling -/
def stratified_sample_size (total_cities : ℕ) (group_cities : ℕ) (sample_size : ℕ) : ℕ :=
  (group_cities * sample_size) / total_cities

/-- Proves that the number of cities to be sampled from Group B is 4 -/
theorem group_b_sample_size :
  let total_cities : ℕ := 36
  let group_b_cities : ℕ := 12
  let sample_size : ℕ := 12
  stratified_sample_size total_cities group_b_cities sample_size = 4 := by
  sorry

#eval stratified_sample_size 36 12 12

end group_b_sample_size_l3110_311034


namespace odd_solution_exists_l3110_311084

theorem odd_solution_exists (k m n : ℕ+) (h : m * n = k^2 + k + 3) :
  (∃ (x y : ℤ), x^2 + 11 * y^2 = 4 * m ∧ x % 2 ≠ 0 ∧ y % 2 ≠ 0) ∨
  (∃ (x y : ℤ), x^2 + 11 * y^2 = 4 * n ∧ x % 2 ≠ 0 ∧ y % 2 ≠ 0) :=
sorry

end odd_solution_exists_l3110_311084


namespace device_records_720_instances_l3110_311029

/-- Represents the number of instances recorded by a device in one hour -/
def instances_recorded (seconds_per_record : ℕ) : ℕ :=
  (60 * 60) / seconds_per_record

/-- Theorem stating that a device recording every 5 seconds for one hour will record 720 instances -/
theorem device_records_720_instances :
  instances_recorded 5 = 720 := by
  sorry

end device_records_720_instances_l3110_311029


namespace splittable_point_range_l3110_311022

/-- A function f is splittable at x_0 if f(x_0 + 1) = f(x_0) + f(1) -/
def IsSplittable (f : ℝ → ℝ) (x_0 : ℝ) : Prop :=
  f (x_0 + 1) = f x_0 + f 1

/-- The logarithm function with base 5 -/
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

/-- The function f(x) = log_5(a / (2^x + 1)) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  log5 (a / (2^x + 1))

theorem splittable_point_range (a : ℝ) :
  (a > 0) → (∃ x_0 : ℝ, IsSplittable (f a) x_0) ↔ (3/2 < a ∧ a < 3) := by
  sorry


end splittable_point_range_l3110_311022


namespace unique_number_l3110_311006

/-- A six-digit number with leftmost digit 7 -/
def SixDigitNumber (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ n / 100000 = 7

/-- Function to move the leftmost digit to the end -/
def moveLeftmostToEnd (n : ℕ) : ℕ :=
  (n % 100000) * 10 + (n / 100000)

/-- The main theorem -/
theorem unique_number : ∃! n : ℕ, SixDigitNumber n ∧ moveLeftmostToEnd n = n / 5 :=
  sorry

end unique_number_l3110_311006


namespace palindrome_expansion_existence_l3110_311049

theorem palindrome_expansion_existence (x y k : ℕ+) : 
  ∃ (N : ℕ+) (b : Fin (k + 1) → ℕ+),
    (∀ i : Fin (k + 1), ∃ (a c : ℕ), 
      N = a * (b i)^2 + c * (b i) + a ∧ 
      a < b i ∧ 
      c < b i) ∧
    (∃ (B : ℕ+), 
      N = x * (B^2 + 1) + y * B ∧ 
      b 0 = B) :=
sorry

end palindrome_expansion_existence_l3110_311049


namespace barney_average_speed_l3110_311072

def initial_reading : ℕ := 2332
def final_reading : ℕ := 2772
def total_time : ℕ := 12

def distance : ℕ := final_reading - initial_reading

def average_speed : ℚ := distance / total_time

theorem barney_average_speed : 
  initial_reading = 2332 → 
  final_reading = 2772 → 
  total_time = 12 → 
  ⌊average_speed⌋ = 36 := by sorry

end barney_average_speed_l3110_311072


namespace horner_v3_value_l3110_311091

/-- Horner's Method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 12 + 35x - 8x^2 + 79x^3 + 6x^4 + 5x^5 + 3x^6 -/
def f : List ℝ := [12, 35, -8, 79, 6, 5, 3]

/-- The x-value at which to evaluate the polynomial -/
def x : ℝ := -4

/-- Theorem: The value of v3 in Horner's Method for f(x) at x = -4 is -57 -/
theorem horner_v3_value : 
  let v0 := f.reverse.head!
  let v1 := v0 * x + f.reverse.tail!.head!
  let v2 := v1 * x + f.reverse.tail!.tail!.head!
  let v3 := v2 * x + f.reverse.tail!.tail!.tail!.head!
  v3 = -57 := by sorry

end horner_v3_value_l3110_311091


namespace square_adjustment_theorem_l3110_311000

theorem square_adjustment_theorem (a b : ℤ) (k : ℤ) : 
  (∃ (b : ℤ), b^2 = a^2 + 2*k ∨ b^2 = a^2 - 2*k) → 
  (∃ (c : ℤ), a^2 + k = c^2 + (b-a)^2 ∨ a^2 - k = c^2 + (b-a)^2) :=
sorry

end square_adjustment_theorem_l3110_311000


namespace middle_number_problem_l3110_311043

theorem middle_number_problem :
  ∃! n : ℕ, 
    (n - 1)^2 + n^2 + (n + 1)^2 = 2030 ∧
    7 ∣ (n^3 - n^2) ∧
    n = 26 := by
  sorry

end middle_number_problem_l3110_311043


namespace arithmetic_sequence_problem_l3110_311065

/-- Given an arithmetic sequence {a_n}, if a₂ + 4a₇ + a₁₂ = 96, then 2a₃ + a₁₅ = 48 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 2 + 4 * a 7 + a 12 = 96 →                       -- given condition
  2 * a 3 + a 15 = 48 := by
sorry

end arithmetic_sequence_problem_l3110_311065


namespace number_order_l3110_311037

/-- Convert a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- Definition of 85 in base 9 --/
def num_85_base9 : Nat := to_decimal [5, 8] 9

/-- Definition of 210 in base 6 --/
def num_210_base6 : Nat := to_decimal [0, 1, 2] 6

/-- Definition of 1000 in base 4 --/
def num_1000_base4 : Nat := to_decimal [0, 0, 0, 1] 4

/-- Definition of 111111 in base 2 --/
def num_111111_base2 : Nat := to_decimal [1, 1, 1, 1, 1, 1] 2

/-- Theorem stating the order of the numbers --/
theorem number_order :
  num_210_base6 > num_85_base9 ∧
  num_85_base9 > num_1000_base4 ∧
  num_1000_base4 > num_111111_base2 :=
by sorry

end number_order_l3110_311037


namespace sarahs_earnings_proof_l3110_311097

/-- Sarah's earnings for an 8-hour day, given Connor's hourly wage and their wage ratio -/
def sarahs_daily_earnings (connors_hourly_wage : ℝ) (wage_ratio : ℝ) (hours_worked : ℝ) : ℝ :=
  connors_hourly_wage * wage_ratio * hours_worked

/-- Theorem stating Sarah's earnings for an 8-hour day -/
theorem sarahs_earnings_proof (connors_hourly_wage : ℝ) (wage_ratio : ℝ) (hours_worked : ℝ)
    (h1 : connors_hourly_wage = 7.20)
    (h2 : wage_ratio = 5)
    (h3 : hours_worked = 8) :
    sarahs_daily_earnings connors_hourly_wage wage_ratio hours_worked = 288 := by
  sorry

#eval sarahs_daily_earnings 7.20 5 8

end sarahs_earnings_proof_l3110_311097


namespace nearest_integer_to_two_plus_sqrt_three_fourth_l3110_311025

theorem nearest_integer_to_two_plus_sqrt_three_fourth (ε : ℝ) (hε : ε > 0) :
  ∃ (n : ℤ), n = 194 ∧ |((2 : ℝ) + Real.sqrt 3)^4 - (n : ℝ)| < (1/2 : ℝ) + ε :=
sorry

end nearest_integer_to_two_plus_sqrt_three_fourth_l3110_311025


namespace remainder_of_1234567_divided_by_257_l3110_311054

theorem remainder_of_1234567_divided_by_257 : 
  1234567 % 257 = 774 := by sorry

end remainder_of_1234567_divided_by_257_l3110_311054


namespace average_weight_increase_l3110_311082

/-- Proves that replacing a person weighing 65 kg with a person weighing 77 kg
    in a group of 8 people increases the average weight by 1.5 kg. -/
theorem average_weight_increase (initial_average : ℝ) :
  let initial_total := 8 * initial_average
  let new_total := initial_total - 65 + 77
  let new_average := new_total / 8
  new_average - initial_average = 1.5 := by
sorry

end average_weight_increase_l3110_311082


namespace fraction_in_lowest_terms_l3110_311076

theorem fraction_in_lowest_terms (n : ℤ) (h : Odd n) :
  Nat.gcd (Int.natAbs (2 * n + 2)) (Int.natAbs (3 * n + 2)) = 1 := by
  sorry

end fraction_in_lowest_terms_l3110_311076


namespace sequence_theorem_l3110_311014

def sequence_property (a : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, a n > 0) ∧ (∀ n : ℕ+, a (n + 1) + 1 / (a n) < 2)

theorem sequence_theorem (a : ℕ+ → ℝ) (h : sequence_property a) :
  (∀ n : ℕ+, a (n + 2) < a (n + 1) ∧ a (n + 1) < 2) ∧
  (∀ n : ℕ+, a n > 1) := by
  sorry

end sequence_theorem_l3110_311014


namespace certain_number_proof_l3110_311015

theorem certain_number_proof (N x : ℝ) (h1 : N / (1 + 3 / x) = 1) (h2 : x = 1) : N = 4 := by
  sorry

end certain_number_proof_l3110_311015


namespace grape_rate_calculation_l3110_311007

/-- The rate per kg for grapes -/
def grape_rate : ℝ := 70

/-- The amount of grapes purchased in kg -/
def grape_amount : ℝ := 3

/-- The rate per kg for mangoes -/
def mango_rate : ℝ := 55

/-- The amount of mangoes purchased in kg -/
def mango_amount : ℝ := 9

/-- The total amount paid -/
def total_paid : ℝ := 705

theorem grape_rate_calculation :
  grape_rate * grape_amount + mango_rate * mango_amount = total_paid :=
by sorry

end grape_rate_calculation_l3110_311007


namespace contractor_fine_calculation_l3110_311060

/-- Calculates the fine per day of absence for a contractor --/
def calculate_fine (total_days : ℕ) (pay_rate : ℚ) (total_received : ℚ) (days_absent : ℕ) : ℚ :=
  let days_worked := total_days - days_absent
  let amount_earned := days_worked * pay_rate
  (amount_earned - total_received) / days_absent

/-- Proves that the fine per day of absence is 7.5 given the contract conditions --/
theorem contractor_fine_calculation :
  calculate_fine 30 25 685 2 = 7.5 := by
  sorry

end contractor_fine_calculation_l3110_311060


namespace nonCongruentTrianglesCount_l3110_311080

-- Define the grid
def Grid := Fin 3 → Fin 3 → ℝ × ℝ

-- Define the grid with 0.5 unit spacing
def standardGrid : Grid :=
  λ i j => (0.5 * i.val, 0.5 * j.val)

-- Define a triangle as a tuple of three points
def Triangle := (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

-- Define congruence for triangles
def areCongruent (t1 t2 : Triangle) : Prop := sorry

-- Define a function to generate all possible triangles from the grid
def allTriangles (g : Grid) : List Triangle := sorry

-- Define a function to count non-congruent triangles
def countNonCongruentTriangles (triangles : List Triangle) : Nat := sorry

-- The main theorem
theorem nonCongruentTrianglesCount :
  countNonCongruentTriangles (allTriangles standardGrid) = 9 := by sorry

end nonCongruentTrianglesCount_l3110_311080


namespace sum_of_squares_on_sides_l3110_311005

/-- Given a triangle XYZ with side XZ = 12 units and perpendicular height from Y to XZ being 5 units,
    the sum of the areas of squares on sides XY and YZ is 122 square units. -/
theorem sum_of_squares_on_sides (X Y Z : ℝ × ℝ) : 
  let XZ : ℝ := Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2)
  let height : ℝ := 5
  let XY : ℝ := Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  let YZ : ℝ := Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2)
  XZ = 12 →
  (∃ D : ℝ × ℝ, (D.1 - X.1) * (Z.2 - X.2) = (Z.1 - X.1) * (D.2 - X.2) ∧ 
                (Y.1 - D.1) * (Z.1 - X.1) = (X.2 - D.2) * (Z.2 - X.2) ∧
                Real.sqrt ((Y.1 - D.1)^2 + (Y.2 - D.2)^2) = height) →
  XY^2 + YZ^2 = 122 :=
by sorry

end sum_of_squares_on_sides_l3110_311005


namespace two_digit_sum_divisibility_l3110_311094

theorem two_digit_sum_divisibility (a b : Nat) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) :
  ∃ k : Int, (10 * a + b) + (10 * b + a) = 11 * k :=
by sorry

end two_digit_sum_divisibility_l3110_311094


namespace largest_number_in_set_l3110_311089

theorem largest_number_in_set (a : ℝ) (h : a = -3) :
  -3 * a = max (-3 * a) (max (5 * a) (max (24 / a) (max (a ^ 2) 1))) :=
by sorry

end largest_number_in_set_l3110_311089


namespace eggs_per_omelet_is_two_l3110_311039

/-- Represents the number of eggs per omelet for the Rotary Club's Omelet Breakfast. -/
def eggs_per_omelet : ℚ :=
  let small_children_tickets : ℕ := 53
  let older_children_tickets : ℕ := 35
  let adult_tickets : ℕ := 75
  let senior_tickets : ℕ := 37
  let small_children_omelets : ℚ := 0.5
  let older_children_omelets : ℚ := 1
  let adult_omelets : ℚ := 2
  let senior_omelets : ℚ := 1.5
  let extra_omelets : ℕ := 25
  let total_eggs : ℕ := 584
  let total_omelets : ℚ := small_children_tickets * small_children_omelets +
                           older_children_tickets * older_children_omelets +
                           adult_tickets * adult_omelets +
                           senior_tickets * senior_omelets +
                           extra_omelets
  total_eggs / total_omelets

/-- Theorem stating that the number of eggs per omelet is 2. -/
theorem eggs_per_omelet_is_two : eggs_per_omelet = 2 := by
  sorry

end eggs_per_omelet_is_two_l3110_311039


namespace three_chords_for_sixty_degrees_l3110_311011

/-- Represents a pair of concentric circles with chords drawn on the larger circle -/
structure ConcentricCirclesWithChords where
  /-- The measure of the angle formed by two adjacent chords at their intersection point -/
  chord_angle : ℝ
  /-- The number of chords needed to complete a full revolution -/
  num_chords : ℕ

/-- Theorem stating that for a 60° chord angle, 3 chords are needed to complete a revolution -/
theorem three_chords_for_sixty_degrees (circles : ConcentricCirclesWithChords) 
  (h : circles.chord_angle = 60) : circles.num_chords = 3 := by
  sorry

#check three_chords_for_sixty_degrees

end three_chords_for_sixty_degrees_l3110_311011


namespace kindergarten_tissue_problem_l3110_311079

theorem kindergarten_tissue_problem :
  ∀ (group1 : ℕ), 
    (group1 * 40 + 10 * 40 + 11 * 40 = 1200) → 
    group1 = 9 := by
  sorry

end kindergarten_tissue_problem_l3110_311079


namespace remainder_2024_3047_mod_800_l3110_311066

theorem remainder_2024_3047_mod_800 : (2024 * 3047) % 800 = 728 := by
  sorry

end remainder_2024_3047_mod_800_l3110_311066


namespace simplify_expression_l3110_311095

theorem simplify_expression (a b : ℝ) (h1 : a + b ≠ 0) (h2 : a - 2*b ≠ 0) (h3 : a^2 - b^2 ≠ 0) (h4 : a^2 - 4*a*b + 4*b^2 ≠ 0) :
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) := by
  sorry

end simplify_expression_l3110_311095


namespace leading_coefficient_is_negative_seven_l3110_311055

def polynomial (x : ℝ) : ℝ := -3 * (x^4 - 2*x^3 + 3*x) + 8 * (x^4 + 5) - 4 * (3*x^4 + x^3 + 1)

theorem leading_coefficient_is_negative_seven :
  ∃ (f : ℝ → ℝ) (a : ℝ), a ≠ 0 ∧ (∀ x, polynomial x = a * x^4 + f x) ∧ a = -7 := by
  sorry

end leading_coefficient_is_negative_seven_l3110_311055


namespace spring_length_formula_l3110_311090

/-- Spring scale properties -/
structure SpringScale where
  initialLength : ℝ
  extensionRate : ℝ

/-- The analytical expression for the total length of a spring -/
def totalLength (s : SpringScale) (mass : ℝ) : ℝ :=
  s.initialLength + s.extensionRate * mass

/-- Theorem: The analytical expression for the total length of the spring is y = 10 + 2x -/
theorem spring_length_formula (s : SpringScale) (mass : ℝ) :
  s.initialLength = 10 ∧ s.extensionRate = 2 →
  totalLength s mass = 10 + 2 * mass := by
  sorry

end spring_length_formula_l3110_311090


namespace fourth_ball_black_probability_l3110_311031

theorem fourth_ball_black_probability 
  (total_balls : Nat) 
  (black_balls : Nat) 
  (red_balls : Nat) 
  (h1 : total_balls = black_balls + red_balls)
  (h2 : black_balls = 4)
  (h3 : red_balls = 4) :
  (black_balls : ℚ) / total_balls = 1 / 2 := by
  sorry

end fourth_ball_black_probability_l3110_311031


namespace water_remaining_l3110_311019

theorem water_remaining (total : ℚ) (used : ℚ) (h1 : total = 3) (h2 : used = 4/3) :
  total - used = 5/3 := by
  sorry

end water_remaining_l3110_311019


namespace pool_capacity_l3110_311018

theorem pool_capacity (C : ℝ) 
  (h1 : 0.4 * C + 300 = 0.7 * C)  -- Adding 300 gallons fills to 70%
  (h2 : 300 = 0.3 * (0.4 * C))    -- 300 gallons is a 30% increase
  : C = 1000 := by
sorry

end pool_capacity_l3110_311018


namespace expression_simplification_l3110_311002

theorem expression_simplification (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let num := a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)
  let den := a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)
  num / den = a + b + c := by
  sorry

end expression_simplification_l3110_311002


namespace jacob_phoebe_age_fraction_l3110_311063

/-- Represents the ages and relationships of Rehana, Phoebe, and Jacob -/
structure AgeRelationship where
  rehana_current_age : ℕ
  jacob_current_age : ℕ
  years_until_comparison : ℕ
  rehana_phoebe_ratio : ℕ

/-- The fraction of Phoebe's age that Jacob's age represents -/
def age_fraction (ar : AgeRelationship) : ℚ :=
  ar.jacob_current_age / (ar.rehana_current_age + ar.years_until_comparison - ar.years_until_comparison * ar.rehana_phoebe_ratio)

/-- Theorem stating that given the conditions, Jacob's age is 3/5 of Phoebe's age -/
theorem jacob_phoebe_age_fraction :
  ∀ (ar : AgeRelationship),
  ar.rehana_current_age = 25 →
  ar.jacob_current_age = 3 →
  ar.years_until_comparison = 5 →
  ar.rehana_phoebe_ratio = 3 →
  age_fraction ar = 3 / 5 := by
  sorry

end jacob_phoebe_age_fraction_l3110_311063


namespace z_squared_in_second_quadrant_l3110_311010

theorem z_squared_in_second_quadrant :
  let z : ℂ := Complex.exp (75 * π / 180 * Complex.I)
  (z^2).re < 0 ∧ (z^2).im > 0 :=
by sorry

end z_squared_in_second_quadrant_l3110_311010
